# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM communicator adapter for the allreduce-latency kernel (SM120/PCIe).

Ported from the pcie-allreduce upstream ``python/pcie_allreduce/runtime.py``
(which only ships a SGLang adapter). It exposes the same interface as vLLM's
``CustomAllreduce`` -- ``should_custom_ar(input)`` and
``custom_all_reduce(input)`` -- so it can be dropped into the
``cuda_communicator.all_reduce`` dispatch chain just before ``ca_comm``.

Design (see project task.md P2):
- Only the 3 P1-winner algorithms are ever launched (``custom_push_oneshot``,
  ``custom_fast_oneshot``, ``custom_twostage_fast``). None use IPC handles, so
  this adapter does NOT build a pynccl fallback comm or exchange IPC handles;
  when the policy misses a shape we return ``None`` and let the caller fall
  through to pynccl.
- CUDA-graph safe: the kernels use device-side epoch slots + persistent
  symmetric-memory buffers allocated once at init, so no ``capture()`` /
  ``register_graph_buffers`` machinery is needed (unlike CustomAllreduce's
  IPC-meta path). P1 validated ``correct_all=True`` under ``mode=cuda_graph``.
- ``include_copy=True``: ``fast_oneshot``/``twostage_fast`` read a pre-filled
  symmetric input buffer, so live input is copied in on every call.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

from .allreduce_latency_config import (
    ArLatencyOptions,
    ArShape,
    LaunchPolicy,
    load_policy_cache,
)
from .allreduce_latency_ext import bench_symm_allreduce as bench

logger = init_logger(__name__)

DTYPE_TO_TORCH = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _torch_dtype(dtype: str) -> torch.dtype:
    try:
        return DTYPE_TO_TORCH[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype in policy: {dtype}") from exc


def _input_dtype_name(tensor: torch.Tensor) -> str:
    if tensor.dtype is torch.bfloat16:
        return "bf16"
    if tensor.dtype is torch.float16:
        return "fp16"
    if tensor.dtype is torch.float32:
        return "fp32"
    return str(tensor.dtype).removeprefix("torch.")


class AllReduceLatency:
    """allreduce-latency communicator with a CustomAllreduce-like interface."""

    def __init__(self, group: ProcessGroup, device: torch.device | int | str,
                 options: ArLatencyOptions):
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized first")

        self.disabled = True
        self.group = group
        self.options = options
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self.device = device
        torch.cuda.set_device(device)

        # ---- policy ----
        self.policy = load_policy_cache(options.config_dir,
                                        options.profile_name,
                                        options.policy_path)
        self.topology = self._load_topology()
        self._validate_policy()
        self._validate_topology()

        dtype = next(iter(self.policy.dtypes))
        self.dtype = _torch_dtype(dtype)
        self.max_numel = self.policy.max_numel
        self.max_size_bytes = min(
            options.max_size_bytes,
            self.max_numel * torch.empty((), dtype=self.dtype).element_size())
        self.max_blocks = self.policy.max_blocks
        self.pdl_sync = bool(options.enable_pdl)
        self.pdl_release = bool(options.enable_pdl)

        # ---- extension + symmetric-memory state ----
        source = Path(options.source_path) if options.source_path else (
            Path(__file__).with_name("allreduce_latency_ext")
            / "symm_allreduce_ext.cu")
        build_dir = Path(options.build_dir) if options.build_dir else (
            Path(__file__).with_name("allreduce_latency_ext") / "build")
        self.ext = bench.build_extension(source, build_dir,
                                         options.compile_verbose)
        self.state = bench.alloc_state(self.max_numel, self.max_blocks,
                                       self.dtype, self.device, group)
        self._op_cache: dict[tuple[str, int, int], Any] = {}

        self.disabled = False
        logger.info(
            "[allreduce-latency] initialized: profile=%s rows=%d tp=%d "
            "hidden=%s dtype=%s max_numel=%d max_blocks=%d pdl=%s",
            self.policy.profile_name, len(self.policy.rows), self.world_size,
            sorted({r.hidden for r in self.policy.rows}), dtype,
            self.max_numel, self.max_blocks, self.pdl_sync)

    # ----------------------------------------------------------------- init
    def _load_topology(self) -> dict[str, Any]:
        path = self.policy.path.parent / "topology.json"
        if not path.exists():
            return {}
        import json
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise RuntimeError(f"invalid topology file: {path}")
        return data

    def _validate_policy(self) -> None:
        schema_version = int(self.policy.data.get("schema_version", 1))
        if schema_version != 1:
            raise RuntimeError(
                f"unsupported policy schema_version={schema_version}: "
                f"{self.policy.path}")
        if not self.policy.rows:
            raise RuntimeError(f"empty policy: {self.policy.path}")
        world_sizes = {row.tp for row in self.policy.rows}
        if self.world_size not in world_sizes:
            raise RuntimeError(
                f"policy {self.policy.path} has TP sizes {sorted(world_sizes)},"
                f" but process group world_size is {self.world_size}")
        if len(self.policy.dtypes) != 1:
            raise RuntimeError(
                "first version supports one dtype per policy; got "
                f"{sorted(self.policy.dtypes)}")
        for row in self.policy.rows:
            self._validate_policy_row(row)

    def _validate_policy_row(self, row: LaunchPolicy) -> None:
        if row.tp not in (2, 4, 8):
            raise RuntimeError(f"unsupported TP in policy row: {row}")
        if row.hidden <= 0 or row.batch <= 0:
            raise RuntimeError(f"policy row must have positive shape: {row}")
        if row.dtype not in DTYPE_TO_TORCH:
            raise RuntimeError(f"unsupported dtype in policy row: {row}")
        if row.algorithm.startswith("custom_"):
            if row.blocks <= 0 or row.blocks > self.policy.max_blocks:
                raise RuntimeError(
                    f"policy row has invalid blocks {row.blocks}: {row}")
            if row.threads <= 0 or row.threads > 1024:
                raise RuntimeError(
                    f"policy row has invalid threads {row.threads}: {row}")

    def _validate_topology(self) -> None:
        """Optional deployment guard.

        The strict ``CUDA_VISIBLE_DEVICES`` comparison is opt-in
        (``VLLM_ALLREDUCE_LATENCY_STRICT_TOPO=1``) because vLLM V1 workers may
        remap the visible-device set, which would otherwise disable the adapter
        spuriously. Correctness of the *algorithm* is already guarded by the
        world_size + exact shape/dtype match in ``should_custom_ar``.
        """
        rank_order = self.topology.get("rank_order", [])
        if rank_order and len(rank_order) < self.world_size:
            raise RuntimeError(
                f"topology rank_order has {len(rank_order)} entries, but "
                f"process group world_size is {self.world_size}")

        strict = os.getenv(
            "VLLM_ALLREDUCE_LATENCY_STRICT_TOPO", "0").strip() not in (
                "", "0", "false", "False")
        cached = str(self.topology.get("cuda_visible_devices", ""))
        current = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cached and current and cached != current:
            msg = ("[allreduce-latency] policy topology cuda_visible_devices="
                   f"{cached!r} != current {current!r}")
            if strict:
                raise RuntimeError(msg)
            logger.warning("%s (ignored; set "
                           "VLLM_ALLREDUCE_LATENCY_STRICT_TOPO=1 to enforce)",
                           msg)

    # -------------------------------------------------------------- runtime
    def _shape_of(self, input: torch.Tensor) -> ArShape:
        hidden = int(input.shape[-1])
        batch = input.numel() // hidden if hidden else 0
        return ArShape(self.world_size, hidden, batch,
                       _input_dtype_name(input)).normalized()

    def should_custom_ar(self, input: torch.Tensor) -> bool:
        if self.disabled:
            return False
        if not input.is_cuda or input.device != self.device:
            return False
        if not input.is_contiguous():
            return False
        hidden = int(input.shape[-1])
        if hidden <= 0 or input.numel() % hidden != 0:
            return False
        numel = input.numel()
        if numel == 0 or numel > self.max_numel:
            return False
        if numel * input.element_size() % 16 != 0:
            return False
        if numel * input.element_size() > self.max_size_bytes:
            return False
        shape = self._shape_of(input)
        if _input_dtype_name(input) != next(iter(self.policy.dtypes)):
            return False
        return self.policy.find(shape) is not None

    def custom_all_reduce(self, input: torch.Tensor) -> torch.Tensor | None:
        """Out-of-place all-reduce. Returns None to signal caller fallback."""
        if not self.should_custom_ar(input):
            return None
        policy = self.policy.find(self._shape_of(input))
        if policy is None:
            return None
        op = self._get_op(policy)
        out = torch.empty_like(input)
        result = op(input, out)
        if result is out:
            return out
        # push_oneshot family reduces in-place / returns its own buffer.
        if result.data_ptr() == out.data_ptr():
            return out
        out.copy_(result)
        return out

    def _get_op(self, policy: LaunchPolicy):
        key = (policy.algorithm, policy.blocks, policy.threads)
        cached = self._op_cache.get(key)
        if cached is not None:
            return cached
        op = bench.make_op(
            policy.algorithm,
            self.ext,
            None,            # pynccl_comm: unused (no nccl fallback here)
            self.state,
            self.rank,
            self.world_size,
            self.group,
            self.max_blocks,
            policy.blocks,
            policy.threads,
            True,            # include_copy: live input -> symmetric buffer
            ipc_push_ar=None,
            pdl_sync=self.pdl_sync,
            pdl_release=self.pdl_release,
        )
        self._op_cache[key] = op
        return op

    def close(self) -> None:
        self._op_cache.clear()
        self.disabled = True
