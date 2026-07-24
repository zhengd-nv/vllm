# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel build + symmetric-memory state + op factory for the allreduce-latency
communicator adapter.

This is a TRIMMED vendoring of the pcie-allreduce upstream
``src/bench_symm_allreduce.py``. Only the pieces the vLLM adapter actually
uses are kept:

- ``build_extension`` / ``signal_elems`` / ``alloc_symm_tensor`` /
  ``alloc_state`` -- reproduced verbatim from upstream (proven symmetric-memory
  buffer layout; do not re-derive sizes).
- ``make_op`` -- reduced to the 3 P1-winner algorithms only:
  ``custom_push_oneshot`` / ``custom_fast_oneshot`` / ``custom_twostage_fast``.

The full upstream file (all algorithms, NCCL/SGLang/CustomAllreduce backends,
argparse/CSV benchmark driver) is intentionally NOT carried into vLLM. The
standalone benchmark still lives at
``tmp/pcie-allreduce/allreduce-latency/src/bench_symm_allreduce.py`` for P1-style
tuning. The CUDA source (``symm_allreduce_ext.cu``) is unchanged and
byte-identical to upstream.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.utils.cpp_extension import load


def build_extension(source: Path, build_dir: Path, verbose: bool) -> Any:
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name="symm_allreduce_ext",
        sources=[str(source)],
        build_directory=str(build_dir),
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-lineinfo",
        ],
        verbose=verbose,
    )


def signal_elems(max_blocks: int, world_size: int) -> int:
    phases = 8
    return phases * max_blocks * world_size + max_blocks


def alloc_symm_tensor(
    numel: int, dtype: torch.dtype, device: torch.device, group_name: str
) -> tuple[torch.Tensor, Any, torch.Tensor, list[int]]:
    tensor = symm_mem.empty((numel,), dtype=dtype, device=device)
    handle = symm_mem.rendezvous(tensor, group=group_name)
    ptrs = [
        handle.get_buffer(peer, (numel,), dtype, storage_offset=0).data_ptr()
        for peer in range(handle.world_size)
    ]
    ptr_tensor = torch.tensor(ptrs, dtype=torch.int64, device=device)
    return tensor, handle, ptr_tensor, [int(ptr) for ptr in ptrs]


def alloc_state(max_numel: int, max_blocks: int, dtype: torch.dtype,
                device: torch.device,
                group: dist.ProcessGroup) -> dict[str, Any]:
    group_name = group.group_name
    world_size = dist.get_world_size(group)
    input_symm, input_handle, input_ptrs, input_ptrs_host = alloc_symm_tensor(
        max_numel, dtype, device, group_name)
    tmp_elems = max_numel * max(2, world_size * 2)
    tmp_symm, tmp_handle, tmp_ptrs, tmp_ptrs_host = alloc_symm_tensor(
        tmp_elems, dtype, device, group_name)
    tmp_symm.zero_()
    elem_size = torch.empty((), dtype=dtype).element_size()
    pack_elems = 16 // elem_size
    flag_elems = (max_numel // pack_elems) * max(2, world_size * 2)
    flag_symm, flag_handle, flag_ptrs, flag_ptrs_host = alloc_symm_tensor(
        flag_elems, torch.int32, device, group_name)
    flag_symm.zero_()
    output_symm, output_handle, output_ptrs, output_ptrs_host = alloc_symm_tensor(
        max_numel, dtype, device, group_name)
    sig_numel = signal_elems(max_blocks, world_size)
    signal_symm = symm_mem.empty((sig_numel,), dtype=torch.int32, device=device)
    signal_symm.zero_()
    signal_handle = symm_mem.rendezvous(signal_symm, group=group_name)
    signal_ptrs = [
        signal_handle.get_buffer(peer, (sig_numel,), torch.int32,
                                 storage_offset=0).data_ptr()
        for peer in range(signal_handle.world_size)
    ]
    signal_ptrs_tensor = torch.tensor(signal_ptrs, dtype=torch.int64,
                                      device=device)
    push_epoch = torch.zeros((max_blocks,), dtype=torch.int32, device=device)
    dist.barrier(group=group)
    torch.cuda.synchronize(device)
    return {
        "input_symm": input_symm,
        "input_handle": input_handle,
        "input_mc_ptr": int(input_handle.multicast_ptr),
        "input_ptrs": input_ptrs,
        "input_ptrs_host": input_ptrs_host,
        "tmp_symm": tmp_symm,
        "tmp_handle": tmp_handle,
        "tmp_ptrs": tmp_ptrs,
        "tmp_ptrs_host": tmp_ptrs_host,
        "flag_symm": flag_symm,
        "flag_handle": flag_handle,
        "flag_ptrs": flag_ptrs,
        "flag_ptrs_host": flag_ptrs_host,
        "output_symm": output_symm,
        "output_handle": output_handle,
        "output_ptrs": output_ptrs,
        "output_ptrs_host": output_ptrs_host,
        "signal_symm": signal_symm,
        "signal_handle": signal_handle,
        "signal_ptrs": signal_ptrs_tensor,
        "push_epoch": push_epoch,
    }


def make_op(name: str, ext: Any, pynccl_comm: Any, state: dict[str, Any],
            rank: int, world_size: int, group: dist.ProcessGroup,
            max_blocks: int, blocks: int, threads: int,
            include_copy: bool, ipc_push_ar: Any | None = None,
            pdl_sync: bool = False, pdl_release: bool = False,
            ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build the launch closure for one of the 3 P1-winner algorithms.

    ``pynccl_comm``/``ipc_push_ar`` are accepted for signature compatibility
    with the upstream factory but are unused here (none of the 3 winners need
    an NCCL fallback comm or IPC handles).
    """
    del pynccl_comm, ipc_push_ar  # unused by the winner algorithms

    input_symm = state["input_symm"]

    def copy_input(inp: torch.Tensor, numel: int) -> None:
        input_symm[:numel].copy_(inp.reshape(-1))

    if name == "custom_push_oneshot":
        def op(inp: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            numel = inp.numel()
            ext.push_oneshot(inp, state["tmp_ptrs"], state["push_epoch"],
                             out, numel, rank, world_size, max_blocks, blocks,
                             threads, pdl_sync, pdl_release)
            return out
        return op

    if name == "custom_fast_oneshot":
        def op(inp: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            numel = inp.numel()
            if include_copy:
                copy_input(inp, numel)
            ext.fast_oneshot(state["input_ptrs"], state["signal_ptrs"], out,
                             numel, rank, world_size, max_blocks, blocks,
                             threads, pdl_sync, pdl_release)
            return out
        return op

    if name == "custom_twostage_fast":
        def op(inp: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            numel = inp.numel()
            if include_copy:
                copy_input(inp, numel)
            ext.twostage_fast(state["input_ptrs"], state["tmp_ptrs"],
                              state["signal_ptrs"], out, numel, rank,
                              world_size, max_blocks, blocks, threads,
                              pdl_sync, pdl_release)
            return out
        return op

    raise ValueError(
        f"allreduce-latency adapter supports only the P1-winner algorithms "
        f"(custom_push_oneshot/custom_fast_oneshot/custom_twostage_fast); "
        f"got {name!r}")
