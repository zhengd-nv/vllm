# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Policy schema + cache loader + env switches for the allreduce-latency
communicator adapter.

Ported from the pcie-allreduce upstream ``python/pcie_allreduce/policy.py``
(schema/loader kept byte-for-byte compatible so P1-generated ``policy.json``
files load unchanged). vLLM-specific env plumbing is added on top; we
deliberately read ``os.getenv`` here instead of registering entries in
``vllm/envs.py`` to keep the integration blast radius to this adapter.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Policy schema (compatible with pcie_allreduce/policy.py)
# --------------------------------------------------------------------------- #

DEFAULT_BATCH_GRID = (1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48,
                      52, 56, 60, 64)

DTYPE_ALIASES = {
    "bfloat16": "bf16",
    "torch.bfloat16": "bf16",
    "bf16": "bf16",
    "float16": "fp16",
    "torch.float16": "fp16",
    "half": "fp16",
    "fp16": "fp16",
    "float32": "fp32",
    "torch.float32": "fp32",
    "fp32": "fp32",
}

ALGORITHM_ALIASES = {
    "tp2_pair_push": "custom_ipc_v2_adaptive",
    "tp2_pair_pull": "custom_tp2_oneshot_pairfast",
    "tp4_flat_oneshot": "custom_fast_oneshot",
    "tp4_pair_tree": "custom_tree",
    "tp4_rsag": "custom_rsag",
    "tp8_hier_4x4_rsag": "custom_ipc_v2_adaptive",
    "tp8_uniform_flat": "custom_oneshot",
    "tp8_generic_tree": "custom_tree",
}


def normalize_dtype(dtype: Any) -> str:
    value = str(dtype).lower()
    value = value.removeprefix("torch.")
    return DTYPE_ALIASES.get(value, value)


def normalize_algorithm(algorithm: str) -> str:
    return ALGORITHM_ALIASES.get(algorithm, algorithm)


@dataclass(frozen=True)
class ArShape:
    tp: int
    hidden: int
    batch: int
    dtype: str = "bf16"

    def normalized(self) -> "ArShape":
        return ArShape(self.tp, self.hidden, self.batch,
                       normalize_dtype(self.dtype))

    @property
    def key(self) -> tuple[int, int, int, str]:
        shape = self.normalized()
        return shape.tp, shape.hidden, shape.batch, shape.dtype


@dataclass(frozen=True)
class LaunchPolicy:
    tp: int
    hidden: int
    batch: int
    dtype: str
    algorithm: str
    blocks: int
    threads: int
    vec_packs: int = 1
    latency_us: float | None = None
    comment: str = ""

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> "LaunchPolicy":
        return cls(
            tp=int(row["tp"]),
            hidden=int(row["hidden"]),
            batch=int(row["batch"]),
            dtype=normalize_dtype(row.get("dtype", "bf16")),
            algorithm=normalize_algorithm(str(row["algorithm"])),
            blocks=int(row["blocks"]),
            threads=int(row["threads"]),
            vec_packs=int(row.get("vec_packs", 1)),
            latency_us=(
                None if row.get("latency_us") in (None, "") else
                float(row["latency_us"])),
            comment=str(row.get("comment", "")),
        )

    @property
    def shape(self) -> ArShape:
        return ArShape(self.tp, self.hidden, self.batch, self.dtype)


class PolicyCache:
    def __init__(self, path: Path, data: dict[str, Any]):
        self.path = path
        self.data = data
        self.profile_name = str(data.get("profile_name", path.parent.name))
        self.fallback = str(data.get("fallback", "nccl"))
        rows = [LaunchPolicy.from_json(row) for row in data.get("rows", [])]
        self.rows = rows
        self._by_shape = {row.shape.key: row for row in rows}

    def find(self, shape: ArShape) -> LaunchPolicy | None:
        return self._by_shape.get(shape.key)

    @property
    def max_numel(self) -> int:
        if not self.rows:
            return 0
        return max(row.hidden * row.batch for row in self.rows)

    @property
    def max_blocks(self) -> int:
        if not self.rows:
            return 1
        return max(row.blocks for row in self.rows)

    @property
    def dtypes(self) -> set[str]:
        return {row.dtype for row in self.rows}


def resolve_policy_path(cache_dir: str | Path | None = None,
                        profile_name: str | None = None,
                        policy_path: str | Path | None = None) -> Path:
    if policy_path is not None:
        return Path(policy_path)
    if cache_dir is None:
        raise FileNotFoundError(
            "allreduce-latency: no config dir set; "
            "export VLLM_ALLREDUCE_LATENCY_CONFIG_DIR")
    root = Path(cache_dir)
    profiles = root / "profiles"
    if profile_name:
        return profiles / profile_name / "policy.json"

    candidates = sorted(profiles.glob("*/policy.json"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"no policy.json found under {profiles}")
    raise RuntimeError(
        f"multiple policy profiles under {profiles}; set "
        "VLLM_ALLREDUCE_LATENCY_PROFILE")


def load_policy_cache(cache_dir: str | Path | None = None,
                      profile_name: str | None = None,
                      policy_path: str | Path | None = None) -> PolicyCache:
    path = resolve_policy_path(cache_dir, profile_name, policy_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return PolicyCache(path, data)


# --------------------------------------------------------------------------- #
# vLLM env switches (os.getenv; not registered in vllm/envs.py on purpose)
# --------------------------------------------------------------------------- #

DEFAULT_MAX_SIZE_BYTES = 8 * 1024 * 1024  # 8 MiB


def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip() not in ("", "0", "false", "False")


def is_allreduce_latency_enabled() -> bool:
    """Master switch: VLLM_USE_ALLREDUCE_LATENCY=1."""
    return _env_bool("VLLM_USE_ALLREDUCE_LATENCY", "0")


@dataclass(frozen=True)
class ArLatencyOptions:
    config_dir: str | None
    profile_name: str | None
    policy_path: str | None
    source_path: str | None
    build_dir: str | None
    max_size_bytes: int
    enable_pdl: bool
    compile_verbose: bool


def build_options() -> ArLatencyOptions:
    """Assemble adapter options from VLLM_ALLREDUCE_LATENCY_* env vars."""
    return ArLatencyOptions(
        config_dir=os.getenv("VLLM_ALLREDUCE_LATENCY_CONFIG_DIR") or None,
        profile_name=os.getenv("VLLM_ALLREDUCE_LATENCY_PROFILE") or None,
        policy_path=os.getenv("VLLM_ALLREDUCE_LATENCY_POLICY_PATH") or None,
        source_path=os.getenv("VLLM_ALLREDUCE_LATENCY_SOURCE_PATH") or None,
        build_dir=os.getenv("VLLM_ALLREDUCE_LATENCY_BUILD_DIR") or None,
        max_size_bytes=int(
            os.getenv("VLLM_ALLREDUCE_LATENCY_MAX_SIZE_BYTES",
                      str(DEFAULT_MAX_SIZE_BYTES))),
        enable_pdl=_env_bool("VLLM_ALLREDUCE_LATENCY_ENABLE_PDL", "0"),
        compile_verbose=_env_bool(
            "VLLM_ALLREDUCE_LATENCY_VERBOSE_BUILD", "0"),
    )
