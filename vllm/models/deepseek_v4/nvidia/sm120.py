# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer SM120 sparse-MLA impl for DeepSeek-V4."""

from typing import TYPE_CHECKING, ClassVar, cast

import torch

from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.common.ops import (
    compute_global_topk_indices_and_lens,
)
from vllm.models.deepseek_v4.nvidia.flashmla import (
    DeepseekV4FlashMLASparseBackend,
    DeepseekV4SparseMLAAttentionImpl,
)
from vllm.utils.flashinfer import has_flashinfer_sparse_mla
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseMetadata
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.models.deepseek_v4.attention import DeepseekV4MLAAttention
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


_DECODE_MAX_TOKENS = 64
_DECODE_SPLIT_TILE = 64
_C128A_TOPK_ALIGNMENT = 128


def _cdiv(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def _decode_num_splits(topk: int, extra_topk: int = 0) -> int:
    return _cdiv(topk, _DECODE_SPLIT_TILE) + _cdiv(extra_topk, _DECODE_SPLIT_TILE)


def _max_decode_workspace_tokens(max_num_batched_tokens: int) -> int:
    return min(int(max_num_batched_tokens), _DECODE_MAX_TOKENS)


def _c128a_max_compressed(max_model_len: int, compress_ratio: int) -> int:
    return (
        _cdiv(
            _cdiv(max_model_len, compress_ratio),
            _C128A_TOPK_ALIGNMENT,
        )
        * _C128A_TOPK_ALIGNMENT
    )


def _get_decode_scratch(
    num_tokens: int,
    num_heads: int,
    d_v: int,
    topk: int,
    extra_topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_splits = _decode_num_splits(topk, extra_topk)
    mid_out, mid_lse = current_workspace_manager().get_simultaneous(
        ((num_tokens, num_heads, num_splits, d_v), torch.bfloat16),
        ((num_tokens, num_heads, num_splits), torch.float32),
    )
    return mid_out, mid_lse


class DeepseekV4FlashInferSM120SparseBackend(DeepseekV4FlashMLASparseBackend):
    """DeepSeek-V4 FlashInfer SM120 sparse-MLA backend."""

    @staticmethod
    def get_name() -> str:
        return "V4_FLASHINFER_MLA_SPARSE"

    @staticmethod
    def get_impl_cls() -> type["DeepseekV4FlashInferSM120SparseImpl"]:
        return DeepseekV4FlashInferSM120SparseImpl


class DeepseekV4FlashInferSM120SparseImpl(DeepseekV4SparseMLAAttentionImpl):
    """SM120 FlashInfer-wrapper-driven sparse-MLA impl for DeepSeek-V4."""

    backend_cls: ClassVar[type[AttentionBackend]] = (
        DeepseekV4FlashInferSM120SparseBackend
    )

    def __init__(self, layer: "DeepseekV4MLAAttention") -> None:
        super().__init__(layer)
        if not has_flashinfer_sparse_mla():
            raise RuntimeError(
                "DeepSeek V4 SM120 sparse MLA requires FlashInfer's "
                "sparse-sm120 MLA wrapper."
            )

        from flashinfer.mla import BatchMLAPagedAttentionWrapper

        wrapper_device = torch.device("cuda", torch.accelerator.current_device_index())
        self._sparse_mla_wrapper = BatchMLAPagedAttentionWrapper(
            torch.empty(1, dtype=torch.int8, device=wrapper_device),
            backend="sparse-sm120",
            max_num_tokens=layer.max_num_batched_tokens,
            max_num_heads=layer.padded_heads,
            d_v=512,
        )

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        if num_heads <= 16:
            return 16
        if num_heads <= 32:
            return 32
        if num_heads <= 64:
            return 64
        if num_heads <= 128:
            return 128
        raise ValueError(
            f"DeepseekV4 SM120 sparse MLA does not support {num_heads} heads "
            "(kernel requires h_q in {16, 32, 64, 128})."
        )

    def forward_mqa(  # type: ignore[override]
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        layer = self.layer
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            self._reserve_decode_workspace(layer)
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            FlashMLASparseMetadata | None, attn_metadata.get(layer.prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(layer.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = layer.compress_ratio <= 1
        # SWA-only layers (compress_ratio <= 1) don't have their own KV cache
        # allocation; layer.kv_cache may be empty after profiling cleanup.
        self_kv_cache = layer.kv_cache if not swa_only else None
        swa_kv_cache = layer.swa_cache_layer.kv_cache

        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            self._forward_prefill(
                layer=layer,
                q=q[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            self._forward_decode(
                layer=layer,
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    def _reserve_decode_workspace(self, layer: "DeepseekV4MLAAttention") -> None:
        if layer.compress_ratio <= 1:
            extra_topk = 0
        elif layer.compress_ratio == 4:
            assert layer.topk_indices_buffer is not None
            extra_topk = layer.topk_indices_buffer.shape[-1]
        elif layer.compress_ratio == 128:
            extra_topk = _c128a_max_compressed(
                layer.max_model_len,
                layer.compress_ratio,
            )
        else:
            raise ValueError(
                f"Unsupported compress_ratio={layer.compress_ratio}; "
                "expected 1, 4, or 128."
            )
        _get_decode_scratch(
            _max_decode_workspace_tokens(layer.max_num_batched_tokens),
            layer.padded_heads,
            512,
            layer.window_size,
            extra_topk,
        )

    def _forward_decode(
        self,
        layer: "DeepseekV4MLAAttention",
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,  # only used when compress_ratio > 1
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: FlashMLASparseMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        extra_sparse_indices = None
        extra_sparse_lengths = None
        if not swa_only:
            if attn_metadata is None:
                raise RuntimeError(
                    "Sparse MLA metadata is required for compressed layers."
                )
            if swa_metadata.is_valid_token is None:
                raise RuntimeError(
                    "SWA validity metadata is required for compressed layers."
                )
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if layer.compress_ratio == 4:
                # C4A: local indices differ per layer (filled by Indexer).
                if layer.topk_indices_buffer is None:
                    raise RuntimeError(
                        "C4A decode requires top-k indices from the indexer."
                    )
                block_size = attn_metadata.block_size // layer.compress_ratio
                global_indices, extra_sparse_lengths = (
                    compute_global_topk_indices_and_lens(
                        layer.topk_indices_buffer[:num_decode_tokens],
                        swa_metadata.token_to_req_indices,
                        attn_metadata.block_table[:num_decodes],
                        block_size,
                        is_valid,
                    )
                )
                extra_sparse_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                # C128A: pre-computed during metadata build.
                extra_sparse_indices = attn_metadata.c128a_global_decode_topk_indices
                extra_sparse_lengths = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens
        assert swa_indices is not None
        assert swa_lens is not None
        extra_topk = (
            extra_sparse_indices.shape[-1] if extra_sparse_indices is not None else 0
        )
        mid_out, mid_lse = _get_decode_scratch(
            num_decode_tokens,
            q.shape[1],
            output.shape[-1],
            swa_indices.shape[-1],
            extra_topk,
        )

        # The wrapper attends through generated sparse indices only.
        q = q.unsqueeze(1)
        swa_cache = layer.swa_cache_layer.kv_cache.unsqueeze(-2)
        if kv_cache is not None:
            kv_cache = kv_cache.unsqueeze(-2)

        assert self._sparse_mla_wrapper is not None, (
            "DeepseekV4FlashInferSM120SparseImpl requires FlashInfer's "
            "sparse-sm120 MLA wrapper to be available."
        )
        self._sparse_mla_wrapper.run_sparse_mla(
            q=q,
            kv_cache=swa_cache,
            sparse_indices=swa_indices,
            out=output,
            sm_scale=layer.scale,
            sparse_lengths=swa_lens,
            sinks=layer.attn_sink,
            extra_kv_cache=kv_cache if not swa_only else None,
            extra_sparse_indices=extra_sparse_indices,
            extra_sparse_lengths=extra_sparse_lengths,
            mid_out=mid_out,
            mid_lse=mid_lse,
        )

    def _forward_prefill(
        self,
        layer: "DeepseekV4MLAAttention",
        q: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        # `_dummy_run` passes synthetic non-None attn_metadata for swa-only
        # layers during cudagraph capture, so check compress_ratio directly.
        swa_only = layer.compress_ratio <= 1

        num_prefills = swa_metadata.num_prefills
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens

        # Derive prefill-local token offsets from the full query_start_loc_cpu.
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        assert query_start_loc_cpu is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        local_topk_indices: torch.Tensor | None
        if swa_only:
            local_topk_indices = None
        elif layer.compress_ratio == 4:
            if layer.topk_indices_buffer is None:
                raise RuntimeError(
                    "C4A prefill requires top-k indices from the indexer."
                )
            local_topk_indices = layer.topk_indices_buffer[
                num_decode_tokens : num_decode_tokens + num_prefill_tokens
            ]
        else:
            # C128A: pre-computed during metadata build.
            if attn_metadata is None:
                raise RuntimeError("C128A prefill metadata is missing.")
            local_topk_indices = attn_metadata.c128a_prefill_topk_indices

        extra_sparse_indices: torch.Tensor | None = None
        extra_sparse_lengths: torch.Tensor | None = None
        if local_topk_indices is not None:
            if attn_metadata is None:
                raise RuntimeError("C4A prefill metadata is missing.")
            if swa_metadata.token_to_req_indices is None:
                raise RuntimeError("C4A prefill request mapping is missing.")
            if swa_metadata.is_valid_token is None:
                raise RuntimeError("C4A prefill validity metadata is missing.")
            prefill_token_slice = slice(
                num_decode_tokens, num_decode_tokens + num_prefill_tokens
            )
            # FlashInfer prefill expects physical KV slots; keep padding rows
            # masked through the metadata validity mask.
            block_size = attn_metadata.block_size // layer.compress_ratio
            extra_sparse_indices, extra_sparse_lengths = (
                compute_global_topk_indices_and_lens(
                    local_topk_indices,
                    swa_metadata.token_to_req_indices[prefill_token_slice],
                    attn_metadata.block_table,
                    block_size,
                    swa_metadata.is_valid_token[prefill_token_slice],
                )
            )

        assert swa_metadata.prefill_swa_indices is not None
        assert swa_metadata.prefill_swa_lens is not None
        assert self._sparse_mla_wrapper is not None

        swa_kv_paged = swa_k_cache.unsqueeze(-2)
        if swa_only:
            extra_kv_paged = None
        else:
            if compressed_k_cache is None:
                raise RuntimeError(
                    "Compressed sparse MLA layers require their compressed KV cache."
                )
            extra_kv_paged = compressed_k_cache.unsqueeze(-2)

        num_chunks = (
            num_prefills + self.PREFILL_CHUNK_SIZE - 1
        ) // self.PREFILL_CHUNK_SIZE
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + self.PREFILL_CHUNK_SIZE, num_prefills)
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            extra_sparse_indices_chunk = (
                extra_sparse_indices[query_start:query_end]
                if extra_sparse_indices is not None
                else None
            )
            extra_sparse_lengths_chunk = (
                extra_sparse_lengths[query_start:query_end]
                if extra_sparse_lengths is not None
                else None
            )
            chunk_tokens = query_end - query_start

            mid_out = None
            mid_lse = None
            if chunk_tokens <= _DECODE_MAX_TOKENS:
                extra_topk = (
                    extra_sparse_indices_chunk.shape[-1]
                    if extra_sparse_indices_chunk is not None
                    else 0
                )
                mid_out, mid_lse = _get_decode_scratch(
                    chunk_tokens,
                    q.shape[1],
                    output.shape[-1],
                    swa_metadata.prefill_swa_indices.shape[-1],
                    extra_topk,
                )

            self._sparse_mla_wrapper.run_sparse_mla(
                q=q[query_start:query_end],
                kv_cache=swa_kv_paged,
                sparse_indices=swa_metadata.prefill_swa_indices[query_start:query_end],
                out=output[query_start:query_end],
                sm_scale=layer.scale,
                sparse_lengths=swa_metadata.prefill_swa_lens[query_start:query_end],
                sinks=layer.attn_sink,
                extra_kv_cache=extra_kv_paged,
                extra_sparse_indices=extra_sparse_indices_chunk,
                extra_sparse_lengths=extra_sparse_lengths_chunk,
                mid_out=mid_out,
                mid_lse=mid_lse,
            )
