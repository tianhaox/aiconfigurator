from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer

from jit_rt import flashinfer_kernels
from jit_rt.config import ConfigError, RuntimeSpec
from jit_rt.runtime import GenerateResult


@dataclass
class CudaGraphStep:
    graph: torch.cuda.CUDAGraph
    token: torch.Tensor
    logits: torch.Tensor
    paged_context: PagedDecodeGraphContext | None = None

    def replay(self, token_id: int) -> torch.Tensor:
        self.token.fill_(token_id)
        self.graph.replay()
        return self.logits


@dataclass
class PagedDecodeGraphContext:
    workspace: torch.Tensor
    indptr: torch.Tensor
    indices: torch.Tensor
    last_page_len: torch.Tensor
    out: torch.Tensor
    wrapper: Any


class Qwen3TorchRuntime:
    def __init__(self, spec: RuntimeSpec, model_dir: Path, state: dict[str, torch.Tensor]) -> None:
        self.spec = spec
        self.model_dir = model_dir
        self.device = torch.device(spec.runtime.device)
        self.dtype = _torch_dtype(spec.runtime.dtype)
        self.state = state
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

        model = spec.model
        cache_shape = (
            model.num_layers,
            spec.runtime.max_seq_len,
            model.resolved_num_key_value_heads,
            model.resolved_head_dim,
        )
        self.key_cache = torch.zeros(cache_shape, dtype=self.dtype, device=self.device)
        self.value_cache = torch.zeros(cache_shape, dtype=self.dtype, device=self.device)
        self.page_size = 16
        self.max_pages = math.ceil(spec.runtime.max_seq_len / self.page_size)
        self.paged_kv_cache = torch.zeros(
            (
                model.num_layers,
                self.max_pages,
                2,
                self.page_size,
                model.resolved_num_key_value_heads,
                model.resolved_head_dim,
            ),
            dtype=self.dtype,
            device=self.device,
        )
        self.inv_freq = _build_inv_freq(model.resolved_head_dim, model.rope_theta, self.device)
        self.num_key_value_groups = model.num_heads // model.resolved_num_key_value_heads
        self.scaling = 1.0 / math.sqrt(model.resolved_head_dim)
        self.use_flashinfer = spec.features.attention == "qwen3_gqa_flashinfer_single_decode"
        self.zero_bias = self._make_zero_bias()
        self.rope_single_indptr = torch.tensor([0, 1], dtype=torch.int32, device=self.device)
        self.rope_chunk_indptr = torch.empty((2,), dtype=torch.int32, device=self.device)
        self.rope_chunk_indptr[0] = 0
        self.rope_offsets = torch.empty((1,), dtype=torch.int32, device=self.device)
        self.cuda_graph_steps: dict[tuple[int, bool], CudaGraphStep] = {}
        self.use_paged_decode = False
        self.paged_decode_workspace = torch.empty((128 * 1024 * 1024,), dtype=torch.uint8, device=self.device)
        self.paged_decode_indptr = torch.empty((2,), dtype=torch.int32, device=self.device)
        self.paged_decode_indices = torch.arange(self.max_pages, dtype=torch.int32, device=self.device)
        self.paged_decode_last_page_len = torch.empty((1,), dtype=torch.int32, device=self.device)
        self.paged_decode_wrapper = None
        self.active_paged_decode_graph_context: PagedDecodeGraphContext | None = None

    @classmethod
    def from_spec(cls, spec: RuntimeSpec) -> Qwen3TorchRuntime:
        spec.validate()
        _validate_device(spec)
        model_dir = _resolve_model_dir(spec)
        _validate_hf_config(spec, model_dir)
        weights_path = _resolve_weights_path(spec, model_dir)
        state = _load_state(weights_path, spec)
        return cls(spec, model_dir, state)

    def encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode_tokens(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def enable_paged_decode(self) -> None:
        if not self.use_flashinfer:
            raise ConfigError("paged decode requires FlashInfer kernels")
        import flashinfer

        self.use_paged_decode = True
        if self.paged_decode_wrapper is None:
            self.paged_decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self.paged_decode_workspace,
                "NHD",
                use_tensor_cores=False,
            )

    @torch.inference_mode()
    def generate(self, prompt_token_ids: list[int], max_new_tokens: int) -> GenerateResult:
        self._validate_request(prompt_token_ids, max_new_tokens)
        self._reset_kv_cache()

        position = 0
        logits: torch.Tensor | None = None
        for token_id in prompt_token_ids:
            logits = self._step(token_id, position)
            position += 1

        if logits is None:
            raise ConfigError("prompt must contain at least one token")

        generated: list[int] = []
        for _ in range(max_new_tokens):
            next_token_id = int(torch.argmax(logits).item())
            generated.append(next_token_id)
            logits = self._step(next_token_id, position)
            position += 1

        return GenerateResult(
            prompt_token_ids=list(prompt_token_ids),
            generated_token_ids=generated,
            token_ids=list(prompt_token_ids) + generated,
        )

    @torch.inference_mode()
    def generate_piecewise_prefill(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        piece_size: int,
    ) -> GenerateResult:
        self._validate_request(prompt_token_ids, max_new_tokens)
        logits, position = self._prefill_piecewise(prompt_token_ids, piece_size)

        generated: list[int] = []
        for _ in range(max_new_tokens):
            next_token_id = int(torch.argmax(logits).item())
            generated.append(next_token_id)
            logits = self._step(next_token_id, position)
            position += 1

        return GenerateResult(
            prompt_token_ids=list(prompt_token_ids),
            generated_token_ids=generated,
            token_ids=list(prompt_token_ids) + generated,
        )

    @torch.inference_mode()
    def generate_cuda_graph(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        prefill_piece_size: int | None = None,
        paged_decode: bool = False,
    ) -> GenerateResult:
        if self.device.type != "cuda":
            raise ConfigError("generate_cuda_graph requires runtime.device=cuda")
        if paged_decode and not self.use_flashinfer:
            raise ConfigError("paged CUDA graph decode requires FlashInfer kernels")
        self._validate_request(prompt_token_ids, max_new_tokens)
        self._prepare_cuda_graph_steps(len(prompt_token_ids) + max_new_tokens, paged_decode=paged_decode)

        if prefill_piece_size is not None:
            logits, position = self._prefill_piecewise(prompt_token_ids, prefill_piece_size)
        else:
            self._reset_kv_cache()
            position = 0
            logits: torch.Tensor | None = None
            for token_id in prompt_token_ids:
                logits = self.cuda_graph_steps[(position, paged_decode)].replay(token_id)
                position += 1
            if logits is None:
                raise ConfigError("prompt must contain at least one token")

        generated: list[int] = []
        for _ in range(max_new_tokens):
            next_token_id = int(torch.argmax(logits).item())
            generated.append(next_token_id)
            logits = self.cuda_graph_steps[(position, paged_decode)].replay(next_token_id)
            position += 1

        return GenerateResult(
            prompt_token_ids=list(prompt_token_ids),
            generated_token_ids=generated,
            token_ids=list(prompt_token_ids) + generated,
        )

    @torch.inference_mode()
    def cuda_graph_probe(self, token_id: int) -> int:
        if self.device.type != "cuda":
            raise ConfigError("cuda_graph_probe requires runtime.device=cuda")
        token = torch.tensor([token_id], dtype=torch.long, device=self.device)
        static_logits = torch.empty((self.spec.model.vocab_size,), dtype=self.dtype, device=self.device)

        self._reset_kv_cache()
        for _ in range(3):
            static_logits.copy_(self._step_from_token_tensor(token, position=0))
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        self._reset_kv_cache()
        with torch.cuda.graph(graph):
            static_logits.copy_(self._step_from_token_tensor(token, position=0))

        graph.replay()
        torch.cuda.synchronize()
        return int(torch.argmax(static_logits).item())

    def _prefill_piecewise(self, prompt_token_ids: list[int], piece_size: int) -> tuple[torch.Tensor, int]:
        if not self.use_flashinfer:
            raise ConfigError("piecewise prefill requires FlashInfer kernels")
        if piece_size <= 0:
            raise ConfigError("prefill piece size must be positive")

        self._reset_kv_cache()
        logits: torch.Tensor | None = None
        position = 0
        for start in range(0, len(prompt_token_ids), piece_size):
            end = min(start + piece_size, len(prompt_token_ids))
            chunk_tokens = torch.tensor(prompt_token_ids[start:end], dtype=torch.long, device=self.device)
            logits = self._prefill_chunk(chunk_tokens, start)
            position = end

        if logits is None:
            raise ConfigError("prompt must contain at least one token")
        return logits, position

    def _prefill_chunk(self, token_ids: torch.Tensor, start_position: int) -> torch.Tensor:
        model = self.spec.model
        x = F.embedding(token_ids, self.state["model.embed_tokens.weight"])
        chunk_len = token_ids.shape[0]
        end_position = start_position + chunk_len
        for layer_idx in range(model.num_layers):
            prefix = f"model.layers.{layer_idx}"
            attn_input = self._rms_norm(x, self.state[f"{prefix}.input_layernorm.weight"])

            q = self._linear(attn_input, self.state[f"{prefix}.self_attn.q_proj.weight"])
            k = self._linear(attn_input, self.state[f"{prefix}.self_attn.k_proj.weight"])
            v = self._linear(attn_input, self.state[f"{prefix}.self_attn.v_proj.weight"])

            q = q.view(chunk_len, model.num_heads, model.resolved_head_dim)
            k = k.view(chunk_len, model.resolved_num_key_value_heads, model.resolved_head_dim)
            v = v.view(chunk_len, model.resolved_num_key_value_heads, model.resolved_head_dim)

            q = self._rms_norm(q, self.state[f"{prefix}.self_attn.q_norm.weight"])
            k = self._rms_norm(k, self.state[f"{prefix}.self_attn.k_norm.weight"])
            q, k = self._apply_rope_chunk(q, k, start_position)

            self._write_kv_chunk(layer_idx, start_position, k, v)
            attn_out = flashinfer_kernels.single_prefill_gqa(
                q,
                self.key_cache[layer_idx, :end_position],
                self.value_cache[layer_idx, :end_position],
                self.scaling,
            ).reshape(chunk_len, -1)
            x = x + self._linear(attn_out, self.state[f"{prefix}.self_attn.o_proj.weight"])

            ffn_input = self._rms_norm(x, self.state[f"{prefix}.post_attention_layernorm.weight"])
            gate = self._linear(ffn_input, self.state[f"{prefix}.mlp.gate_proj.weight"])
            up = self._linear(ffn_input, self.state[f"{prefix}.mlp.up_proj.weight"])
            x = x + self._linear(self._swiglu(gate, up), self.state[f"{prefix}.mlp.down_proj.weight"])

        x = self._rms_norm(x, self.state["model.norm.weight"])
        return self._linear(x[-1], self.state["lm_head.weight"])

    def _prepare_cuda_graph_steps(self, total_tokens: int, paged_decode: bool = False) -> None:
        missing_positions = [
            position for position in range(total_tokens) if (position, paged_decode) not in self.cuda_graph_steps
        ]
        if not missing_positions:
            return

        self._reset_kv_cache()
        for position in missing_positions:
            self.cuda_graph_steps[(position, paged_decode)] = self._capture_cuda_graph_step(
                position,
                paged_decode=paged_decode,
            )
        self._reset_kv_cache()

    def _capture_cuda_graph_step(self, position: int, paged_decode: bool = False) -> CudaGraphStep:
        token = torch.empty((1,), dtype=torch.long, device=self.device)
        token.fill_(0)
        logits = torch.empty((self.spec.model.vocab_size,), dtype=self.dtype, device=self.device)
        paged_context = self._make_paged_decode_graph_context(position) if paged_decode else None

        previous_context = self.active_paged_decode_graph_context
        self.active_paged_decode_graph_context = paged_context
        try:
            for _ in range(2):
                logits.copy_(self._step_from_token_tensor(token, position))
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                logits.copy_(self._step_from_token_tensor(token, position))
            return CudaGraphStep(graph=graph, token=token, logits=logits, paged_context=paged_context)
        finally:
            self.active_paged_decode_graph_context = previous_context

    def _validate_request(self, prompt_token_ids: list[int], max_new_tokens: int) -> None:
        if not prompt_token_ids:
            raise ConfigError("prompt must contain at least one token")
        if max_new_tokens < 0:
            raise ConfigError("max_new_tokens must be non-negative")
        total_tokens = len(prompt_token_ids) + max_new_tokens
        if total_tokens > self.spec.runtime.max_seq_len:
            raise ConfigError(
                f"request needs {total_tokens} tokens, max_seq_len is {self.spec.runtime.max_seq_len}",
            )
        vocab_size = self.spec.model.vocab_size
        for token_id in prompt_token_ids:
            if token_id < 0 or token_id >= vocab_size:
                raise ConfigError(f"token id {token_id} is outside vocab range [0, {vocab_size})")

    def _step(self, token_id: int, position: int) -> torch.Tensor:
        token = torch.tensor([token_id], dtype=torch.long, device=self.device)
        return self._step_from_token_tensor(token, position)

    def _step_from_token_tensor(self, token: torch.Tensor, position: int) -> torch.Tensor:
        model = self.spec.model
        x = F.embedding(token, self.state["model.embed_tokens.weight"]).squeeze(0)
        if self.use_paged_decode and self.active_paged_decode_graph_context is None:
            self._plan_paged_decode(position)
        for layer_idx in range(model.num_layers):
            prefix = f"model.layers.{layer_idx}"
            attn_input = self._rms_norm(x, self.state[f"{prefix}.input_layernorm.weight"])

            q = self._linear(attn_input, self.state[f"{prefix}.self_attn.q_proj.weight"])
            k = self._linear(attn_input, self.state[f"{prefix}.self_attn.k_proj.weight"])
            v = self._linear(attn_input, self.state[f"{prefix}.self_attn.v_proj.weight"])

            q = q.view(model.num_heads, model.resolved_head_dim)
            k = k.view(model.resolved_num_key_value_heads, model.resolved_head_dim)
            v = v.view(model.resolved_num_key_value_heads, model.resolved_head_dim)

            q = self._rms_norm(q, self.state[f"{prefix}.self_attn.q_norm.weight"])
            k = self._rms_norm(k, self.state[f"{prefix}.self_attn.k_norm.weight"])
            q, k = self._apply_rope(q, k, position)

            self._write_kv(layer_idx, position, k, v)
            attn_out = self._attention(layer_idx, position, q)
            x = x + self._linear(attn_out, self.state[f"{prefix}.self_attn.o_proj.weight"])

            ffn_input = self._rms_norm(x, self.state[f"{prefix}.post_attention_layernorm.weight"])
            gate = self._linear(ffn_input, self.state[f"{prefix}.mlp.gate_proj.weight"])
            up = self._linear(ffn_input, self.state[f"{prefix}.mlp.up_proj.weight"])
            x = x + self._linear(self._swiglu(gate, up), self.state[f"{prefix}.mlp.down_proj.weight"])

        x = self._rms_norm(x, self.state["model.norm.weight"])
        return self._linear(x, self.state["lm_head.weight"])

    def _attention(self, layer_idx: int, position: int, q: torch.Tensor) -> torch.Tensor:
        seq_len = position + 1
        keys = self.key_cache[layer_idx, :seq_len]
        values = self.value_cache[layer_idx, :seq_len]
        if self.use_flashinfer:
            if self.active_paged_decode_graph_context is not None:
                context = self.active_paged_decode_graph_context
                out = context.wrapper.run(
                    q.unsqueeze(0),
                    self.paged_kv_cache[layer_idx],
                    out=context.out,
                    enable_pdl=False,
                )
                return out.reshape(-1)
            if self.use_paged_decode:
                if self.paged_decode_wrapper is None:
                    raise ConfigError("paged decode wrapper is not initialized")
                out = self.paged_decode_wrapper.run(
                    q.unsqueeze(0),
                    self.paged_kv_cache[layer_idx],
                    enable_pdl=False,
                )
                return out.reshape(-1)
            return flashinfer_kernels.single_decode_gqa(q, keys, values, self.scaling).reshape(-1)
        return torch_reference_attention(q, keys, values, self.num_key_value_groups, self.scaling)

    def _reset_kv_cache(self) -> None:
        self.key_cache.zero_()
        self.value_cache.zero_()
        self.paged_kv_cache.zero_()

    def _write_kv(self, layer_idx: int, position: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self.key_cache[layer_idx, position] = k
        self.value_cache[layer_idx, position] = v
        page_idx = position // self.page_size
        page_offset = position % self.page_size
        self.paged_kv_cache[layer_idx, page_idx, 0, page_offset] = k
        self.paged_kv_cache[layer_idx, page_idx, 1, page_offset] = v

    def _write_kv_chunk(self, layer_idx: int, start_position: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self.key_cache[layer_idx, start_position : start_position + k.shape[0]] = k
        self.value_cache[layer_idx, start_position : start_position + v.shape[0]] = v
        for offset in range(k.shape[0]):
            self._write_kv(layer_idx, start_position + offset, k[offset], v[offset])

    def _plan_paged_decode(self, position: int) -> None:
        if self.paged_decode_wrapper is None:
            raise ConfigError("paged decode wrapper is not initialized")
        seq_len = position + 1
        pages_used = math.ceil(seq_len / self.page_size)
        last_page_len = seq_len - ((pages_used - 1) * self.page_size)
        self.paged_decode_indptr[0] = 0
        self.paged_decode_indptr[1] = pages_used
        self.paged_decode_last_page_len[0] = last_page_len
        self.paged_decode_wrapper.plan(
            self.paged_decode_indptr,
            self.paged_decode_indices[:pages_used],
            self.paged_decode_last_page_len,
            self.spec.model.num_heads,
            self.spec.model.resolved_num_key_value_heads,
            self.spec.model.resolved_head_dim,
            self.page_size,
            pos_encoding_mode="NONE",
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
            o_data_type=self.dtype,
            sm_scale=self.scaling,
            disable_split_kv=True,
        )

    def _make_paged_decode_graph_context(self, position: int) -> PagedDecodeGraphContext:
        import flashinfer

        seq_len = position + 1
        pages_used = math.ceil(seq_len / self.page_size)
        last_page_len = seq_len - ((pages_used - 1) * self.page_size)
        workspace = torch.empty((128 * 1024 * 1024,), dtype=torch.uint8, device=self.device)
        indptr = torch.tensor([0, pages_used], dtype=torch.int32, device=self.device)
        indices = torch.arange(self.max_pages, dtype=torch.int32, device=self.device)
        last_page_len_tensor = torch.tensor([last_page_len], dtype=torch.int32, device=self.device)
        out = torch.empty(
            (1, self.spec.model.num_heads, self.spec.model.resolved_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        wrapper = flashinfer.CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
            workspace,
            indptr,
            indices,
            last_page_len_tensor,
            "NHD",
            use_tensor_cores=False,
        )
        wrapper.plan(
            indptr,
            indices[:pages_used],
            last_page_len_tensor,
            self.spec.model.num_heads,
            self.spec.model.resolved_num_key_value_heads,
            self.spec.model.resolved_head_dim,
            self.page_size,
            pos_encoding_mode="NONE",
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
            o_data_type=self.dtype,
            sm_scale=self.scaling,
            disable_split_kv=True,
        )
        return PagedDecodeGraphContext(
            workspace=workspace,
            indptr=indptr,
            indices=indices,
            last_page_len=last_page_len_tensor,
            out=out,
            wrapper=wrapper,
        )

    def _linear(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if self.use_flashinfer:
            return flashinfer_kernels.tiny_linear(x, weight, self.zero_bias[weight.shape[0]])
        return F.linear(x, weight)

    def _rms_norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if self.use_flashinfer:
            return flashinfer_kernels.rmsnorm(x, weight, self.spec.model.rms_norm_eps)
        return rms_norm(x, weight, self.spec.model.rms_norm_eps)

    def _swiglu(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if self.use_flashinfer:
            return flashinfer_kernels.swiglu(gate, up)
        return F.silu(gate) * up

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_flashinfer:
            self.rope_offsets.fill_(position)
            return flashinfer_kernels.apply_rope_single(
                q,
                k,
                self.rope_single_indptr,
                self.rope_offsets,
                self.spec.model.rope_theta,
            )
        return apply_rotary(q, k, self.inv_freq, position)

    def _apply_rope_chunk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        start_position: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.rope_offsets.fill_(start_position)
        self.rope_chunk_indptr[1].fill_(q.shape[0])
        return flashinfer_kernels.apply_rope_chunk(
            q,
            k,
            self.rope_chunk_indptr,
            self.rope_offsets,
            self.spec.model.rope_theta,
        )

    def _make_zero_bias(self) -> dict[int, torch.Tensor]:
        if not self.use_flashinfer:
            return {}
        out_features = {
            weight.shape[0] for key, weight in self.state.items() if key.endswith(".weight") and weight.dim() == 2
        }
        return {size: torch.zeros(size, dtype=torch.bfloat16, device=self.device) for size in out_features}


def torch_reference_attention(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    num_key_value_groups: int,
    scaling: float,
) -> torch.Tensor:
    keys = repeat_kv(keys, num_key_value_groups)
    values = repeat_kv(values, num_key_value_groups)
    scores = torch.einsum("hd,hsd->hs", q, keys) * scaling
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    context = torch.einsum("hs,hsd->hd", probs, values)
    return context.reshape(-1)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = x.dtype
    x_float = x.to(torch.float32)
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x_float * torch.rsqrt(variance + eps)
    return (weight * x_norm.to(input_dtype)).to(input_dtype)


def apply_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    inv_freq: torch.Tensor,
    position: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = inv_freq * float(position)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=q.dtype)
    sin = emb.sin().to(dtype=q.dtype)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states.permute(1, 0, 2)
    seq_len, num_key_value_heads, head_dim = hidden_states.shape
    hidden_states = hidden_states.permute(1, 0, 2)
    hidden_states = hidden_states[:, None, :, :].expand(num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(num_key_value_heads * n_rep, seq_len, head_dim)


def _build_inv_freq(head_dim: int, rope_theta: float, device: torch.device) -> torch.Tensor:
    return 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))


def _torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ConfigError(f"unsupported torch dtype: {dtype}")


def _validate_device(spec: RuntimeSpec) -> None:
    if spec.runtime.device == "cuda":
        if not torch.cuda.is_available():
            raise ConfigError("runtime.device=cuda requested but torch.cuda.is_available() is false")
        if spec.target.sm is not None:
            major, minor = torch.cuda.get_device_capability(0)
            actual = f"{major}{minor}"
            if actual != spec.target.sm:
                raise ConfigError(f"manifest requires sm{spec.target.sm}, actual CUDA device is sm{actual}")
    elif spec.target.sm is not None:
        raise ConfigError("target.sm must be null when runtime.device is cpu")


def _resolve_model_dir(spec: RuntimeSpec) -> Path:
    if spec.weights.local_path:
        model_dir = Path(spec.weights.local_path).expanduser().resolve()
        if not model_dir.exists():
            raise ConfigError(f"weights.local_path does not exist: {model_dir}")
        return model_dir

    if not spec.weights.repo_id:
        raise ConfigError("weights.repo_id is required when weights.local_path is not set")

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    return Path(
        snapshot_download(
            spec.weights.repo_id,
            revision=spec.weights.revision,
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
            ],
        ),
    )


def _resolve_weights_path(spec: RuntimeSpec, model_dir: Path) -> Path:
    weights_path = model_dir / "model.safetensors"
    if weights_path.exists():
        return weights_path
    if not spec.weights.repo_id:
        raise ConfigError(f"model.safetensors not found in local path: {model_dir}")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    return Path(hf_hub_download(spec.weights.repo_id, "model.safetensors", revision=spec.weights.revision))


def _validate_hf_config(spec: RuntimeSpec, model_dir: Path) -> None:
    cfg = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    expected = {
        "model_type": "qwen3",
        "vocab_size": spec.model.vocab_size,
        "hidden_size": spec.model.hidden_size,
        "num_hidden_layers": spec.model.num_layers,
        "num_attention_heads": spec.model.num_heads,
        "num_key_value_heads": spec.model.resolved_num_key_value_heads,
        "head_dim": spec.model.resolved_head_dim,
        "intermediate_size": spec.model.intermediate_size,
    }
    for key, value in expected.items():
        actual = getattr(cfg, key)
        if actual != value:
            raise ConfigError(f"HF config mismatch for {key}: manifest={value}, hf={actual}")
    if float(cfg.rope_theta) != float(spec.model.rope_theta):
        raise ConfigError(f"HF config mismatch for rope_theta: manifest={spec.model.rope_theta}, hf={cfg.rope_theta}")
    if float(cfg.rms_norm_eps) != float(spec.model.rms_norm_eps):
        raise ConfigError(
            f"HF config mismatch for rms_norm_eps: manifest={spec.model.rms_norm_eps}, hf={cfg.rms_norm_eps}",
        )
    if cfg.sliding_window is not None or cfg.use_sliding_window:
        raise ConfigError("sliding-window attention is not supported")


def _load_state(weights_path: Path, spec: RuntimeSpec) -> dict[str, torch.Tensor]:
    if weights_path.stat().st_size < 1024 * 1024:
        raise ConfigError(f"model.safetensors is unexpectedly small: {weights_path}")
    state = load_file(str(weights_path), device=spec.runtime.device)
    dtype = _torch_dtype(spec.runtime.dtype)
    return {key: tensor.to(dtype=dtype) if tensor.is_floating_point() else tensor for key, tensor in state.items()}
