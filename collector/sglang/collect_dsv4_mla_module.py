# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 compressed-attention module collector for SGLang.

This is intentionally separate from collect_mla_module.py.  DeepSeek-V4 HCA/CSA
uses SGLang's "compressed" backend and a different module signature, but the
external collector shape is kept close to AIC's existing MLA module collector:

    python collect_dsv4_mla_module.py --mode generation --attn-kind csa
    python collect_dsv4_mla_module.py --mode context --attn-kind hca --seq-lens 128

`--attn-kind` is the single public switch for SWA/CSA/HCA:

    swa -> compress_ratio=0
    csa -> compress_ratio=4
    hca -> compress_ratio=128

The benchmark measures CUDA Graph replay of `layer.self_attn(...)` only.  That
includes Q/KV projection, norm/rope, cache store, compressor, C4 indexer/topk
for CSA, and the final FlashMLA call; it does not include MHC pre/post or MLP.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import random
import shutil
import sys
import tempfile
import traceback
from collections.abc import Iterable
from importlib.metadata import version as get_version

import torch

# DSV4 local forks default to replacing small patched configs with packaged
# config_backup_small.json.  That is correct for serving smoke tests, but wrong
# for this collector because we deliberately patch `compress_ratios` to isolate
# SWA/CSA/HCA in a 1-layer model.
os.environ.setdefault("SGLANG_APPLY_CONFIG_BACKUP", "none")

try:
    from helper import benchmark_with_power, log_perf
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from helper import benchmark_with_power, log_perf


ATTN_KIND_TO_COMPRESS_RATIO = {
    "swa": 0,
    "csa": 4,
    "hca": 128,
}

COMPRESS_RATIO_TO_ATTN_KIND = {
    v: k for k, v in ATTN_KIND_TO_COMPRESS_RATIO.items()
}

DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Pro"
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


def _parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def _resolve_perf_path(output_path: str | None, default_name: str) -> str:
    if not output_path:
        return default_name
    if output_path.endswith(".txt"):
        return output_path
    os.makedirs(output_path, exist_ok=True)
    return os.path.join(output_path, default_name)


def _copy_non_weight_files(src_dir: str, dst_dir: str) -> None:
    for fname in os.listdir(src_dir):
        src_path = os.path.join(src_dir, fname)
        if not os.path.isfile(src_path):
            continue
        if fname.endswith(_WEIGHT_SUFFIXES) or fname == "config.json":
            continue
        dst_path = os.path.join(dst_dir, fname)
        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)


def _download_non_weight_model_files(model_id: str) -> tuple[str, dict]:
    from huggingface_hub import hf_hub_download, list_repo_files

    try:
        files = list_repo_files(model_id)
    except Exception:
        files = ["config.json"]

    config_file = None
    for fname in files:
        if fname.endswith(_WEIGHT_SUFFIXES):
            continue
        try:
            path = hf_hub_download(model_id, fname)
            if fname == "config.json":
                config_file = path
        except Exception:
            continue

    if config_file is None:
        config_file = hf_hub_download(model_id, "config.json")

    with open(config_file) as f:
        config = json.load(f)
    return os.path.dirname(config_file), config


def _resolve_model_path(
    model_path: str,
    *,
    attn_kind: str,
    num_layers: int,
    shrink_unused_moe: bool,
    disable_weight_quant: bool,
    strip_auto_map: bool = True,
) -> str:
    """Create a local config dir patched for a single DSV4 attention kind."""

    if os.path.isdir(model_path):
        src_dir = model_path
        with open(os.path.join(src_dir, "config.json")) as f:
            config = json.load(f)
    else:
        src_dir, config = _download_non_weight_model_files(model_path)

    config = copy.deepcopy(config)
    if strip_auto_map:
        config.pop("auto_map", None)

    compress_ratio = ATTN_KIND_TO_COMPRESS_RATIO[attn_kind]
    config["model_type"] = "deepseek_ref"
    config["num_hidden_layers"] = num_layers
    config["num_key_value_heads"] = 1
    if config.get("architectures") != ["DeepseekV4ForCausalLM"]:
        config["architectures"] = ["DeepseekV4ForCausalLM"]

    if disable_weight_quant:
        config.pop("quantization_config", None)
        config.pop("compression_config", None)

    old_ratios = config.get("compress_ratios") or []
    if old_ratios:
        config["compress_ratios"] = [compress_ratio] * max(len(old_ratios), num_layers)
    else:
        config["compress_ratios"] = [compress_ratio] * num_layers

    if shrink_unused_moe:
        # The benchmark calls only layer.self_attn.  Keeping the production MoE
        # shape makes model construction allocate routed/shared expert weights
        # that are never used by this collector.
        config["n_routed_experts"] = min(int(config.get("n_routed_experts", 8)), 8)
        config["num_experts_per_tok"] = min(
            int(config.get("num_experts_per_tok", 2)), 2
        )
        config["moe_intermediate_size"] = min(
            int(config.get("moe_intermediate_size", 256)), 256
        )
        config["n_shared_experts"] = 0

    tmp_dir = os.path.join(
        tempfile.gettempdir(),
        f"aic_dsv4_{attn_kind}_{model_path.replace('/', '_')}_{os.getpid()}",
    )
    os.makedirs(tmp_dir, exist_ok=True)
    _copy_non_weight_files(src_dir, tmp_dir)
    with open(os.path.join(tmp_dir, "config.json"), "w") as f:
        json.dump(config, f)
    return tmp_dir


def _load_model_runner(
    model_path: str,
    *,
    attn_kind: str,
    num_layers: int,
    kv_cache_dtype: str,
    device: str,
    mem_fraction_static: float,
    max_total_tokens: int | None,
    shrink_unused_moe: bool,
    disable_weight_quant: bool,
):
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import suppress_other_loggers

    suppress_other_loggers()
    torch.cuda.set_device(device)

    local_model_path = _resolve_model_path(
        model_path,
        attn_kind=attn_kind,
        num_layers=num_layers,
        shrink_unused_moe=shrink_unused_moe,
        disable_weight_quant=disable_weight_quant,
    )
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0

    server_args = ServerArgs(
        model_path=local_model_path,
        dtype="auto",
        device="cuda",
        load_format=os.environ.get("SGLANG_LOAD_FORMAT", "dummy"),
        tp_size=1,
        trust_remote_code=True,
        mem_fraction_static=mem_fraction_static,
        disable_radix_cache=True,
        # The module benchmark below captures its own CUDA Graph and fails if
        # capture is not possible.  Keep SGLang's serving-level graph runner off
        # so it does not add unrelated full-model graph state to this collector.
        disable_cuda_graph=True,
        kv_cache_dtype=kv_cache_dtype,
        max_total_tokens=max_total_tokens,
        max_running_requests=16,
        max_prefill_tokens=max(max_total_tokens or 4096, 2048),
    )
    server_args.quantization = None
    server_args.enable_piecewise_cuda_graph = False
    server_args.attention_backend = "compressed"

    print(
        f"[dsv4-collector] model_path {model_path} -> {local_model_path}; "
        f"attn_kind={attn_kind}, backend=compressed, kv_cache_dtype={kv_cache_dtype}, "
        f"max_total_tokens={max_total_tokens}, shrink_unused_moe={shrink_unused_moe}, "
        f"disable_weight_quant={disable_weight_quant}"
    )

    _set_envs_and_config(server_args)
    model_config = ModelConfig.from_server_args(server_args)
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=mem_fraction_static,
        gpu_id=gpu_id,
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        moe_ep_rank=0,
        moe_ep_size=1,
        nccl_port=29500 + random.randint(0, 10000),
        server_args=server_args,
    )
    allocator = model_runner.token_to_kv_pool_allocator
    pool_parts = []
    for name in (
        "max_total_num_tokens",
        "full_max_total_num_tokens",
        "swa_max_total_num_tokens",
        "c4_max_total_num_tokens",
        "c128_max_total_num_tokens",
        "c4_state_pool_size",
        "c128_state_pool_size",
    ):
        if hasattr(model_runner, name):
            pool_parts.append(f"{name}={getattr(model_runner, name)}")
    if hasattr(allocator, "debug_print"):
        pool_parts.append(allocator.debug_print().strip())
    elif hasattr(allocator, "available_size"):
        pool_parts.append(f"available_size={allocator.available_size()}")
    print("[dsv4-collector] pool " + ", ".join(pool_parts))
    return model_runner


def _make_reqs(batch_size: int, seq_len: int, *, decode: bool):
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    reqs = []
    for i in range(batch_size):
        req = Req(
            rid=str(i),
            origin_input_text="",
            origin_input_ids=list(torch.randint(0, 10000, (seq_len,)).tolist()),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        req.prefix_indices = torch.empty((0,), dtype=torch.int64)
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids)
        req.logprob_start_len = 0
        if decode:
            req.cached_tokens = 0
            req.already_computed = 0
        reqs.append(req)
    return reqs


def _build_forward_batch(model_runner, batch_size: int, seq_len: int, *, is_prefill: bool):
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.chunk_cache import ChunkCache
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()

    reqs = _make_reqs(batch_size, seq_len, decode=not is_prefill)
    cache_params = CacheInitParams(
        disable=True,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        page_size=model_runner.token_to_kv_pool_allocator.page_size,
    )
    tree_cache = ChunkCache(cache_params)
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=tree_cache,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )

    if is_prefill:
        batch.prepare_for_extend()
    else:
        batch.prepare_for_extend()
        batch.output_ids = torch.randint(
            0, 10000, (batch_size,), dtype=torch.int64, device="cuda"
        )
        batch.prepare_for_decode()

    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    model_runner.attn_backend.init_forward_metadata(forward_batch)
    return forward_batch


def _make_inputs(
    model_runner,
    *,
    batch_size: int,
    seq_len: int,
    is_prefill: bool,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_size = model_runner.model.config.hidden_size
    if is_prefill:
        n_tokens = batch_size * seq_len
        positions = (
            torch.arange(seq_len, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .contiguous()
            .flatten()
        )
    else:
        n_tokens = batch_size
        positions = torch.full((batch_size,), seq_len, dtype=torch.int64, device=device)

    hidden_states = torch.randn(
        n_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    return hidden_states, positions


def _log_result(
    *,
    output_path: str | None,
    model_path: str,
    mode: str,
    attn_kind: str,
    compress_ratio: int,
    batch_size: int,
    seq_len: int,
    kv_cache_dtype: str,
    latency_ms: float,
    version: str,
    device_name: str,
    power_stats: dict | None = None,
) -> None:
    perf_filename = _resolve_perf_path(
        output_path, f"dsv4_{attn_kind}_{mode}_module_perf.txt"
    )
    is_prefill = mode == "context"
    log_perf(
        item_list=[
            {
                "model": model_path,
                "architecture": "DeepseekV4ForCausalLM",
                "mla_dtype": "bfloat16",
                "kv_cache_dtype": kv_cache_dtype,
                "gemm_type": "bfloat16",
                "num_heads": 128,
                "batch_size": batch_size,
                "isl": seq_len if is_prefill else 1,
                "tp_size": 1,
                "step": 0 if is_prefill else seq_len,
                "compress_ratio": compress_ratio,
                "latency": f"{latency_ms:.4f}",
            }
        ],
        framework="SGLang",
        version=version,
        device_name=device_name,
        op_name=f"dsv4_{attn_kind}_{mode}_module",
        kernel_source="compressed_flashmla",
        perf_filename=perf_filename,
        power_stats=power_stats,
    )


def run_dsv4_mla_module(
    *,
    model_path: str = DEFAULT_MODEL,
    mode: str,
    attn_kind: str,
    batch_sizes: Iterable[int],
    seq_lens: Iterable[int],
    layer_id: int = 0,
    num_layers: int = 1,
    kv_cache_dtype: str = "fp8_e4m3",
    num_warmup: int = 5,
    num_iterations: int = 20,
    graph_repeat: int = 1,
    device: str = "cuda:0",
    output_path: str | None = None,
    mem_fraction_static: float = 0.5,
    max_total_tokens: int | None = 4096,
    shrink_unused_moe: bool = True,
    disable_weight_quant: bool = True,
) -> list[dict[str, float]]:
    if num_iterations < 3:
        raise ValueError("num_iterations must be at least 3")
    if graph_repeat < 1:
        raise ValueError("graph_repeat must be at least 1")

    is_prefill = mode == "context"
    compress_ratio = ATTN_KIND_TO_COMPRESS_RATIO[attn_kind]
    model_runner = _load_model_runner(
        model_path,
        attn_kind=attn_kind,
        num_layers=max(num_layers, layer_id + 1),
        kv_cache_dtype=kv_cache_dtype,
        device=device,
        mem_fraction_static=mem_fraction_static,
        max_total_tokens=max_total_tokens,
        shrink_unused_moe=shrink_unused_moe,
        disable_weight_quant=disable_weight_quant,
    )

    attention_module = model_runner.model.model.layers[layer_id].self_attn
    actual_ratio = getattr(attention_module, "compress_ratio", None)
    if actual_ratio != compress_ratio:
        raise RuntimeError(
            f"target layer compress_ratio mismatch: expected {compress_ratio}, got {actual_ratio}"
        )

    print(
        f"[dsv4-collector] layer={layer_id}, attn_kind={attn_kind}, "
        f"compress_ratio={actual_ratio}, mode={mode}"
    )

    version = get_version("sglang")
    device_name = torch.cuda.get_device_name(device)
    results = []
    try:
        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                print(f"\n{mode}: batch_size={batch_size}, seq_len={seq_len}")
                try:
                    forward_batch = _build_forward_batch(
                        model_runner, batch_size, seq_len, is_prefill=is_prefill
                    )
                    hidden_states, positions = _make_inputs(
                        model_runner,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        is_prefill=is_prefill,
                        device=device,
                    )

                    def kernel_func():
                        with torch.no_grad():
                            return attention_module(
                                x=hidden_states,
                                positions=positions,
                                forward_batch=forward_batch,
                            )

                    with benchmark_with_power(
                        device=torch.device(device),
                        kernel_func=kernel_func,
                        num_warmups=num_warmup,
                        num_runs=num_iterations,
                        repeat_n=graph_repeat,
                        allow_graph_fail=False,
                        use_cuda_graph=True,
                    ) as bench_result:
                        pass

                    if not bench_result.get("used_cuda_graph", False):
                        raise RuntimeError("benchmark_with_power did not use CUDA Graph")

                    latency_ms = float(bench_result["latency_ms"])
                    stats = {
                        "mean_ms": latency_ms,
                        "median_ms": latency_ms,
                        "min_ms": latency_ms,
                        "max_ms": latency_ms,
                        "std_ms": 0.0,
                        "n": int(bench_result.get("num_runs_executed", num_iterations)),
                        "used_cuda_graph": True,
                        "power_stats": bench_result.get("power_stats"),
                        "throttled": bool(bench_result.get("throttled", False)),
                    }
                    print(
                        f"  latency mean={stats['mean_ms']:.4f} ms, "
                        f"median={stats['median_ms']:.4f} ms, "
                        f"min={stats['min_ms']:.4f} ms, max={stats['max_ms']:.4f} ms, "
                        f"std={stats['std_ms']:.4f} ms, n={stats['n']}"
                    )
                    _log_result(
                        output_path=output_path,
                        model_path=model_path,
                        mode=mode,
                        attn_kind=attn_kind,
                        compress_ratio=compress_ratio,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        kv_cache_dtype=kv_cache_dtype,
                        latency_ms=stats["mean_ms"],
                        version=version,
                        device_name=device_name,
                        power_stats=stats.get("power_stats"),
                    )
                    stats.update(
                        {
                            "batch_size": batch_size,
                            "seq_len": seq_len,
                            "compress_ratio": compress_ratio,
                        }
                    )
                    results.append(stats)
                except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
                    print(f"  OOM: batch_size={batch_size}, seq_len={seq_len}; skipping")
                    torch.cuda.empty_cache()
                except Exception:
                    traceback.print_exc()
                    raise
                finally:
                    model_runner.req_to_token_pool.clear()
                    model_runner.token_to_kv_pool_allocator.clear()
                    torch.cuda.empty_cache()
                    gc.collect()
    finally:
        del model_runner
        torch.cuda.empty_cache()
        gc.collect()
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect DeepSeek-V4 HCA/CSA/SWA attention-module latency on SGLang."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--mode", choices=["context", "generation"], required=True)
    parser.add_argument("--attn-kind", choices=["swa", "csa", "hca"], required=True)
    parser.add_argument("--batch-sizes", default="1")
    parser.add_argument("--seq-lens", default="128")
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--kv-cache-dtype", default="fp8_e4m3")
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-iterations", type=int, default=20)
    parser.add_argument(
        "--graph-repeat",
        type=int,
        default=1,
        help="Repeat the measured module inside one CUDA Graph replay and report per-call latency.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--mem-fraction-static", type=float, default=0.5)
    parser.add_argument("--max-total-tokens", type=int, default=4096)
    parser.add_argument(
        "--keep-standard-moe",
        action="store_true",
        help="Keep production MoE shape during model init. Not needed for attention-only collection.",
    )
    parser.add_argument(
        "--keep-weight-quant",
        action="store_true",
        help="Keep checkpoint weight quantization config. Default removes it to avoid quantized projection setup.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_dsv4_mla_module(
        model_path=args.model_path,
        mode=args.mode,
        attn_kind=args.attn_kind,
        batch_sizes=_parse_int_list(args.batch_sizes),
        seq_lens=_parse_int_list(args.seq_lens),
        layer_id=args.layer_id,
        num_layers=args.num_layers,
        kv_cache_dtype=args.kv_cache_dtype,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
        graph_repeat=args.graph_repeat,
        device=args.device,
        output_path=args.output_path,
        mem_fraction_static=args.mem_fraction_static,
        max_total_tokens=args.max_total_tokens,
        shrink_unused_moe=not args.keep_standard_moe,
        disable_weight_quant=not args.keep_weight_quant,
    )


if __name__ == "__main__":
    main()
