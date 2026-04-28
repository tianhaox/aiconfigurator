# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 mHC pre/post module collector for SGLang."""

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
from collections.abc import Sequence
from importlib.metadata import version as get_version

import torch

os.environ.setdefault("SGLANG_APPLY_CONFIG_BACKUP", "none")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

try:
    from helper import EXIT_CODE_RESTART, benchmark_with_power, log_perf
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from helper import EXIT_CODE_RESTART, benchmark_with_power, log_perf


DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Pro"
PERF_FILENAME = "dsv4_mhc_module_perf.txt"
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
_NUM_TOKENS_SWEEP = (
    1,
    2,
    4,
    8,
    16,
    32,
    48,
    64,
    80,
    96,
    128,
    160,
    192,
    256,
    320,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
    12288,
    16384,
    20480,
    32768,
    49152,
    65536,
    98304,
    131072,
)


def _parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _env_csv_ints(name: str) -> set[int] | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


def _build_mhc_token_sweep() -> list[int]:
    token_filter = _env_csv_ints("COLLECTOR_DSV4_MHC_NUM_TOKENS")
    max_tokens = _env_int("COLLECTOR_DSV4_MHC_MAX_TOKENS", 131072)
    max_cases = _env_int("COLLECTOR_DSV4_MHC_MAX_CASES", 0)

    tokens = [
        value
        for value in _NUM_TOKENS_SWEEP
        if value <= max_tokens and (token_filter is None or value in token_filter)
    ]
    return tokens[:max_cases] if max_cases > 0 else tokens


def get_dsv4_mhc_module_test_cases() -> list[dict]:
    return [
        {"id": f"dsv4_mhc_{op}_{num_tokens}", "params": [op, num_tokens]}
        for op in ("pre", "post")
        for num_tokens in _build_mhc_token_sweep()
    ]


def _resolve_perf_path(output_path: str | None, filename: str | None) -> str:
    filename = filename or PERF_FILENAME
    if not output_path:
        return filename
    if output_path.endswith(".txt"):
        return output_path
    os.makedirs(output_path, exist_ok=True)
    return os.path.join(output_path, filename)


def _copy_aux_model_files(src_dir: str, dst_dir: str) -> None:
    for fname in os.listdir(src_dir):
        src_path = os.path.join(src_dir, fname)
        if (
            not os.path.isfile(src_path)
            or fname == "config.json"
            or fname.endswith(_WEIGHT_SUFFIXES)
        ):
            continue
        shutil.copy2(src_path, os.path.join(dst_dir, fname))


def _download_aux_model_files(model_id: str) -> tuple[str, dict]:
    from huggingface_hub import hf_hub_download, list_repo_files

    try:
        repo_files = list_repo_files(model_id)
    except Exception:
        repo_files = ["config.json"]

    config_file = None
    for fname in repo_files:
        if fname.endswith(_WEIGHT_SUFFIXES):
            continue
        try:
            path = hf_hub_download(model_id, fname)
        except Exception:
            continue
        if fname == "config.json":
            config_file = path

    if config_file is None:
        config_file = hf_hub_download(model_id, "config.json")

    with open(config_file) as f:
        config = json.load(f)
    return os.path.dirname(config_file), config


def _read_model_config(model_path: str) -> tuple[str, dict]:
    if os.path.isdir(model_path):
        with open(os.path.join(model_path, "config.json")) as f:
            return model_path, json.load(f)
    return _download_aux_model_files(model_path)


def _patched_model_dir(model_path: str) -> str:
    src_dir, config = _read_model_config(model_path)
    config = copy.deepcopy(config)

    config.pop("auto_map", None)
    config.pop("quantization_config", None)
    config.pop("compression_config", None)
    config.update(
        {
            "architectures": ["DeepseekV4ForCausalLM"],
            "model_type": "deepseek_ref",
            "num_hidden_layers": 1,
            "num_key_value_heads": 1,
            "n_shared_experts": 0,
        }
    )
    config["compress_ratios"] = [0] * max(len(config.get("compress_ratios") or []), 1)
    config["n_routed_experts"] = min(int(config.get("n_routed_experts", 8)), 8)
    config["num_experts_per_tok"] = min(int(config.get("num_experts_per_tok", 2)), 2)
    config["moe_intermediate_size"] = min(
        int(config.get("moe_intermediate_size", 256)), 256
    )

    tmp_dir = os.path.join(
        tempfile.gettempdir(),
        f"aic_dsv4_mhc_{model_path.replace('/', '_')}_{os.getpid()}",
    )
    os.makedirs(tmp_dir, exist_ok=True)
    _copy_aux_model_files(src_dir, tmp_dir)
    with open(os.path.join(tmp_dir, "config.json"), "w") as f:
        json.dump(config, f)
    return tmp_dir


def _load_one_layer_runner(
    model_path: str,
    *,
    device: str,
    mem_fraction_static: float,
):
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import suppress_other_loggers

    suppress_other_loggers()
    device_obj = torch.device(device)
    torch.cuda.set_device(device_obj)

    local_model_path = _patched_model_dir(model_path)
    gpu_id = device_obj.index if device_obj.index is not None else torch.cuda.current_device()
    server_args = ServerArgs(
        model_path=local_model_path,
        dtype="auto",
        device="cuda",
        load_format=os.environ.get("SGLANG_LOAD_FORMAT", "dummy"),
        tp_size=1,
        trust_remote_code=True,
        mem_fraction_static=mem_fraction_static,
        disable_radix_cache=True,
        disable_cuda_graph=True,
        kv_cache_dtype="fp8_e4m3",
        max_total_tokens=4096,
        max_running_requests=16,
        max_prefill_tokens=4096,
    )
    server_args.quantization = None
    server_args.enable_piecewise_cuda_graph = False
    server_args.attention_backend = "compressed"

    print(f"[dsv4-mhc-collector] model_path {model_path} -> {local_model_path}")

    _set_envs_and_config(server_args)
    model_config = ModelConfig.from_server_args(server_args)
    return ModelRunner(
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


def _hidden_size(layer) -> int:
    return int(layer.hc_attn_fn.shape[1] // layer.hc_mult)


def _make_residual(layer, num_tokens: int, device: str) -> torch.Tensor:
    return torch.randn(
        num_tokens,
        layer.hc_mult,
        _hidden_size(layer),
        dtype=torch.bfloat16,
        device=device,
    )


def _mhc_call_args(layer):
    # A real DSV4 layer executes mHC once before attention and once before FFN.
    # This collector folds both calls into the reported pre/post op.
    return (
        (layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base),
        (layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base),
    )


def _make_kernel(layer, op: str, residual: torch.Tensor):
    if op == "pre":
        call_args = _mhc_call_args(layer)

        def kernel():
            return [layer.hc_pre(residual, *args) for args in call_args]

        return kernel

    if op == "post":
        with torch.no_grad():
            post_inputs = [
                layer.hc_pre(residual, *args) for args in _mhc_call_args(layer)
            ]
        torch.cuda.synchronize()

        def kernel():
            return [
                layer.hc_post(x, residual, post, comb)
                for x, post, comb in post_inputs
            ]

        return kernel

    raise ValueError(f"unsupported mHC op: {op}")


def _benchmark_mhc_kernel(
    *,
    device: str,
    kernel_func,
    num_warmup: int,
    num_iterations: int,
) -> dict:
    def timed_kernel():
        with torch.no_grad():
            return kernel_func()

    with benchmark_with_power(
        device=torch.device(device),
        kernel_func=timed_kernel,
        num_warmups=num_warmup,
        num_runs=num_iterations,
        repeat_n=1,
        allow_graph_fail=False,
        use_cuda_graph=True,
    ) as bench_result:
        pass

    if not bench_result.get("used_cuda_graph", False):
        raise RuntimeError("benchmark_with_power did not use CUDA Graph")
    return bench_result


def _log_result(
    *,
    output_path: str | None,
    perf_filename: str | None,
    model_path: str,
    op: str,
    num_tokens: int,
    hc_mult: int,
    hidden_size: int,
    latency_ms: float,
    version: str,
    device_name: str,
    power_stats: dict | None,
) -> None:
    log_perf(
        item_list=[
            {
                "model": model_path,
                "architecture": "DeepseekV4ForCausalLM",
                "mla_dtype": "bfloat16",
                "kv_cache_dtype": "none",
                "gemm_type": "bfloat16",
                "num_heads": 0,
                "batch_size": num_tokens,
                "isl": 1,
                "tp_size": 1,
                "step": 0,
                "mhc_op": op,
                "num_tokens": num_tokens,
                "hc_mult": hc_mult,
                "hidden_size": hidden_size,
                "latency": f"{latency_ms:.4f}",
            }
        ],
        framework="SGLang",
        version=version,
        device_name=device_name,
        op_name=f"dsv4_mhc_{op}_module",
        kernel_source="sglang_mhc",
        perf_filename=_resolve_perf_path(output_path, perf_filename),
        power_stats=power_stats,
    )


def run_dsv4_mhc_module(
    *,
    ops: Sequence[str],
    num_tokens_cases: Sequence[int] | None = None,
    model_path: str = DEFAULT_MODEL,
    num_warmup: int = 5,
    num_iterations: int = 20,
    device: str = "cuda:0",
    output_path: str | None = None,
    mem_fraction_static: float = 0.5,
    perf_filename: str | None = None,
) -> list[dict[str, float]]:
    if num_iterations < 3:
        raise ValueError("num_iterations must be at least 3")

    token_cases = [
        int(num_tokens) for num_tokens in (num_tokens_cases or _build_mhc_token_sweep())
    ]
    model_runner = _load_one_layer_runner(
        model_path,
        device=device,
        mem_fraction_static=mem_fraction_static,
    )

    layer = model_runner.model.model.layers[0]
    hidden_size = _hidden_size(layer)
    version = get_version("sglang")
    device_name = torch.cuda.get_device_name(device)
    results: list[dict[str, float]] = []

    print(
        "[dsv4-mhc-collector] "
        f"hc_mult={layer.hc_mult}, hidden_size={hidden_size}, "
        f"tilelang_pre={os.environ.get('SGLANG_OPT_USE_TILELANG_MHC_PRE', 'default')}, "
        f"tilelang_post={os.environ.get('SGLANG_OPT_USE_TILELANG_MHC_POST', 'default')}"
    )

    try:
        for op in ops:
            for num_tokens in token_cases:
                print(f"\n{op}: num_tokens={num_tokens}")
                try:
                    residual = _make_residual(layer, num_tokens, device)
                    bench_result = _benchmark_mhc_kernel(
                        device=device,
                        kernel_func=_make_kernel(layer, op, residual),
                        num_warmup=num_warmup,
                        num_iterations=num_iterations,
                    )
                except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
                    print(f"  OOM: op={op}, num_tokens={num_tokens}; skipping")
                    torch.cuda.empty_cache()
                    continue

                latency_ms = float(bench_result["latency_ms"])
                print(f"  latency={latency_ms:.4f} ms")
                _log_result(
                    output_path=output_path,
                    perf_filename=perf_filename,
                    model_path=model_path,
                    op=op,
                    num_tokens=num_tokens,
                    hc_mult=layer.hc_mult,
                    hidden_size=hidden_size,
                    latency_ms=latency_ms,
                    version=version,
                    device_name=device_name,
                    power_stats=bench_result.get("power_stats"),
                )
                results.append(
                    {
                        "op": op,
                        "num_tokens": num_tokens,
                        "mean_ms": latency_ms,
                        "n": int(bench_result.get("num_runs_executed", num_iterations)),
                        "used_cuda_graph": True,
                        "throttled": bool(bench_result.get("throttled", False)),
                    }
                )
                torch.cuda.empty_cache()
                gc.collect()
    finally:
        del model_runner
        torch.cuda.empty_cache()
        gc.collect()
    return results


def run_dsv4_mhc_module_worker(
    op: str,
    num_tokens: int,
    model_path: str = DEFAULT_MODEL,
    perf_filename: str | None = None,
    device: str = "cuda:0",
) -> None:
    run_dsv4_mhc_module(
        ops=[op],
        num_tokens_cases=[num_tokens],
        model_path=model_path,
        device=device,
        perf_filename=perf_filename,
    )
    sys.exit(EXIT_CODE_RESTART)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect DeepSeek-V4 mHC pre/post module latency on SGLang."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--op", choices=["pre", "post", "all"], default="all")
    parser.add_argument("--num-tokens", default=None)
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-iterations", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--mem-fraction-static", type=float, default=0.5)
    args = parser.parse_args()

    run_dsv4_mhc_module(
        ops=["pre", "post"] if args.op == "all" else [args.op],
        num_tokens_cases=_parse_int_list(args.num_tokens) if args.num_tokens else None,
        model_path=args.model_path,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
        device=args.device,
        output_path=args.output_path,
        mem_fraction_static=args.mem_fraction_static,
    )


if __name__ == "__main__":
    main()
