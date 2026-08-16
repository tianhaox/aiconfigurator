from __future__ import annotations

import argparse
from pathlib import Path

from jit_rt.factory import load_runtime
from jit_rt.tokenizer import format_token_ids, parse_token_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the jit_rt static-batch POC.")
    parser.add_argument("manifest", type=Path, help="Runtime manifest YAML.")
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt", help='Space or comma separated token ids, for example "1 2 3".')
    prompt.add_argument("--text", help="Text prompt for runtimes that provide a tokenizer.")
    parser.add_argument("--max-new-tokens", type=int, default=4, help="Number of tokens to generate.")
    parser.add_argument(
        "--cuda-graph-probe",
        action="store_true",
        help="Capture and replay one fixed decode step when the runtime supports it.",
    )
    parser.add_argument(
        "--cuda-graph-decode",
        action="store_true",
        help="Use per-position CUDA graph replay for the full batch=1 token loop.",
    )
    parser.add_argument(
        "--prefill-piece-size",
        type=int,
        help="Use piecewise FlashInfer prefill chunks of this size before decode.",
    )
    parser.add_argument(
        "--paged-decode",
        action="store_true",
        help="Use FlashInfer paged-KV batch decode wrapper for non-graph decode attention.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime = load_runtime(args.manifest)
    if args.text is not None:
        if not hasattr(runtime, "encode_text"):
            raise TypeError("--text requires a runtime with a tokenizer")
        prompt = runtime.encode_text(args.text)
    else:
        prompt = parse_token_ids(args.prompt, runtime.spec.model.vocab_size)

    if args.paged_decode:
        if not hasattr(runtime, "enable_paged_decode"):
            raise TypeError("--paged-decode requires a runtime with enable_paged_decode")
        if not args.cuda_graph_decode:
            runtime.enable_paged_decode()

    if args.cuda_graph_decode:
        if not hasattr(runtime, "generate_cuda_graph"):
            raise TypeError("--cuda-graph-decode requires a runtime with generate_cuda_graph")
        result = runtime.generate_cuda_graph(
            prompt,
            max_new_tokens=args.max_new_tokens,
            prefill_piece_size=args.prefill_piece_size,
            paged_decode=args.paged_decode,
        )
    elif args.prefill_piece_size is not None:
        if not hasattr(runtime, "generate_piecewise_prefill"):
            raise TypeError("--prefill-piece-size requires a runtime with generate_piecewise_prefill")
        result = runtime.generate_piecewise_prefill(
            prompt,
            max_new_tokens=args.max_new_tokens,
            piece_size=args.prefill_piece_size,
        )
    else:
        result = runtime.generate(prompt, max_new_tokens=args.max_new_tokens)

    print(f"prompt:    {format_token_ids(result.prompt_token_ids)}")
    print(f"generated: {format_token_ids(result.generated_token_ids)}")
    print(f"tokens:    {format_token_ids(result.token_ids)}")
    if hasattr(runtime, "decode_tokens"):
        print(f"text:      {runtime.decode_tokens(result.token_ids)}")
    if args.cuda_graph_probe:
        if not hasattr(runtime, "cuda_graph_probe"):
            raise TypeError("--cuda-graph-probe requires a runtime with cuda_graph_probe")
        print(f"cuda_graph_probe_next: {runtime.cuda_graph_probe(result.prompt_token_ids[0])}")
    return 0
