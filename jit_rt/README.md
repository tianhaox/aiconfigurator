# jit_rt

`jit_rt` is a small proof-of-concept for a deployment-specific inference
runtime generator.

The first checkpoint is intentionally narrow:

- one request
- static batch size 1
- dense decoder-only toy model
- fixed feature set from a manifest
- fixed dense KV cache layout
- greedy sampling
- CPU `numpy` reference kernels

The goal is to make the glue modules explicit before replacing individual
reference kernels with FlashInfer, CUDA, Triton, or framework-extracted kernels.
Unsupported combinations fail at manifest load time instead of falling back.

## Run

From the repository root:

```bash
PYTHONPATH=. python -m jit_rt jit_rt/examples/tiny_static.yaml --prompt "1 2 3" --max-new-tokens 4
```

Expected output shape:

```text
prompt:    1 2 3
generated: ...
tokens:    1 2 3 ...
```

## Layout

- `config.py`: manifest schema and guardrails
- `weights.py`: synthetic toy weights for the runnable checkpoint
- `kernels.py`: reference kernel contracts
- `kv_cache.py`: fixed-size KV cache
- `runtime.py`: request execution and token loop
- `cli.py`: smoke-test CLI

The next useful step is to add a real weight converter and a GPU kernel adapter
behind the same runtime contracts.

## Qwen3-0.6B Real Checkpoint

The Qwen3 path is still narrow: `torch_reference`, CUDA `sm90`, BF16, static
batch size 1, greedy sampling, and dense static KV cache.

```bash
HF_HUB_DISABLE_XET=1 PYTHONPATH=. python -m jit_rt \
  jit_rt/examples/qwen3_0_6b.yaml \
  --text "Hello" \
  --max-new-tokens 2 \
  --prefill-piece-size 4 \
  --paged-decode \
```

To run decode with per-position CUDA graph replay instead of the paged decode
wrapper, omit `--paged-decode`:

```bash
HF_HUB_DISABLE_XET=1 PYTHONPATH=. python -m jit_rt \
  jit_rt/examples/qwen3_0_6b.yaml \
  --text "Hello" \
  --max-new-tokens 2 \
  --cuda-graph-decode \
  --cuda-graph-probe
```

To run piecewise prefill followed by paged-KV decode captured in CUDA graphs:

```bash
HF_HUB_DISABLE_XET=1 PYTHONPATH=. python -m jit_rt \
  jit_rt/examples/qwen3_0_6b.yaml \
  --text "Hello world, this is a test" \
  --max-new-tokens 2 \
  --prefill-piece-size 4 \
  --cuda-graph-decode \
  --paged-decode
```

This path loads `Qwen/Qwen3-0.6B` from Hugging Face safetensors and runs a
hand-written Qwen3 forward loop. It does not call `AutoModelForCausalLM.generate`.
Attention uses FlashInfer `single_decode_with_kv_cache` for the decode kernel.
`--cuda-graph-decode` captures one graph per requested token position and replays
the full batch=1 token loop. `--cuda-graph-probe` keeps a smaller fixed-step
capture sanity check. `--prefill-piece-size` runs prompt tokens in FlashInfer
prefill chunks before decode. `--paged-decode` stores KV in a paged layout and
uses FlashInfer `BatchDecodeWithPagedKVCacheWrapper` for non-graph decode.
With `--cuda-graph-decode --paged-decode`, each captured token position owns a
separate FlashInfer `CUDAGraphBatchDecodeWithPagedKVCacheWrapper` and metadata
buffer.
