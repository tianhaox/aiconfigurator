#!/usr/bin/env python3
"""Generate prefill FPM train/validation/test shape plans."""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAX_TOKENS = 65_536 * 4
DEFAULT_MAX_BATCH_TOTAL_CONTEXT_TOKENS = 1024 * 1024
DEFAULT_BATCH_ANCHORS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
DEFAULT_NEW_TOKEN_ANCHORS = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
    12_288,
    16_384,
    24_576,
    32_768,
    49_152,
    65_536,
]
DEFAULT_ABS_PREFIX_ANCHORS = [0, 1, 4, 16, 64, 256, 1024, 4096, 16_384, 65_536, 131_072]
CSV_FIELDS = [
    "shape_id",
    "split",
    "distribution",
    "phase",
    "region",
    "batch_size",
    "new_tokens",
    "past_kv_tokens",
    "total_context_tokens",
    "sum_prefill_tokens",
    "sum_prefill_kv_tokens",
    "attention_work_tokens",
    "static_isl",
    "static_prefix",
    "static_osl",
    "latency_ms",
    "status",
    "error",
]


@dataclass(frozen=True)
class Shape:
    split: str
    distribution: str
    batch_size: int
    new_tokens: int
    past_kv_tokens: int

    @property
    def total_context_tokens(self) -> int:
        return self.new_tokens + self.past_kv_tokens

    @property
    def sum_prefill_tokens(self) -> int:
        return self.batch_size * self.new_tokens

    @property
    def sum_prefill_kv_tokens(self) -> int:
        return self.batch_size * self.past_kv_tokens

    @property
    def attention_work_tokens(self) -> int:
        return self.batch_size * self.new_tokens * self.total_context_tokens

    def key(self) -> tuple[int, int, int]:
        return (self.batch_size, self.new_tokens, self.past_kv_tokens)


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def powers_of_two_up_to(max_value: int) -> list[int]:
    values: list[int] = []
    current = 1
    while current <= max_value:
        values.append(current)
        current *= 2
    return values


def is_valid_shape(
    batch_size: int,
    new_tokens: int,
    past_kv_tokens: int,
    *,
    max_batch_size: int,
    max_new_tokens: int,
    max_total_context_tokens: int,
    max_sum_prefill_tokens: int,
    max_batch_total_context_tokens: int | None,
) -> bool:
    if not (1 <= batch_size <= max_batch_size):
        return False
    if not (1 <= new_tokens <= max_new_tokens):
        return False
    if past_kv_tokens < 0:
        return False

    total_context_tokens = new_tokens + past_kv_tokens
    if total_context_tokens > max_total_context_tokens:
        return False
    if batch_size * new_tokens > max_sum_prefill_tokens:
        return False
    return not (
        max_batch_total_context_tokens is not None
        and batch_size * total_context_tokens > max_batch_total_context_tokens
    )


def prefix_anchors_for_new_tokens(new_tokens: int, max_total_context_tokens: int) -> list[int]:
    candidates: set[int] = {0}
    for multiplier in [1, 3, 7, 15, 31]:
        candidates.add(new_tokens * multiplier)
    candidates.update(DEFAULT_ABS_PREFIX_ANCHORS)
    candidates.add(max_total_context_tokens - new_tokens)
    return sorted(prefix for prefix in candidates if 0 <= prefix <= max_total_context_tokens - new_tokens)


def classify_region(batch_size: int, new_tokens: int, past_kv_tokens: int) -> str:
    total_context_tokens = new_tokens + past_kv_tokens
    sum_prefill_tokens = batch_size * new_tokens
    prefix_ratio = past_kv_tokens / total_context_tokens

    if new_tokens <= 16 and past_kv_tokens >= 4096:
        return "short_new_long_prefix"
    if new_tokens <= 16 and total_context_tokens <= 128:
        return "short_new_short_context"
    if new_tokens <= 16:
        return "short_new"
    if total_context_tokens >= 131_072:
        return "ultra_long_total"
    if new_tokens >= 32_768:
        return "ultra_long_new"
    if new_tokens >= 8192:
        return "long_new"
    if batch_size >= 16 and sum_prefill_tokens >= 65_536:
        return "high_batch_high_tokens"
    if batch_size >= 16:
        return "high_batch"
    if sum_prefill_tokens >= 65_536:
        return "high_tokens"
    if past_kv_tokens == 0:
        return "no_prefix_mid"
    if prefix_ratio >= 0.75:
        return "prefix_dominant"
    return "mid"


def build_train_grid(args: argparse.Namespace) -> list[Shape]:
    shapes: list[Shape] = []
    batch_anchors = parse_int_list(args.batch_anchors)
    new_token_anchors = parse_int_list(args.new_token_anchors)
    max_batch_total = none_if_non_positive(args.max_batch_total_context_tokens)

    for batch_size in batch_anchors:
        for new_tokens in new_token_anchors:
            for past_kv_tokens in prefix_anchors_for_new_tokens(new_tokens, args.max_total_context_tokens):
                if is_valid_shape(
                    batch_size,
                    new_tokens,
                    past_kv_tokens,
                    max_batch_size=args.max_batch_size,
                    max_new_tokens=args.max_new_tokens,
                    max_total_context_tokens=args.max_total_context_tokens,
                    max_sum_prefill_tokens=args.max_sum_prefill_tokens,
                    max_batch_total_context_tokens=max_batch_total,
                ):
                    shapes.append(
                        Shape(
                            split="train",
                            distribution="regular_anchor_grid",
                            batch_size=batch_size,
                            new_tokens=new_tokens,
                            past_kv_tokens=past_kv_tokens,
                        )
                    )
    return dedupe_shapes(shapes)


def dedupe_shapes(shapes: Iterable[Shape]) -> list[Shape]:
    out: list[Shape] = []
    seen: set[tuple[str, tuple[int, int, int]]] = set()
    for shape in shapes:
        key = (shape.split, shape.key())
        if key in seen:
            continue
        seen.add(key)
        out.append(shape)
    return out


def none_if_non_positive(value: int) -> int | None:
    return value if value > 0 else None


def random_log_int(rng: random.Random, lo: int, hi: int) -> int:
    if lo >= hi:
        return lo
    value = 2 ** rng.uniform(math.log2(lo), math.log2(hi))
    return max(lo, min(hi, round(value)))


def lhs_log_values(rng: random.Random, count: int, lo: int, hi: int) -> list[int]:
    if count <= 0:
        return []
    if lo >= hi:
        return [lo] * count
    lo_log = math.log2(lo)
    hi_log = math.log2(hi)
    values = []
    for index in range(count):
        frac = (index + rng.random()) / count
        values.append(max(lo, min(hi, round(2 ** (lo_log + frac * (hi_log - lo_log))))))
    rng.shuffle(values)
    return values


def make_shape_from_total(
    *,
    split: str,
    distribution: str,
    batch_size: int,
    new_tokens: int,
    total_context_tokens: int,
) -> Shape:
    total_context_tokens = max(total_context_tokens, new_tokens)
    return Shape(
        split=split,
        distribution=distribution,
        batch_size=batch_size,
        new_tokens=new_tokens,
        past_kv_tokens=total_context_tokens - new_tokens,
    )


def build_validation_random(args: argparse.Namespace, occupied: set[tuple[int, int, int]]) -> list[Shape]:
    rng = random.Random(args.seed + 101)
    batch_values = lhs_log_values(rng, args.val_count * 4, 1, args.max_batch_size)
    new_token_values = lhs_log_values(rng, args.val_count * 4, 1, args.max_new_tokens)
    total_values = lhs_log_values(rng, args.val_count * 4, 1, args.max_total_context_tokens)
    shapes: list[Shape] = []
    max_batch_total = none_if_non_positive(args.max_batch_total_context_tokens)

    for batch_size, new_tokens, total_context_tokens in zip(
        batch_values,
        new_token_values,
        total_values,
        strict=True,
    ):
        if len(shapes) >= args.val_count:
            break
        total_context_tokens = max(total_context_tokens, new_tokens)
        shape = make_shape_from_total(
            split="val",
            distribution="lhs_log_uniform",
            batch_size=batch_size,
            new_tokens=new_tokens,
            total_context_tokens=total_context_tokens,
        )
        if shape.key() in occupied:
            continue
        if is_valid_shape(
            shape.batch_size,
            shape.new_tokens,
            shape.past_kv_tokens,
            max_batch_size=args.max_batch_size,
            max_new_tokens=args.max_new_tokens,
            max_total_context_tokens=args.max_total_context_tokens,
            max_sum_prefill_tokens=args.max_sum_prefill_tokens,
            max_batch_total_context_tokens=max_batch_total,
        ):
            occupied.add(shape.key())
            shapes.append(shape)

    return fill_random_until_count(args, rng, shapes, occupied, split="val", distribution="lhs_log_uniform")


def build_test_random(args: argparse.Namespace, occupied: set[tuple[int, int, int]]) -> list[Shape]:
    rng = random.Random(args.seed + 202)
    shapes: list[Shape] = []

    generators = [
        (0.20, lambda: random_log_uniform_shape(args, rng, split="test", distribution="mixture_log_uniform")),
        (0.18, lambda: random_short_new_long_prefix_shape(args, rng)),
        (0.14, lambda: random_short_new_short_context_shape(args, rng)),
        (0.18, lambda: random_ultra_long_shape(args, rng)),
        (0.14, lambda: random_high_batch_shape(args, rng)),
        (0.16, lambda: random_boundary_shape(args, rng)),
    ]
    target_counts = allocate_counts(args.test_count, [weight for weight, _ in generators])
    for target_count, (_, generator) in zip(target_counts, generators, strict=True):
        append_generated_shapes(args, shapes, occupied, target_count, generator)

    return shapes


def allocate_counts(total: int, weights: list[float]) -> list[int]:
    raw = [total * weight / sum(weights) for weight in weights]
    counts = [int(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(weights)), key=lambda index: raw[index] - counts[index], reverse=True)
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def append_generated_shapes(
    args: argparse.Namespace,
    shapes: list[Shape],
    occupied: set[tuple[int, int, int]],
    target_count: int,
    generator,
) -> None:
    attempts = 0
    start_count = len(shapes)
    while len(shapes) - start_count < target_count:
        attempts += 1
        if attempts > 100_000:
            raise RuntimeError("Could not generate enough test shapes for one region bucket.")
        shape = generator()
        if shape.key() in occupied:
            continue
        if shape_is_valid_with_args(shape, args):
            occupied.add(shape.key())
            shapes.append(shape)


def fill_random_until_count(
    args: argparse.Namespace,
    rng: random.Random,
    shapes: list[Shape],
    occupied: set[tuple[int, int, int]],
    *,
    split: str,
    distribution: str,
) -> list[Shape]:
    attempts = 0
    while len(shapes) < (args.val_count if split == "val" else args.test_count):
        attempts += 1
        if attempts > 100_000:
            raise RuntimeError(f"Could not generate enough {split} shapes under the configured constraints.")
        shape = random_log_uniform_shape(args, rng, split=split, distribution=distribution)
        if shape.key() in occupied:
            continue
        if shape_is_valid_with_args(shape, args):
            occupied.add(shape.key())
            shapes.append(shape)
    return shapes


def random_log_uniform_shape(args: argparse.Namespace, rng: random.Random, *, split: str, distribution: str) -> Shape:
    batch_size = random_log_int(rng, 1, args.max_batch_size)
    max_total_for_batch = max_total_context_for_batch(args, batch_size)
    max_new_for_batch = max_new_tokens_for_batch(args, batch_size, max_total_for_batch)
    new_tokens = random_log_int(rng, 1, max_new_for_batch)
    total_context_tokens = random_log_int(rng, new_tokens, max_total_for_batch)
    return make_shape_from_total(
        split=split,
        distribution=distribution,
        batch_size=batch_size,
        new_tokens=new_tokens,
        total_context_tokens=total_context_tokens,
    )


def random_long_prefix_shape(args: argparse.Namespace, rng: random.Random) -> Shape:
    batch_size = random_log_int(rng, 1, args.max_batch_size)
    max_total_for_batch = max_total_context_for_batch(args, batch_size)
    max_new_for_batch = max_new_tokens_for_batch(args, batch_size, max_total_for_batch)
    new_tokens = random_log_int(rng, 1, max(1, min(max_new_for_batch, 8192)))
    total_lo = max(new_tokens, min(max_total_for_batch, args.max_total_context_tokens // 16))
    total_context_tokens = random_log_int(rng, total_lo, max_total_for_batch)
    return make_shape_from_total(
        split="test",
        distribution="mixture_long_prefix",
        batch_size=batch_size,
        new_tokens=new_tokens,
        total_context_tokens=total_context_tokens,
    )


def random_boundary_shape(args: argparse.Namespace, rng: random.Random) -> Shape:
    batch_size = rng.choice([1, 2, 4, 8, 16, 24, 32])
    max_total_for_batch = max_total_context_for_batch(args, batch_size)
    max_new_for_batch = max_new_tokens_for_batch(args, batch_size, max_total_for_batch)
    if rng.random() < 0.5:
        new_tokens = max(1, int(max_new_for_batch * rng.uniform(0.75, 1.0)))
        total_context_tokens = random_log_int(rng, new_tokens, max_total_for_batch)
    else:
        new_tokens = random_log_int(rng, 1, max_new_for_batch)
        total_context_tokens = max(new_tokens, int(max_total_for_batch * rng.uniform(0.75, 1.0)))
    return make_shape_from_total(
        split="test",
        distribution="mixture_boundary",
        batch_size=batch_size,
        new_tokens=new_tokens,
        total_context_tokens=total_context_tokens,
    )


def random_small_step_shape(args: argparse.Namespace, rng: random.Random) -> Shape:
    batch_size = rng.randint(1, args.max_batch_size)
    new_tokens = rng.randint(1, min(128, args.max_new_tokens))
    total_context_tokens = random_log_int(rng, new_tokens, max_total_context_for_batch(args, batch_size))
    return make_shape_from_total(
        split="test",
        distribution="mixture_small_step",
        batch_size=batch_size,
        new_tokens=new_tokens,
        total_context_tokens=total_context_tokens,
    )


def random_short_new_long_prefix_shape(args: argparse.Namespace, rng: random.Random) -> Shape:
    batch_size = random_log_int(rng, 1, args.max_batch_size)
    max_total_for_batch = max_total_context_for_batch(args, batch_size)
    new_tokens = rng.randint(1, min(16, max_new_tokens_for_batch(args, batch_size, max_total_for_batch)))
    total_lo = max(new_tokens + 4096, min(max_total_for_batch, 8192))
    total_context_tokens = random_log_int(rng, total_lo, max_total_for_batch)
    return make_shape_from_total(
        split="test",
        distribution="region_short_new_long_prefix",
        batch_size=batch_size,
        new_tokens=new_tokens,
        total_context_tokens=total_context_tokens,
    )


def random_short_new_short_context_shape(args: argparse.Namespace, rng: random.Random) -> Shape:
    batch_size = rng.randint(1, args.max_batch_size)
    max_total_for_batch = max_total_context_for_batch(args, batch_size)
    new_tokens = rng.randint(1, min(16, max_new_tokens_for_batch(args, batch_size, max_total_for_batch)))
    total_context_tokens = rng.randint(new_tokens, min(128, max_total_for_batch))
    return make_shape_from_total(
        split="test",
        distribution="region_short_new_short_context",
        batch_size=batch_size,
        new_tokens=new_tokens,
        total_context_tokens=total_context_tokens,
    )


def random_ultra_long_shape(args: argparse.Namespace, rng: random.Random) -> Shape:
    batch_size = random_log_int(rng, 1, args.max_batch_size)
    max_total_for_batch = max_total_context_for_batch(args, batch_size)
    max_new_for_batch = max_new_tokens_for_batch(args, batch_size, max_total_for_batch)
    if max_total_for_batch >= 131_072:
        total_context_tokens = random_log_int(rng, 131_072, max_total_for_batch)
        new_token_lo = min(64, max_new_for_batch)
        new_tokens = random_log_int(rng, new_token_lo, min(max_new_for_batch, total_context_tokens))
    else:
        new_tokens = random_log_int(rng, max(1, max_new_for_batch // 2), max_new_for_batch)
        total_context_tokens = random_log_int(rng, new_tokens, max_total_for_batch)
    return make_shape_from_total(
        split="test",
        distribution="region_ultra_long",
        batch_size=batch_size,
        new_tokens=new_tokens,
        total_context_tokens=total_context_tokens,
    )


def random_high_batch_shape(args: argparse.Namespace, rng: random.Random) -> Shape:
    batch_size = rng.randint(max(16, args.max_batch_size // 2), args.max_batch_size)
    max_total_for_batch = max_total_context_for_batch(args, batch_size)
    max_new_for_batch = max_new_tokens_for_batch(args, batch_size, max_total_for_batch)
    new_tokens = random_log_int(rng, 1, max_new_for_batch)
    total_context_tokens = random_log_int(rng, new_tokens, max_total_for_batch)
    return make_shape_from_total(
        split="test",
        distribution="region_high_batch",
        batch_size=batch_size,
        new_tokens=new_tokens,
        total_context_tokens=total_context_tokens,
    )


def max_total_context_for_batch(args: argparse.Namespace, batch_size: int) -> int:
    batch_total_cap = none_if_non_positive(args.max_batch_total_context_tokens)
    max_total = args.max_total_context_tokens
    if batch_total_cap is not None:
        max_total = min(max_total, max(1, batch_total_cap // batch_size))
    return max_total


def max_new_tokens_for_batch(args: argparse.Namespace, batch_size: int, max_total_for_batch: int) -> int:
    return min(args.max_new_tokens, max_total_for_batch, max(1, args.max_sum_prefill_tokens // batch_size))


def shape_is_valid_with_args(shape: Shape, args: argparse.Namespace) -> bool:
    return is_valid_shape(
        shape.batch_size,
        shape.new_tokens,
        shape.past_kv_tokens,
        max_batch_size=args.max_batch_size,
        max_new_tokens=args.max_new_tokens,
        max_total_context_tokens=args.max_total_context_tokens,
        max_sum_prefill_tokens=args.max_sum_prefill_tokens,
        max_batch_total_context_tokens=none_if_non_positive(args.max_batch_total_context_tokens),
    )


def row_for_shape(shape_id: str, shape: Shape) -> dict[str, str | int]:
    return {
        "shape_id": shape_id,
        "split": shape.split,
        "distribution": shape.distribution,
        "phase": "prefill",
        "region": classify_region(shape.batch_size, shape.new_tokens, shape.past_kv_tokens),
        "batch_size": shape.batch_size,
        "new_tokens": shape.new_tokens,
        "past_kv_tokens": shape.past_kv_tokens,
        "total_context_tokens": shape.total_context_tokens,
        "sum_prefill_tokens": shape.sum_prefill_tokens,
        "sum_prefill_kv_tokens": shape.sum_prefill_kv_tokens,
        "attention_work_tokens": shape.attention_work_tokens,
        "static_isl": shape.total_context_tokens,
        "static_prefix": shape.past_kv_tokens,
        "static_osl": 1,
        "latency_ms": "",
        "status": "planned",
        "error": "",
    }


def write_shapes(path: Path, shapes: list[Shape]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for index, shape in enumerate(shapes):
            writer.writerow(row_for_shape(f"prefill_{index:06d}", shape))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("fpm/data/prefill_shapes.csv"))
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--val-count", type=int, default=512)
    parser.add_argument("--test-count", type=int, default=512)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=65_536)
    parser.add_argument("--max-total-context-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-sum-prefill-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--max-batch-total-context-tokens",
        type=int,
        default=DEFAULT_MAX_BATCH_TOTAL_CONTEXT_TOKENS,
        help="Optional cap for b * (s + past_kv). Set 0 to disable.",
    )
    parser.add_argument(
        "--batch-anchors",
        default=",".join(str(value) for value in DEFAULT_BATCH_ANCHORS),
        help="Comma-separated train-grid batch anchors.",
    )
    parser.add_argument(
        "--new-token-anchors",
        default=",".join(str(value) for value in DEFAULT_NEW_TOKEN_ANCHORS),
        help="Comma-separated train-grid new-token anchors.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    train_shapes = build_train_grid(args)
    occupied = {shape.key() for shape in train_shapes}
    val_shapes = build_validation_random(args, occupied)
    test_shapes = build_test_random(args, occupied)
    all_shapes = train_shapes + val_shapes + test_shapes
    write_shapes(args.output, all_shapes)

    print(f"wrote {len(all_shapes)} shapes to {args.output}")
    print(f"train={len(train_shapes)} val={len(val_shapes)} test={len(test_shapes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
