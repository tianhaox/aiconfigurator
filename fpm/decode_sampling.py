#!/usr/bin/env python3
"""Generate decode FPM train/validation/test shape plans."""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAX_KV_TOKENS = 65_536 * 4
DEFAULT_MAX_BATCH_KV_TOKENS = 1024 * 1024
DEFAULT_BATCH_ANCHORS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
DEFAULT_KV_ANCHORS = [
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
    98_304,
    131_072,
    196_608,
    262_144,
]
CSV_FIELDS = [
    "shape_id",
    "split",
    "distribution",
    "phase",
    "region",
    "batch_size",
    "past_kv_tokens",
    "attention_kv_tokens",
    "sum_decode_tokens",
    "sum_decode_kv_tokens",
    "attention_work_tokens",
    "static_isl",
    "static_osl",
    "latency_ms",
    "status",
    "error",
]


@dataclass(frozen=True)
class DecodeShape:
    split: str
    distribution: str
    batch_size: int
    past_kv_tokens: int

    @property
    def attention_kv_tokens(self) -> int:
        return self.past_kv_tokens + 1

    @property
    def sum_decode_tokens(self) -> int:
        return self.batch_size

    @property
    def sum_decode_kv_tokens(self) -> int:
        return self.batch_size * self.past_kv_tokens

    @property
    def attention_work_tokens(self) -> int:
        return self.batch_size * self.attention_kv_tokens

    def key(self) -> tuple[int, int]:
        return (self.batch_size, self.past_kv_tokens)


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def none_if_non_positive(value: int) -> int | None:
    return value if value > 0 else None


def is_valid_shape(
    batch_size: int,
    past_kv_tokens: int,
    *,
    max_batch_size: int,
    max_kv_tokens: int,
    max_batch_kv_tokens: int | None,
) -> bool:
    if not (1 <= batch_size <= max_batch_size):
        return False
    if not (1 <= past_kv_tokens <= max_kv_tokens):
        return False
    return not (
        max_batch_kv_tokens is not None
        and batch_size * (past_kv_tokens + 1) > max_batch_kv_tokens
    )


def classify_region(batch_size: int, past_kv_tokens: int) -> str:
    attention_work_tokens = batch_size * (past_kv_tokens + 1)
    if batch_size >= 512 and past_kv_tokens >= 4096:
        return "high_batch_long_kv"
    if batch_size >= 512:
        return "very_high_batch"
    if batch_size >= 128 and past_kv_tokens >= 8192:
        return "high_batch_long_kv"
    if batch_size >= 128:
        return "high_batch"
    if attention_work_tokens >= 786_432:
        return "batch_kv_boundary"
    if past_kv_tokens >= 131_072:
        return "ultra_long_kv"
    if past_kv_tokens >= 32_768:
        return "long_kv"
    if past_kv_tokens <= 128:
        return "short_kv"
    return "mid_kv"


def dedupe_shapes(shapes: Iterable[DecodeShape]) -> list[DecodeShape]:
    out: list[DecodeShape] = []
    seen: set[tuple[str, tuple[int, int]]] = set()
    for shape in shapes:
        key = (shape.split, shape.key())
        if key in seen:
            continue
        seen.add(key)
        out.append(shape)
    return out


def build_train_grid(args: argparse.Namespace) -> list[DecodeShape]:
    shapes: list[DecodeShape] = []
    max_batch_kv = none_if_non_positive(args.max_batch_kv_tokens)
    for batch_size in parse_int_list(args.batch_anchors):
        for past_kv_tokens in parse_int_list(args.kv_anchors):
            if is_valid_shape(
                batch_size,
                past_kv_tokens,
                max_batch_size=args.max_batch_size,
                max_kv_tokens=args.max_kv_tokens,
                max_batch_kv_tokens=max_batch_kv,
            ):
                shapes.append(
                    DecodeShape(
                        split="train",
                        distribution="regular_anchor_grid",
                        batch_size=batch_size,
                        past_kv_tokens=past_kv_tokens,
                    )
                )
    return dedupe_shapes(shapes)


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


def max_kv_for_batch(args: argparse.Namespace, batch_size: int) -> int:
    max_batch_kv = none_if_non_positive(args.max_batch_kv_tokens)
    if max_batch_kv is None:
        return args.max_kv_tokens
    return max(1, min(args.max_kv_tokens, max_batch_kv // batch_size - 1))


def shape_is_valid_with_args(shape: DecodeShape, args: argparse.Namespace) -> bool:
    return is_valid_shape(
        shape.batch_size,
        shape.past_kv_tokens,
        max_batch_size=args.max_batch_size,
        max_kv_tokens=args.max_kv_tokens,
        max_batch_kv_tokens=none_if_non_positive(args.max_batch_kv_tokens),
    )


def build_validation_random(args: argparse.Namespace, occupied: set[tuple[int, int]]) -> list[DecodeShape]:
    rng = random.Random(args.seed + 101)
    batch_values = lhs_log_values(rng, args.val_count * 4, 1, args.max_batch_size)
    kv_values = lhs_log_values(rng, args.val_count * 4, 1, args.max_kv_tokens)
    shapes: list[DecodeShape] = []
    for batch_size, past_kv_tokens in zip(batch_values, kv_values, strict=True):
        if len(shapes) >= args.val_count:
            break
        past_kv_tokens = min(past_kv_tokens, max_kv_for_batch(args, batch_size))
        shape = DecodeShape(
            split="val",
            distribution="lhs_log_uniform",
            batch_size=batch_size,
            past_kv_tokens=past_kv_tokens,
        )
        if shape.key() in occupied:
            continue
        if shape_is_valid_with_args(shape, args):
            occupied.add(shape.key())
            shapes.append(shape)
    return fill_random_until_count(args, rng, shapes, occupied, split="val", distribution="lhs_log_uniform")


def build_test_random(args: argparse.Namespace, occupied: set[tuple[int, int]]) -> list[DecodeShape]:
    rng = random.Random(args.seed + 202)
    shapes: list[DecodeShape] = []
    generators = [
        (0.20, lambda: random_log_uniform_shape(args, rng, split="test", distribution="mixture_log_uniform")),
        (0.18, lambda: random_short_kv_shape(args, rng)),
        (0.18, lambda: random_long_kv_shape(args, rng)),
        (0.18, lambda: random_high_batch_shape(args, rng)),
        (0.16, lambda: random_boundary_shape(args, rng)),
        (0.10, lambda: random_linear_interior_shape(args, rng)),
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
    shapes: list[DecodeShape],
    occupied: set[tuple[int, int]],
    target_count: int,
    generator,
) -> None:
    attempts = 0
    start_count = len(shapes)
    while len(shapes) - start_count < target_count:
        attempts += 1
        if attempts > 100_000:
            raise RuntimeError("Could not generate enough decode test shapes.")
        shape = generator()
        if shape.key() in occupied:
            continue
        if shape_is_valid_with_args(shape, args):
            occupied.add(shape.key())
            shapes.append(shape)


def fill_random_until_count(
    args: argparse.Namespace,
    rng: random.Random,
    shapes: list[DecodeShape],
    occupied: set[tuple[int, int]],
    *,
    split: str,
    distribution: str,
) -> list[DecodeShape]:
    target = args.val_count if split == "val" else args.test_count
    attempts = 0
    while len(shapes) < target:
        attempts += 1
        if attempts > 100_000:
            raise RuntimeError(f"Could not generate enough {split} shapes.")
        shape = random_log_uniform_shape(args, rng, split=split, distribution=distribution)
        if shape.key() in occupied:
            continue
        if shape_is_valid_with_args(shape, args):
            occupied.add(shape.key())
            shapes.append(shape)
    return shapes


def random_log_uniform_shape(
    args: argparse.Namespace,
    rng: random.Random,
    *,
    split: str,
    distribution: str,
) -> DecodeShape:
    batch_size = random_log_int(rng, 1, args.max_batch_size)
    past_kv_tokens = random_log_int(rng, 1, max_kv_for_batch(args, batch_size))
    return DecodeShape(split=split, distribution=distribution, batch_size=batch_size, past_kv_tokens=past_kv_tokens)


def random_short_kv_shape(args: argparse.Namespace, rng: random.Random) -> DecodeShape:
    batch_size = random_log_int(rng, 1, args.max_batch_size)
    past_kv_tokens = rng.randint(1, min(128, max_kv_for_batch(args, batch_size)))
    return DecodeShape(
        split="test",
        distribution="region_short_kv",
        batch_size=batch_size,
        past_kv_tokens=past_kv_tokens,
    )


def random_long_kv_shape(args: argparse.Namespace, rng: random.Random) -> DecodeShape:
    batch_size = random_log_int(rng, 1, args.max_batch_size)
    max_kv = max_kv_for_batch(args, batch_size)
    lower = min(max_kv, 32_768)
    past_kv_tokens = random_log_int(rng, lower, max_kv)
    return DecodeShape(
        split="test",
        distribution="region_long_kv",
        batch_size=batch_size,
        past_kv_tokens=past_kv_tokens,
    )


def random_high_batch_shape(args: argparse.Namespace, rng: random.Random) -> DecodeShape:
    batch_size = rng.randint(max(128, args.max_batch_size // 4), args.max_batch_size)
    past_kv_tokens = random_log_int(rng, 1, max_kv_for_batch(args, batch_size))
    return DecodeShape(
        split="test",
        distribution="region_high_batch",
        batch_size=batch_size,
        past_kv_tokens=past_kv_tokens,
    )


def random_boundary_shape(args: argparse.Namespace, rng: random.Random) -> DecodeShape:
    batch_size = rng.choice([4, 8, 16, 32, 64, 128, 256, 512, 1024])
    max_kv = max_kv_for_batch(args, batch_size)
    lower = max(1, int(max_kv * 0.70))
    past_kv_tokens = rng.randint(lower, max_kv)
    return DecodeShape(
        split="test",
        distribution="region_batch_kv_boundary",
        batch_size=batch_size,
        past_kv_tokens=past_kv_tokens,
    )


def random_linear_interior_shape(args: argparse.Namespace, rng: random.Random) -> DecodeShape:
    batch_size = rng.randint(1, args.max_batch_size)
    past_kv_tokens = rng.randint(1, max_kv_for_batch(args, batch_size))
    return DecodeShape(
        split="test",
        distribution="mixture_linear_interior",
        batch_size=batch_size,
        past_kv_tokens=past_kv_tokens,
    )


def row_for_shape(shape_id: str, shape: DecodeShape) -> dict[str, str | int]:
    return {
        "shape_id": shape_id,
        "split": shape.split,
        "distribution": shape.distribution,
        "phase": "decode",
        "region": classify_region(shape.batch_size, shape.past_kv_tokens),
        "batch_size": shape.batch_size,
        "past_kv_tokens": shape.past_kv_tokens,
        "attention_kv_tokens": shape.attention_kv_tokens,
        "sum_decode_tokens": shape.sum_decode_tokens,
        "sum_decode_kv_tokens": shape.sum_decode_kv_tokens,
        "attention_work_tokens": shape.attention_work_tokens,
        "static_isl": shape.past_kv_tokens,
        "static_osl": 2,
        "latency_ms": "",
        "status": "planned",
        "error": "",
    }


def write_shapes(path: Path, shapes: list[DecodeShape]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for index, shape in enumerate(shapes):
            writer.writerow(row_for_shape(f"decode_{index:06d}", shape))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("fpm/data/decode_shapes.csv"))
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--val-count", type=int, default=512)
    parser.add_argument("--test-count", type=int, default=512)
    parser.add_argument("--max-batch-size", type=int, default=1024)
    parser.add_argument("--max-kv-tokens", type=int, default=DEFAULT_MAX_KV_TOKENS)
    parser.add_argument(
        "--max-batch-kv-tokens",
        type=int,
        default=DEFAULT_MAX_BATCH_KV_TOKENS,
        help="Optional cap for b * (past_kv + 1). Set 0 to disable.",
    )
    parser.add_argument(
        "--batch-anchors",
        default=",".join(str(value) for value in DEFAULT_BATCH_ANCHORS),
        help="Comma-separated train-grid batch anchors.",
    )
    parser.add_argument(
        "--kv-anchors",
        default=",".join(str(value) for value in DEFAULT_KV_ANCHORS),
        help="Comma-separated train-grid past-KV anchors.",
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
