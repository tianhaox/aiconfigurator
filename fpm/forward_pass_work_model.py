#!/usr/bin/env python3
"""Evaluate forward-pass work-space latency models on FPM CSV data."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EPS = 1e-9
DEFAULT_BATCH_KV_CAP = 1024 * 1024


@dataclass
class Dataset:
    rows: list[dict[str, str]]
    raw_x: np.ndarray
    work_x: np.ndarray
    local_x: np.ndarray
    y: np.ndarray | None = None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_split_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def float_field(row: dict[str, str], name: str) -> float:
    return float(row[name])


def usable_rows(rows: list[dict[str, str]], splits: set[str], phase: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        if row.get("phase", phase) != phase:
            continue
        if row.get("split") not in splits:
            continue
        try:
            latency = float(row.get("latency_ms", "") or "nan")
        except ValueError:
            continue
        if row.get("status") == "ok" and math.isfinite(latency) and latency > 0:
            out.append(row)
    return out


def shape_key(row: dict[str, str], phase: str) -> tuple[int, ...]:
    if phase == "prefill":
        return (
            int(float(row["batch_size"])),
            int(float(row["new_tokens"])),
            int(float(row["past_kv_tokens"])),
        )
    if phase == "decode":
        return (
            int(float(row["batch_size"])),
            int(float(row["past_kv_tokens"])),
        )
    raise ValueError(f"unsupported phase: {phase}")


def train_plan_keys(path: Path, splits: set[str], phase: str) -> set[tuple[int, ...]]:
    keys: set[tuple[int, ...]] = set()
    for row in read_csv(path):
        if row.get("phase", phase) != phase:
            continue
        if row.get("split") not in splits:
            continue
        keys.add(shape_key(row, phase))
    return keys


def infer_phase(rows: list[dict[str, str]]) -> str:
    for row in rows:
        phase = row.get("phase")
        if phase in {"prefill", "decode"}:
            return phase
    raise ValueError("could not infer phase from CSV")


def prefill_features(row: dict[str, str]) -> tuple[list[float], list[float], list[float]]:
    batch_size = float_field(row, "batch_size")
    new_tokens = float_field(row, "new_tokens")
    past_kv_tokens = float_field(row, "past_kv_tokens")
    total_context_tokens = new_tokens + past_kv_tokens
    sum_tokens = batch_size * new_tokens
    prefix_work = batch_size * new_tokens * past_kv_tokens
    self_work = batch_size * new_tokens * (new_tokens + 1.0) / 2.0
    context_work = prefix_work + self_work
    avg_context = context_work / max(sum_tokens, EPS)
    prefix_ratio = past_kv_tokens / max(total_context_tokens, EPS)

    raw_x = [
        math.log2(batch_size),
        math.log2(new_tokens),
        math.log2(total_context_tokens),
        math.log2(sum_tokens),
        math.log2(max(batch_size * new_tokens * total_context_tokens, EPS)),
        prefix_ratio,
    ]
    work_x = [
        1.0,
        batch_size,
        sum_tokens,
        prefix_work,
        self_work,
    ]
    local_x = [
        math.log2(batch_size),
        math.log2(sum_tokens),
        math.log2(avg_context + 1.0),
        math.log2(context_work + 1.0),
        prefix_ratio,
    ]
    return raw_x, work_x, local_x


def decode_features(row: dict[str, str]) -> tuple[list[float], list[float], list[float]]:
    batch_size = float_field(row, "batch_size")
    past_kv_tokens = float_field(row, "past_kv_tokens")
    attention_kv_tokens = past_kv_tokens + 1.0
    context_work = batch_size * attention_kv_tokens

    raw_x = [
        math.log2(batch_size),
        math.log2(attention_kv_tokens),
        math.log2(context_work),
        min(context_work / DEFAULT_BATCH_KV_CAP, 4.0),
    ]
    work_x = [
        1.0,
        batch_size,
        context_work,
    ]
    local_x = [
        math.log2(batch_size),
        math.log2(attention_kv_tokens),
        math.log2(context_work),
    ]
    return raw_x, work_x, local_x


def build_dataset(rows: list[dict[str, str]], phase: str, *, require_target: bool = True) -> Dataset:
    if not rows:
        raise ValueError("empty dataset")
    raw_features: list[list[float]] = []
    work_features: list[list[float]] = []
    local_features: list[list[float]] = []
    y_values: list[float] = []

    for row in rows:
        if phase == "prefill":
            raw_x, work_x, local_x = prefill_features(row)
        elif phase == "decode":
            raw_x, work_x, local_x = decode_features(row)
        else:
            raise ValueError(f"unsupported phase: {phase}")
        raw_features.append(raw_x)
        work_features.append(work_x)
        local_features.append(local_x)
        if require_target:
            y_values.append(float_field(row, "latency_ms"))

    return Dataset(
        rows=rows,
        raw_x=np.asarray(raw_features, dtype=float),
        work_x=np.asarray(work_features, dtype=float),
        local_x=np.asarray(local_features, dtype=float),
        y=np.asarray(y_values, dtype=float) if require_target else None,
    )


class RawIdwModel:
    """Direct log-latency interpolation in the existing structured shape space."""

    def __init__(self, *, k_neighbors: int, idw_power: float) -> None:
        self.k_neighbors = k_neighbors
        self.idw_power = idw_power
        self.x_min: np.ndarray | None = None
        self.x_scale: np.ndarray | None = None
        self.train_x: np.ndarray | None = None
        self.train_log_y: np.ndarray | None = None

    def fit(self, dataset: Dataset) -> None:
        if dataset.y is None:
            raise ValueError("training dataset requires latency targets")
        self.x_min = np.nanmin(dataset.raw_x, axis=0)
        self.x_scale = np.maximum(np.nanmax(dataset.raw_x, axis=0) - self.x_min, EPS)
        self.train_x = (dataset.raw_x - self.x_min) / self.x_scale
        self.train_log_y = np.log(np.maximum(dataset.y, EPS))

    def predict(self, dataset: Dataset) -> np.ndarray:
        if self.x_min is None or self.x_scale is None:
            raise RuntimeError("model is not fitted")
        if self.train_x is None or self.train_log_y is None:
            raise RuntimeError("model is not fitted")
        query = (dataset.raw_x - self.x_min) / self.x_scale
        log_pred = idw_predict(query, self.train_x, self.train_log_y, self.k_neighbors, self.idw_power)
        return np.maximum(np.exp(log_pred), EPS)


class WorkGlobalAffineModel:
    """Global affine fit in forward-pass work feature space."""

    def __init__(self, *, ridge: float, clip_factor: float) -> None:
        self.ridge = ridge
        self.clip_factor = clip_factor
        self.scale: np.ndarray | None = None
        self.coef: np.ndarray | None = None
        self.y_min: float | None = None
        self.y_max: float | None = None

    def fit(self, dataset: Dataset) -> None:
        if dataset.y is None:
            raise ValueError("training dataset requires latency targets")
        self.scale = feature_scale(dataset.work_x)
        self.y_min = float(np.min(dataset.y))
        self.y_max = float(np.max(dataset.y))
        self.coef = solve_ridge(dataset.work_x / self.scale, dataset.y, self.ridge)

    def predict(self, dataset: Dataset) -> np.ndarray:
        if self.scale is None or self.coef is None:
            raise RuntimeError("model is not fitted")
        pred = (dataset.work_x / self.scale) @ self.coef
        return self._clip(pred)

    def _clip(self, pred: np.ndarray) -> np.ndarray:
        if self.y_min is None or self.y_max is None:
            return np.maximum(pred, EPS)
        return np.clip(
            pred,
            max(self.y_min / self.clip_factor, EPS),
            self.y_max * self.clip_factor,
        )


class WorkLocalAffineModel:
    """Weighted local affine fit in forward-pass work feature space."""

    def __init__(
        self,
        *,
        ridge: float,
        k_neighbors: int,
        idw_power: float,
        clip_factor: float,
    ) -> None:
        self.ridge = ridge
        self.k_neighbors = k_neighbors
        self.idw_power = idw_power
        self.clip_factor = clip_factor
        self.work_scale: np.ndarray | None = None
        self.local_min: np.ndarray | None = None
        self.local_scale: np.ndarray | None = None
        self.train_work_x: np.ndarray | None = None
        self.train_local_x: np.ndarray | None = None
        self.train_y: np.ndarray | None = None
        self.y_min: float | None = None
        self.y_max: float | None = None

    def fit(self, dataset: Dataset) -> None:
        if dataset.y is None:
            raise ValueError("training dataset requires latency targets")
        self.work_scale = feature_scale(dataset.work_x)
        self.local_min = np.nanmin(dataset.local_x, axis=0)
        self.local_scale = np.maximum(np.nanmax(dataset.local_x, axis=0) - self.local_min, EPS)
        self.train_work_x = dataset.work_x / self.work_scale
        self.train_local_x = (dataset.local_x - self.local_min) / self.local_scale
        self.train_y = dataset.y
        self.y_min = float(np.min(dataset.y))
        self.y_max = float(np.max(dataset.y))

    def predict(self, dataset: Dataset) -> np.ndarray:
        if (
            self.work_scale is None
            or self.local_min is None
            or self.local_scale is None
            or self.train_work_x is None
            or self.train_local_x is None
            or self.train_y is None
        ):
            raise RuntimeError("model is not fitted")
        query_local = (dataset.local_x - self.local_min) / self.local_scale
        query_work = dataset.work_x / self.work_scale
        pred = np.zeros(query_work.shape[0], dtype=float)
        k = min(self.k_neighbors, self.train_work_x.shape[0])

        for index, point in enumerate(query_local):
            distances = np.linalg.norm(self.train_local_x - point, axis=1)
            nearest = np.argpartition(distances, k - 1)[:k]
            nearest_distances = distances[nearest]
            exact = nearest_distances <= 1e-12
            if np.any(exact):
                pred[index] = float(np.mean(self.train_y[nearest][exact]))
                continue
            weights = 1.0 / np.maximum(nearest_distances, EPS) ** self.idw_power
            coef = solve_weighted_ridge(
                self.train_work_x[nearest],
                self.train_y[nearest],
                weights,
                self.ridge,
            )
            pred[index] = float(query_work[index] @ coef)
        return self._clip(pred)

    def _clip(self, pred: np.ndarray) -> np.ndarray:
        if self.y_min is None or self.y_max is None:
            return np.maximum(pred, EPS)
        return np.clip(
            pred,
            max(self.y_min / self.clip_factor, EPS),
            self.y_max * self.clip_factor,
        )


class WorkHybridModel:
    """Global work affine baseline plus local interpolation of log residuals."""

    def __init__(
        self,
        *,
        ridge: float,
        k_neighbors: int,
        idw_power: float,
        clip_factor: float,
    ) -> None:
        self.baseline = WorkGlobalAffineModel(ridge=ridge, clip_factor=clip_factor)
        self.k_neighbors = k_neighbors
        self.idw_power = idw_power
        self.local_min: np.ndarray | None = None
        self.local_scale: np.ndarray | None = None
        self.train_local_x: np.ndarray | None = None
        self.train_log_residual: np.ndarray | None = None

    def fit(self, dataset: Dataset) -> None:
        if dataset.y is None:
            raise ValueError("training dataset requires latency targets")
        self.baseline.fit(dataset)
        baseline_pred = self.baseline.predict(dataset)
        self.local_min = np.nanmin(dataset.local_x, axis=0)
        self.local_scale = np.maximum(np.nanmax(dataset.local_x, axis=0) - self.local_min, EPS)
        self.train_local_x = (dataset.local_x - self.local_min) / self.local_scale
        self.train_log_residual = np.log(np.maximum(dataset.y, EPS)) - np.log(np.maximum(baseline_pred, EPS))

    def predict(self, dataset: Dataset) -> np.ndarray:
        if self.local_min is None or self.local_scale is None:
            raise RuntimeError("model is not fitted")
        if self.train_local_x is None or self.train_log_residual is None:
            raise RuntimeError("model is not fitted")
        baseline_pred = self.baseline.predict(dataset)
        query = (dataset.local_x - self.local_min) / self.local_scale
        residual = idw_predict(
            query,
            self.train_local_x,
            self.train_log_residual,
            self.k_neighbors,
            self.idw_power,
        )
        return np.maximum(baseline_pred * np.exp(residual), EPS)


def feature_scale(x: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.nanmax(np.abs(x), axis=0), EPS)
    scale[0] = 1.0
    return scale


def solve_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    penalty = ridge * np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    lhs = x.T @ x + penalty
    rhs = x.T @ y
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def solve_weighted_ridge(x: np.ndarray, y: np.ndarray, weights: np.ndarray, ridge: float) -> np.ndarray:
    weights = np.maximum(weights, EPS)
    wx = x * np.sqrt(weights)[:, None]
    wy = y * np.sqrt(weights)
    return solve_ridge(wx, wy, ridge)


def idw_predict(
    query: np.ndarray,
    train_x: np.ndarray,
    train_values: np.ndarray,
    k_neighbors: int,
    idw_power: float,
) -> np.ndarray:
    if k_neighbors <= 0:
        raise ValueError("k_neighbors must be positive")
    out = np.zeros(query.shape[0], dtype=float)
    k = min(k_neighbors, train_x.shape[0])
    for index, point in enumerate(query):
        distances = np.linalg.norm(train_x - point, axis=1)
        nearest = np.argpartition(distances, k - 1)[:k]
        nearest_distances = distances[nearest]
        exact = nearest_distances <= 1e-12
        if np.any(exact):
            out[index] = float(np.mean(train_values[nearest][exact]))
            continue
        weights = 1.0 / np.maximum(nearest_distances, EPS) ** idw_power
        out[index] = float(np.sum(weights * train_values[nearest]) / np.sum(weights))
    return out


def classify_region(row: dict[str, str], phase: str) -> str:
    if row.get("region"):
        return row["region"]
    if phase == "prefill":
        batch_size = int(float(row["batch_size"]))
        new_tokens = int(float(row["new_tokens"]))
        past_kv_tokens = int(float(row["past_kv_tokens"]))
        total_context_tokens = new_tokens + past_kv_tokens
        sum_prefill_tokens = batch_size * new_tokens
        prefix_ratio = past_kv_tokens / max(total_context_tokens, 1)
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

    batch_size = int(float(row["batch_size"]))
    past_kv_tokens = int(float(row["past_kv_tokens"]))
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


def evaluate(model, dataset: Dataset) -> tuple[dict[str, float], np.ndarray]:
    if dataset.y is None:
        raise ValueError("evaluation dataset requires latency targets")
    pred = model.predict(dataset)
    ape = np.abs(pred - dataset.y) / np.maximum(dataset.y, EPS) * 100.0
    return (
        {
            "count": float(len(dataset.rows)),
            "mape": float(np.mean(ape)),
            "mdape": float(np.median(ape)),
            "p90_ape": float(np.percentile(ape, 90)),
            "p95_ape": float(np.percentile(ape, 95)),
            "p99_ape": float(np.percentile(ape, 99)),
            "max_ape": float(np.max(ape)),
        },
        pred,
    )


def print_group_metrics(dataset: Dataset, pred: np.ndarray, phase: str, group_by: str) -> None:
    if dataset.y is None:
        return
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(dataset.rows):
        key = classify_region(row, phase) if group_by == "region" else row.get(group_by, "")
        groups.setdefault(key or "unknown", []).append(index)
    for key in sorted(groups):
        indexes = np.asarray(groups[key], dtype=int)
        actual = dataset.y[indexes]
        group_pred = pred[indexes]
        ape = np.abs(group_pred - actual) / np.maximum(actual, EPS) * 100.0
        print(
            f"    {group_by}={key}: n={len(indexes)} "
            f"MAPE={float(np.mean(ape)):.2f}% MdAPE={float(np.median(ape)):.2f}% "
            f"P90={float(np.percentile(ape, 90)):.2f}% Max={float(np.max(ape)):.2f}%"
        )


def make_model(name: str, args: argparse.Namespace):
    if name == "raw_idw":
        return RawIdwModel(k_neighbors=args.idw_neighbors, idw_power=args.idw_power)
    if name == "work_global":
        return WorkGlobalAffineModel(ridge=args.ridge, clip_factor=args.clip_factor)
    if name == "work_local":
        return WorkLocalAffineModel(
            ridge=args.ridge,
            k_neighbors=args.local_neighbors,
            idw_power=args.local_power,
            clip_factor=args.clip_factor,
        )
    if name == "work_hybrid":
        return WorkHybridModel(
            ridge=args.ridge,
            k_neighbors=args.idw_neighbors,
            idw_power=args.idw_power,
            clip_factor=args.clip_factor,
        )
    raise ValueError(f"unsupported model: {name}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path)
    parser.add_argument(
        "--train-plan",
        type=Path,
        help="Optional planned shape CSV used to filter training rows by shape.",
    )
    parser.add_argument("--phase", choices=["auto", "prefill", "decode"], default="auto")
    parser.add_argument("--train-splits", default="train")
    parser.add_argument("--eval-splits", default="train,val,test")
    parser.add_argument(
        "--models",
        default="raw_idw,work_global,work_local,work_hybrid",
        help="Comma-separated model list.",
    )
    parser.add_argument("--idw-neighbors", type=int, default=8)
    parser.add_argument("--idw-power", type=float, default=2.0)
    parser.add_argument("--local-neighbors", type=int, default=24)
    parser.add_argument("--local-power", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--clip-factor", type=float, default=100.0)
    parser.add_argument("--group-by", default="")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    rows = read_csv(args.data)
    eval_rows_source = read_csv(args.eval_data) if args.eval_data is not None else rows
    phase = infer_phase(rows) if args.phase == "auto" else args.phase
    train_rows = usable_rows(rows, set(parse_split_list(args.train_splits)), phase)
    if args.train_plan is not None:
        keys = train_plan_keys(args.train_plan, set(parse_split_list(args.train_splits)), phase)
        train_rows = [row for row in train_rows if shape_key(row, phase) in keys]
    if len(train_rows) < 5:
        raise SystemExit(f"Need at least 5 successful training rows, found {len(train_rows)}.")

    train_dataset = build_dataset(train_rows, phase)
    for model_name in parse_split_list(args.models):
        model = make_model(model_name, args)
        model.fit(train_dataset)
        print(f"{model_name}:")
        for split in parse_split_list(args.eval_splits):
            split_rows = usable_rows(eval_rows_source, {split}, phase)
            if not split_rows:
                print(f"  {split}: no successful rows")
                continue
            dataset = build_dataset(split_rows, phase)
            metrics, pred = evaluate(model, dataset)
            print(
                f"  {split}: n={int(metrics['count'])} "
                f"MAPE={metrics['mape']:.2f}% MdAPE={metrics['mdape']:.2f}% "
                f"P90={metrics['p90_ape']:.2f}% P95={metrics['p95_ape']:.2f}% "
                f"P99={metrics['p99_ape']:.2f}% Max={metrics['max_ape']:.2f}%"
            )
            if args.group_by:
                print_group_metrics(dataset, pred, phase, args.group_by)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
