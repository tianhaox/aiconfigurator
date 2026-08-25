#!/usr/bin/env python3
"""Fetch the host-side probe inputs gen_dummy_models/probes need but the repo
does not ship: HF config originals and tokenizer/processor/custom-code assets.

The SM90 campaign backfilled these by hand ("configs/tokenizers/dummies
backfilled", 9e82e562); this codifies that step so a fresh box (the b200
sweep was one) reproduces it.

    AIC_PROBE_WORKSPACE=<ws> python3 fetch_assets.py --configs   # <ws>/configs/*.json
    AIC_PROBE_WORKSPACE=<ws> python3 fetch_assets.py --assets    # tokenizers etc. into dummy dirs

--configs populates <ws>/configs with <org>_<repo>.json (+ _hfquant.json when
the repo ships one), preferring the verbatim HF config and falling back to the
checkout's aic-core model_configs for gated repos. --assets downloads once per
repo into <ws>/hf_assets/ and RELATIVE-symlinks into every dummy variant dir
(relative links resolve identically on host and under the container's /work
mount). Gated repos are served from the declared public MIRRORS (identical
asset copies); every substitution is recorded in hf_assets/backfill_manifest.json
— provenance, never silent.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.request
from pathlib import Path

WS = Path(os.environ.get("AIC_PROBE_WORKSPACE", Path.cwd()))
HERE = Path(__file__).resolve().parent

ASSET_NAMES = {
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "tokenizer.model", "vocab.json", "merges.txt", "generation_config.json",
    "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json",
    "chat_template.json", "chat_template.jinja", "tiktoken.model",
}
# gated official repo -> public mirror carrying identical tokenizer assets;
# every use lands in backfill_manifest.json["substituted"]. Tokenizer content
# does not alter kernel identity, but substitution is still provenance.
MIRRORS = {
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": "unsloth/Llama-4-Scout-17B-16E-Instruct",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct": "unsloth/Llama-4-Maverick-17B-128E-Instruct",
    "meta-llama/Meta-Llama-3.1-405B": "unsloth/Meta-Llama-3.1-8B",  # 3.1 family shares one tokenizer
    "meta-llama/Meta-Llama-3.1-70B": "unsloth/Meta-Llama-3.1-70B",
    "meta-llama/Meta-Llama-3.1-8B": "unsloth/Meta-Llama-3.1-8B",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
    "Qwen/Qwen3-32B-FP8-Static-PerTensor": "Qwen/Qwen3-32B",
}


def hf_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "aic-probe/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def list_files(repo: str) -> set[str] | None:
    try:
        return {e["path"] for e in json.loads(hf_get(f"https://huggingface.co/api/models/{repo}/tree/main"))
                if e.get("type") == "file"}
    except Exception:
        return None


def fetch_configs() -> None:
    cfg = WS / "configs"
    cfg.mkdir(parents=True, exist_ok=True)
    # repo checkout layout (collector/facts/ -> repo root) or workspace layout
    # (<ws>/probe/ -> <ws>/aic checkout)
    aic_mc = next((p for p in (
        HERE.parent.parent / "aic-core/src/aiconfigurator_core/model_configs",
        WS / "aic/aic-core/src/aiconfigurator_core/model_configs",
    ) if p.is_dir()), HERE.parent.parent / "aic-core/src/aiconfigurator_core/model_configs")
    roster = HERE / "configs" / "repos.txt"
    repos = [ln.split()[0] for ln in roster.read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    missing = []
    for repo in repos:
        dst = cfg / (repo.replace("/", "_") + ".json")
        if not dst.exists():
            try:  # verbatim HF config first — aic-core copies flatten some archs
                dst.write_bytes(hf_get(f"https://huggingface.co/{repo}/resolve/main/config.json"))
            except Exception as e:
                aic = aic_mc / (repo.replace("/", "--") + "_config.json")
                if aic.exists():
                    shutil.copy(aic, dst)
                else:
                    missing.append((repo, str(e)))
                    continue
        hq = cfg / (repo.replace("/", "_") + "_hfquant.json")
        if not hq.exists():
            try:
                hq.write_bytes(hf_get(f"https://huggingface.co/{repo}/resolve/main/hf_quant_config.json"))
            except Exception:
                pass  # most repos ship none
    for f in ("repos.txt", "dsv4_expert_dtypes.json"):
        shutil.copy(HERE / "configs" / f, cfg / f)
    print(f"configs: {len(repos) - len(missing)}/{len(repos)}")
    for repo, err in missing:
        print(f"MISSING {repo}: {err[:100]} — fetch by hand or record a roster exclusion")


def wanted(files: set[str], cfg: dict) -> set[str]:
    w = {f for f in files if f in ASSET_NAMES}
    if cfg.get("auto_map"):  # custom code imports siblings — take every top-level .py
        w |= {f for f in files if f.endswith(".py") and "/" not in f}
    return w


def fetch_dummy_assets() -> None:
    assets = WS / "hf_assets"
    dm = WS / "dummy_models"
    manifest = {"substituted": {}, "missing": {}, "fetched": {}}
    variants = json.loads((dm / "variants_manifest.json").read_text())["variants"]
    by_repo: dict[str, list[dict]] = {}
    for v in variants:
        by_repo.setdefault(v["repo"], []).append(v)

    for repo, vs in sorted(by_repo.items()):
        src_repo = MIRRORS.get(repo, repo)
        files = list_files(src_repo)
        if files is None:
            manifest["missing"][repo] = "repo tree unreadable (gated, no mirror)"
            print(f"MISSING {repo}")
            continue
        adir = assets / repo.replace("/", "_")
        adir.mkdir(parents=True, exist_ok=True)
        vdirs = [p for v in vs for p in dm.glob(f"*/{v['variant']}") if p.is_dir()]
        cfg = json.loads((vdirs[0] / "config.json").read_text()) if vdirs else {}
        got = []
        for f in sorted(wanted(files, cfg)):
            dst = adir / f
            if not dst.exists():
                try:
                    dst.write_bytes(hf_get(f"https://huggingface.co/{src_repo}/resolve/main/{f}"))
                except Exception as e:
                    print(f"  {repo}/{f}: {e}")
                    continue
            got.append(f)
        if src_repo != repo:
            manifest["substituted"][repo] = src_repo
        manifest["fetched"][repo] = got
        for vdir in vdirs:
            for f in got:
                dst = vdir / f
                if not (dst.exists() or dst.is_symlink()):
                    dst.symlink_to(os.path.relpath(adir / f, vdir))
        print(f"ok {repo}: {len(got)} assets{' (mirror ' + src_repo + ')' if src_repo != repo else ''}")

    (assets / "backfill_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nsubstituted={len(manifest['substituted'])} missing={len(manifest['missing'])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", action="store_true", help="fetch HF config originals into <ws>/configs")
    ap.add_argument("--assets", action="store_true", help="backfill tokenizer/custom-code assets into dummy dirs")
    args = ap.parse_args()
    if not (args.configs or args.assets):
        ap.error("pick --configs and/or --assets")
    if args.configs:
        fetch_configs()
    if args.assets:
        fetch_dummy_assets()


if __name__ == "__main__":
    main()
