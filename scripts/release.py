#!/usr/bin/env python3
"""Resumable dual-source release pipeline for cloud.lazycat.aipod.qwen35-9b-defiant-fable.

Persists stage state to `artifacts/release-work/pipeline.json`.
Stages:
  1. validate            - Go tests (-race) + vet + manifest schema checks
  2. build_control_bundle - Build amd64 controller + deterministic .tar.gz + update control-manifest.json
  3. verify_artifacts    - Verify Thor runtime archive/imageId and model SHA256s are locked
  4. publish             - Verify 40-char immutable HF + ModelScope revisions & control bundle URLs
  5. build_lpk           - Build thin LPK via `lzc-cli project build`
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT / "artifacts" / "release-work" / "pipeline.json"
CONTROL_BINARY_NAME = "qwen35-9b-lpk"


def sha256_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        while chunk := f.read(4 << 20):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_pipeline() -> dict[str, Any]:
    if PIPELINE_PATH.is_file():
        return json.loads(PIPELINE_PATH.read_text(encoding="utf-8"))
    return {"schema": "qwen35-9b.release-pipeline.v1", "steps": {}}


def run_step(state: dict[str, Any], name: str, fn: Callable[[], dict[str, Any] | None]) -> None:
    row = state["steps"].get(name, {})
    row["status"] = "running"
    row["startedUnix"] = time.time()
    state["steps"][name] = row
    write_json(PIPELINE_PATH, state)
    try:
        details = fn() or {}
        row["status"] = "passed"
        row["details"] = details
        row["finishedUnix"] = time.time()
        row.pop("error", None)
    except Exception as exc:
        row["status"] = "blocked" if isinstance(exc, RuntimeError) else "failed"
        row["error"] = str(exc)
        row["finishedUnix"] = time.time()
        write_json(PIPELINE_PATH, state)
        raise
    write_json(PIPELINE_PATH, state)


def stage_validate() -> dict[str, Any]:
    subprocess.run(["go", "test", "-race", "./...", "-count=1"], cwd=ROOT, check=True)
    subprocess.run(["go", "vet", "./..."], cwd=ROOT, check=True)
    models = json.loads((ROOT / "manifests" / "models.json").read_text(encoding="utf-8"))
    runtimes = json.loads((ROOT / "manifests" / "runtimes.json").read_text(encoding="utf-8"))
    caches = json.loads((ROOT / "manifests" / "caches.json").read_text(encoding="utf-8"))
    return {
        "modelsStatus": models.get("status"),
        "runtimesStatus": runtimes.get("status"),
        "cachesStatus": caches.get("status"),
    }


def stage_build_control_bundle() -> dict[str, Any]:
    dist_dir = ROOT / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    bin_path = dist_dir / f"{CONTROL_BINARY_NAME}-amd64"
    subprocess.run(
        ["go", "build", "-trimpath", "-ldflags", "-s -w", "-o", str(bin_path), "./cmd/controller"],
        cwd=ROOT,
        env={**os.environ, "CGO_ENABLED": "0", "GOOS": "linux", "GOARCH": "amd64"},
        check=True,
    )
    bin_sha, bin_size = sha256_file(bin_path)

    bundle_path = dist_dir / f"{CONTROL_BINARY_NAME}-amd64-control.tar.gz"
    with bundle_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(mode="w", fileobj=gz, format=tarfile.GNU_FORMAT) as tf:
                ti = tarfile.TarInfo(CONTROL_BINARY_NAME)
                ti.uid = 0
                ti.gid = 0
                ti.mtime = 0
                ti.mode = 0o755
                ti.size = bin_size
                with bin_path.open("rb") as fbin:
                    tf.addfile(ti, fileobj=fbin)

    bundle_sha, bundle_size = sha256_file(bundle_path)
    ctrl_manifest_path = ROOT / "lpk" / "content" / "control-manifest.json"
    ctrl = json.loads(ctrl_manifest_path.read_text(encoding="utf-8"))
    ctrl["binarySha256"] = bin_sha
    ctrl["size"] = bundle_size
    ctrl["sha256"] = bundle_sha
    write_json(ctrl_manifest_path, ctrl)
    return {
        "binary": str(bin_path.relative_to(ROOT)),
        "binarySha256": bin_sha,
        "bundle": str(bundle_path.relative_to(ROOT)),
        "bundleSize": bundle_size,
        "bundleSha256": bundle_sha,
    }


def stage_verify_device_artifacts() -> dict[str, Any]:
    models = json.loads((ROOT / "manifests" / "models.json").read_text(encoding="utf-8"))
    runtimes = json.loads((ROOT / "manifests" / "runtimes.json").read_text(encoding="utf-8"))
    missing = []
    for g in models.get("models", []):
        if not g.get("files"):
            missing.append(f"models.json:{g['id']} files empty")
    for target in ("thor", "orin"):
        rt = runtimes.get("targets", {}).get(target, {})
        if not rt.get("sha256") or not rt.get("imageId") or not rt.get("runtimeImageSha256"):
            missing.append(f"runtimes.json:{target} runtime archive/imageId not yet locked")
    if missing:
        raise RuntimeError("Artifact lock incomplete: " + "; ".join(missing))
    return {"thorLocked": True, "orinLocked": True}


def stage_publish_dual_source() -> dict[str, Any]:
    models = json.loads((ROOT / "manifests" / "models.json").read_text(encoding="utf-8"))
    runtimes = json.loads((ROOT / "manifests" / "runtimes.json").read_text(encoding="utf-8"))
    ctrl = json.loads((ROOT / "lpk" / "content" / "control-manifest.json").read_text(encoding="utf-8"))
    unpinned = []
    for g in models.get("models", []):
        for prov in ("huggingface", "modelscope"):
            if len(g.get("sources", {}).get(prov, {}).get("revision", "")) != 40:
                unpinned.append(f"model:{g['id']}:{prov}")
    for target, rt in runtimes.get("targets", {}).items():
        for prov in ("huggingface", "modelscope"):
            if len(rt.get("sources", {}).get(prov, {}).get("revision", "")) != 40:
                unpinned.append(f"runtime:{target}:{prov}")
    ctrl_sha = ctrl.get("sha256", "")
    if not ctrl.get("huggingfaceUrl") or not ctrl.get("modelscopeUrl") or not ctrl_sha:
        unpinned.append("control-manifest URLs/sha256 missing")
    if unpinned:
        raise RuntimeError("Dual-source revisions incomplete: " + ", ".join(unpinned))
    return {"dualSourcePinned": True}


def stage_build_lpk() -> dict[str, Any]:
    stage_publish_dual_source()
    lzc_cli = subprocess.run(["which", "lzc-cli"], capture_output=True, text=True)
    if lzc_cli.returncode != 0:
        raise RuntimeError("lzc-cli not found in PATH")
    subprocess.run(["lzc-cli", "project", "build"], cwd=ROOT, check=True)
    return {"built": True}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage",
        choices=("validate", "build_control_bundle", "verify_artifacts", "publish", "build_lpk", "all"),
        default="all",
    )
    args = p.parse_args()
    state = load_pipeline()
    stages = {
        "validate": stage_validate,
        "build_control_bundle": stage_build_control_bundle,
        "verify_artifacts": stage_verify_device_artifacts,
        "publish": stage_publish_dual_source,
        "build_lpk": stage_build_lpk,
    }
    order = ["validate", "build_control_bundle", "verify_artifacts", "publish", "build_lpk"]
    to_run = order if args.stage == "all" else [args.stage]
    for name in to_run:
        try:
            run_step(state, name, stages[name])
        except RuntimeError as gate_err:
            print(f"[gate:{name}] BLOCKED: {gate_err}", file=sys.stderr, flush=True)
            sys.exit(2)
    print(f"[release] Stage(s) {', '.join(to_run)} passed.", flush=True)


if __name__ == "__main__":
    main()
