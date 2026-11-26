#!/usr/bin/env python3
"""Build, slim (FROM scratch), audit, and export the minimal Thor llama.cpp runtime image.

Constructs a pure dependency-closure rootfs directly from the R10-optimized `llama-server`
binary and static ARM64 `healthcheck` binary — ZERO Python, ZERO compilers/headers, ZERO
package managers, ZERO static archives, ZERO model weights — builds a single-layer
`FROM scratch` Docker image (`--provenance=false --sbom=false`), exports compressed `.tar.gz`,
extracts exact Docker IDs (`imageId`, `configImageId`, `runtimeImageSha256`, `unpackedSize`),
and writes a complete layer/content audit JSON.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any


FORBIDDEN_SUFFIXES = (".gguf", ".safetensors", ".pt", ".pth", ".onnx", ".ckpt", ".a", ".whl", ".pyc", ".h", ".hpp", ".cuh")
FORBIDDEN_BINS = ("nvcc", "ptxas", "cmake", "ninja", "gcc", "g++", "pip", "pip3", "apt", "apt-get", "dpkg", "python", "python3", "curl", "wget")
# Driver-injected libraries provided at runtime by NVIDIA Container Runtime (--gpus all)
HOST_INJECTED_PREFIXES = ("libcuda.so", "libnvidia-ml.so", "libnvidia-", "libnvdla", "libnvrm")


def sha256_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    total = 0
    with path.open("rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
            total += len(chunk)
    return h.hexdigest(), total


def resolve_ldd_closure(binaries: list[Path], ld_library_path: str) -> list[tuple[str, Path]]:
    """Return list of (soname_or_abs_path, real_resolved_file) needed by binaries."""
    env = {**os.environ, "LD_LIBRARY_PATH": ld_library_path}
    seen: dict[str, Path] = {}
    queue: list[Path] = list(binaries)
    visited_bins: set[Path] = set()

    while queue:
        cur = queue.pop(0).resolve()
        if cur in visited_bins or not cur.is_file():
            continue
        visited_bins.add(cur)
        out = subprocess.check_output(["ldd", str(cur)], env=env, text=True, stderr=subprocess.STDOUT)
        for line in out.splitlines():
            line = line.strip()
            if not line or "linux-vdso.so" in line or "not a dynamic executable" in line:
                continue
            m_arrow = re.match(r"^(\S+)\s+=>\s+(\S+)\s+\(0x", line)
            m_direct = re.match(r"^(/\S+)\s+\(0x", line)
            if m_arrow:
                soname, target = m_arrow.group(1), m_arrow.group(2)
                if any(soname.startswith(p) for p in HOST_INJECTED_PREFIXES):
                    continue
                tp = Path(target)
                if tp.is_file() and soname not in seen:
                    seen[soname] = tp
                    queue.append(tp)
            elif m_direct:
                absp = Path(m_direct.group(1))
                if absp.is_file() and str(absp) not in seen:
                    seen[str(absp)] = absp
                    queue.append(absp)
            elif "not found" in line:
                raise RuntimeError(f"Unresolved shared library in {cur}: {line}")

    return sorted(seen.items(), key=lambda x: x[0])


def copy_with_soname(dest_dir: Path, soname: str, real_path: Path) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    real_resolved = real_path.resolve()
    dst_real = dest_dir / real_resolved.name
    if not dst_real.exists():
        shutil.copy2(real_resolved, dst_real)
    # Create symlink for soname if different
    link_name = Path(soname).name
    dst_link = dest_dir / link_name
    if link_name != real_resolved.name and not dst_link.exists():
        os.symlink(real_resolved.name, dst_link)
    return dst_real.stat().st_size


def build_rootfs(
    stage_dir: Path,
    server_bin: Path,
    build_bin_dir: Path,
    healthcheck_bin: Path,
    llama_license: Path,
    target: str = "thor",
) -> dict[str, Any]:
    rootfs = stage_dir / "rootfs"
    if rootfs.exists():
        shutil.rmtree(rootfs)
    rootfs.mkdir(parents=True)

    for rel, mode in (
        ("runtime", 0o755),
        ("runtime/lib", 0o755),
        ("lib", 0o755),
        ("lib/aarch64-linux-gnu", 0o755),
        ("usr/lib/aarch64-linux-gnu", 0o755),
        ("licenses", 0o755),
        ("tmp", 0o1777),
        ("var/tmp", 0o1777),
        ("root", 0o755),
        ("root/.cache", 0o755),
        ("root/.nv", 0o755),
        ("root/.triton", 0o755),
        ("model", 0o755),
        ("run", 0o755),
    ):
        d = rootfs / rel
        d.mkdir(parents=True, exist_ok=True)
        os.chmod(d, mode)

    # Copy llama-server and healthcheck
    dst_srv = rootfs / "runtime" / "llama-server"
    shutil.copy2(server_bin, dst_srv)
    os.chmod(dst_srv, 0o755)
    subprocess.run(["strip", "--strip-unneeded", str(dst_srv)], check=False)

    dst_hc = rootfs / "runtime" / "healthcheck"
    shutil.copy2(healthcheck_bin, dst_hc)
    os.chmod(dst_hc, 0o755)

    # Also find any runtime-loaded llama/ggml/mtmd shared objects in build_bin_dir
    extra_bins = [server_bin]
    for so in sorted(build_bin_dir.glob("lib*.so*")):
        if so.is_file() and not so.is_symlink():
            extra_bins.append(so)

    ld_path = f"{build_bin_dir}:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/lib/aarch64-linux-gnu"
    deps = resolve_ldd_closure(extra_bins, ld_path)

    # Also ensure libcublasLt.so.13 (dlopened by libcublas) is included if present in /usr/local/cuda/lib64
    cuda_lib_dir = Path("/usr/local/cuda/lib64")
    for extra_pat in ("libcublasLt.so.13*", "libcublas.so.13*", "libcudart.so.13*"):
        for cand in sorted(cuda_lib_dir.glob(extra_pat)):
            if cand.is_file() and not cand.is_symlink():
                soname = cand.name.split(".so.")[0] + ".so." + cand.name.split(".so.")[1].split(".")[0]
                if not any(s == soname for s, _ in deps):
                    deps.append((soname, cand))

    copied_libs: list[dict[str, Any]] = []
    for soname, real_path in deps:
        if soname.startswith("/lib/ld-linux"):
            # Dynamic linker MUST live at its exact PT_INTERP path (/lib/ld-linux-aarch64.so.1)
            dst_interp = rootfs / soname.lstrip("/")
            dst_interp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(real_path.resolve(), dst_interp)
            os.chmod(dst_interp, 0o755)
            copied_libs.append({"soname": soname, "path": str(dst_interp.relative_to(rootfs)), "bytes": dst_interp.stat().st_size})
        else:
            sz = copy_with_soname(rootfs / "runtime" / "lib", soname, real_path)
            copied_libs.append({"soname": soname, "path": f"runtime/lib/{Path(soname).name}", "bytes": sz})

    # Copy build_bin_dir lib*.so* symlinks & targets into /runtime/lib
    for so in sorted(build_bin_dir.glob("lib*.so*")):
        if so.is_file() and not so.is_symlink():
            dst = rootfs / "runtime" / "lib" / so.name
            if not dst.exists():
                shutil.copy2(so, dst)
                subprocess.run(["strip", "--strip-unneeded", str(dst)], check=False)
                copied_libs.append({"soname": so.name, "path": f"runtime/lib/{so.name}", "bytes": dst.stat().st_size})
        elif so.is_symlink():
            dst = rootfs / "runtime" / "lib" / so.name
            if not dst.exists():
                os.symlink(os.readlink(so), dst)

    # Licenses & provenance notice
    if llama_license.is_file():
        shutil.copy2(llama_license, rootfs / "licenses" / "LICENSE-llama.cpp-MIT.txt")
    target_label = "Thor (sm_110a)" if target == "thor" else "Orin (sm_87)"
    patch_desc = "Thor sm_110a optimization patches R1-R20" if target == "thor" else "Thor+Orin sm_87 optimization patches R1-R19 + O1-O5"
    (rootfs / "licenses" / "NOTICE.txt").write_text(
        f"Qwen3.5-9B-The-Defiant-Fable {target_label} Runtime (FROM scratch)\n"
        "- Upstream model: DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF (Apache-2.0)\n"
        "- Base model: Qwen/Qwen3.5-9B (Apache-2.0)\n"
        f"- Inference engine: ggml-org/llama.cpp (MIT) with {patch_desc}\n"
        "- Zero model weights bundled; weights are mounted read-only at /model.\n",
        encoding="utf-8",
    )

    # Compute unpacked size & run negative audit
    unpacked_bytes = 0
    file_count = 0
    violations: list[str] = []
    for p in rootfs.rglob("*"):
        if p.is_file() and not p.is_symlink():
            st = p.stat()
            unpacked_bytes += st.st_size
            file_count += 1
            rel = p.relative_to(rootfs).as_posix()
            if p.suffix.lower() in FORBIDDEN_SUFFIXES:
                violations.append(f"forbidden_suffix:{rel}")
            if p.name in FORBIDDEN_BINS:
                violations.append(f"forbidden_bin:{rel}")

    if violations:
        raise RuntimeError(f"Slim rootfs audit FAILED with violations: {violations}")

    return {
        "rootfs": str(rootfs),
        "fileCount": file_count,
        "unpackedBytes": unpacked_bytes,
        "libs": copied_libs,
        "violations": violations,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=("thor", "orin"), default="thor")
    ap.add_argument("--server-bin", required=True)
    ap.add_argument("--healthcheck-bin", required=True)
    ap.add_argument("--llama-license", required=True)
    ap.add_argument("--image", default="")
    ap.add_argument("--stage-dir", required=True)
    ap.add_argument("--out-archive", required=True)
    ap.add_argument("--out-audit", required=True)
    ap.add_argument("--env", action="append", default=[])
    args = ap.parse_args()
    if not args.image:
        args.image = f"local/qwen35-9b-defiant-fable-{args.target}:0.2.0"

    stage_dir = Path(args.stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    server_bin = Path(args.server_bin)
    build_bin_dir = server_bin.parent

    rootfs_info = build_rootfs(
        stage_dir=stage_dir,
        server_bin=server_bin,
        build_bin_dir=build_bin_dir,
        healthcheck_bin=Path(args.healthcheck_bin),
        llama_license=Path(args.llama_license),
        target=args.target,
    )

    default_envs: dict[str, str] = {
        "LD_LIBRARY_PATH": "/runtime/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/lib/aarch64-linux-gnu",
        "GGML_CUDA_MMQ_MAX_BATCH": "256",
        "GGML_CUDA_MMQ_FEW_MAX_NCG": "4",
        "GGML_CUDA_CUBLAS_COMPUTE_TYPE": "quant_f16",
        "GGML_CUDA_F16_PREFER_F32_OUT": "1",
        "GGML_CUDA_GLU_VEC4": "1",
        "GGML_CUDA_BINBCAST_VEC4": "1",
        "GGML_CUDA_ACT_CACHE": "1",
        "GGML_CUDA_ASYNC_DEQUANT": "0",
        "LLAMA_MTP_DRAFT_HEAD_TYPE": "q2_k",
        "LLAMA_MTP_ADAPTIVE": "1",
        "LLAMA_MTP_ADAPTIVE_K_LOW": "3",
        "LLAMA_MTP_ADAPTIVE_K_MID": "2",
        "LLAMA_MTP_ADAPTIVE_K_HIGH": "2",
        "GGML_CUDA_PDL": "1" if args.target == "thor" else "0",
    }
    if args.target == "orin":
        default_envs["OMP_NUM_THREADS"] = "4"
    for kv in args.env:
        if "=" in kv:
            k, v = kv.split("=", 1)
            default_envs[k] = v

    env_lines = "".join(f"ENV {k}={v}\n" for k, v in default_envs.items())
    dockerfile = stage_dir / "Dockerfile"
    dockerfile.write_text(
        "FROM scratch\n"
        "COPY rootfs/ /\n"
        + env_lines
        + "WORKDIR /runtime\n"
        "EXPOSE 18156\n"
        'ENTRYPOINT ["/runtime/llama-server"]\n',
        encoding="utf-8",
    )

    # Build single-layer FROM scratch image (provenance=false, sbom=false)
    build_cmd = [
        "docker", "build",
        "--platform", "linux/arm64",
        "--provenance=false",
        "--sbom=false",
        "-t", args.image,
        "-f", str(dockerfile),
        str(stage_dir),
    ]
    res = subprocess.run(build_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        # Fallback if legacy builder doesn't recognize --provenance/--sbom
        build_cmd_legacy = [
            "docker", "build",
            "--platform", "linux/arm64",
            "-t", args.image,
            "-f", str(dockerfile),
            str(stage_dir),
        ]
        subprocess.run(build_cmd_legacy, check=True)

    # Inspect built image
    insp = json.loads(subprocess.check_output(["docker", "image", "inspect", args.image], text=True))[0]
    image_id = insp["Id"]
    layers = insp.get("RootFS", {}).get("Layers", [])
    unpacked_size = int(insp.get("Size", rootfs_info["unpackedBytes"]))

    # Save & compress image archive
    out_archive = Path(args.out_archive)
    out_archive.parent.mkdir(parents=True, exist_ok=True)
    raw_tar = stage_dir / "image.tar"
    subprocess.run(["docker", "save", "-o", str(raw_tar), args.image], check=True)

    # Inspect manifest.json inside docker save tar to get exact configImageId
    config_image_id = image_id
    with tarfile.open(raw_tar, "r") as tf:
        mf = json.loads(tf.extractfile("manifest.json").read().decode("utf-8"))
        cfg_name = mf[0]["Config"]
        cfg_hex = Path(cfg_name).stem.replace("blobs/sha256/", "")
        if len(cfg_hex) == 64:
            config_image_id = f"sha256:{cfg_hex}"

    # Compress with pigz/gzip -6
    with raw_tar.open("rb") as fin, out_archive.open("wb") as fout:
        with gzip.GzipFile(filename="", mode="wb", fileobj=fout, compresslevel=6, mtime=0) as gz:
            shutil.copyfileobj(fin, gz, length=4 << 20)
    raw_tar.unlink(missing_ok=True)

    archive_sha256, archive_size = sha256_file(out_archive)
    runtime_image_sha256 = image_id.replace("sha256:", "")

    audit = {
        "schema": "model-host.runtime-slim-audit.v1",
        "target": args.target,
        "image": args.image,
        "imageId": image_id,
        "configImageId": config_image_id,
        "runtimeImageSha256": runtime_image_sha256,
        "archiveName": out_archive.name,
        "archiveSizeBytes": archive_size,
        "archiveSha256": archive_sha256,
        "unpackedSizeBytes": unpacked_size,
        "rootfsFileCount": rootfs_info["fileCount"],
        "layerCount": len(layers),
        "layers": layers,
        "baseImage": "scratch",
        "negativeAuditPassed": True,
        "violations": [],
        "containsWeights": False,
        "containsPython": False,
        "containsCompilersOrHeaders": False,
        "containsStaticArchives": False,
        "hostDriverInjectedPrefixes": list(HOST_INJECTED_PREFIXES),
        "libraries": rootfs_info["libs"],
    }

    out_audit = Path(args.out_audit)
    out_audit.parent.mkdir(parents=True, exist_ok=True)
    out_audit.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
