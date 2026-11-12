"""Build a self-contained Kaggle kernel script: embeds Final_Paper/code (gemnav, swe_local, experiments) as a
base64 tarball, then appends a driver. No credentials are embedded.

python build_kernel.py <driver.py> <out_dir> <slug> [--gpu] [--internet]
"""
import base64
import io
import json
import os
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)

PREAMBLE = r'''
import base64, io, os, sys, tarfile
_B = "{b64}"
os.makedirs("/kaggle/working/code", exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(_B)), mode="r:gz").extractall("/kaggle/tmp/code")
sys.path.insert(0, "/kaggle/tmp/code"); sys.path.insert(0, "/kaggle/tmp/code/experiments")
CODE_DIR = "/kaggle/tmp/code"
'''


def main():
    driver, out_dir, slug = sys.argv[1:4]
    flags = set(sys.argv[4:])
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for sub in ("gemnav", "swe_local", "experiments", "kaggle"):
            for dp, dn, fn in os.walk(os.path.join(CODE, sub)):
                dn[:] = [d for d in dn if d != "__pycache__"]
                for f in fn:
                    if f.endswith(".py"):
                        full = os.path.join(dp, f)
                        tar.add(full, arcname=os.path.relpath(full, CODE))
    b64 = base64.b64encode(buf.getvalue()).decode()
    os.makedirs(out_dir, exist_ok=True)
    code_file = slug + ".py"
    with open(os.path.join(out_dir, code_file), "w") as f:
        f.write(PREAMBLE.replace("{b64}", b64))
        f.write(open(driver).read())
    meta = {"id": f"hemerao/{slug}", "title": slug, "code_file": code_file, "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": "--gpu" in flags, "enable_internet": "--internet" in flags,
            "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
    json.dump(meta, open(os.path.join(out_dir, "kernel-metadata.json"), "w"), indent=1)
    print(out_dir, code_file, len(b64) // 1024, "KiB")


if __name__ == "__main__":
    main()
