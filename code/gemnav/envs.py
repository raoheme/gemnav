"""Task execution environments. The agent always edits a *host* git working copy (`workdir`) so that
tree-sitter, embeddings and pylsp see real files; tests run inside the task's own environment.

  LocalEnv  : workdir is the environment (unit tests, plain repos)
  DockerEnv : official SWE-bench instance image (`swebench/sweb.eval.x86_64.<id>`) via Docker Desktop.
              /testbed is copied out once; changed files are copied back in before each test run.
  RootfsEnv : Kaggle — image unpacked as a rootfs (swe_paper compute/kaggle/common/rootfs.py Sandbox);
              workdir is <root>/testbed itself, commands run under chroot/proot.
All `run` calls return (returncode, output) and never raise on non-zero exit.
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import tarfile
import time

ACTIVATE = "source /opt/miniconda3/bin/activate testbed 2>/dev/null || true; "


def image_for(iid: str) -> str:
    return "swebench/sweb.eval.x86_64." + iid.replace("__", "_1776_").lower() + ":latest"


class LocalEnv:
    def __init__(self, workdir: str):
        self.workdir = os.path.abspath(workdir)

    def run(self, cmd: str, timeout: int = 600) -> tuple[int, str]:
        try:
            r = subprocess.run(["bash", "-c", cmd], cwd=self.workdir, capture_output=True, text=True, timeout=timeout)
            return r.returncode, (r.stdout + r.stderr)
        except subprocess.TimeoutExpired:
            return 124, "[TIMEOUT]"

    def put_external(self, name: str, text: str) -> str:
        """Write a harness-owned file OUTSIDE the repo (frozen reproduction script). Returns its in-env path."""
        d = os.path.join(os.path.dirname(self.workdir), "_gemnav_ext")
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, name)
        if os.path.exists(p):
            os.chmod(p, 0o644)
        open(p, "w").write(text)
        os.chmod(p, 0o444)
        return p

    def sync(self, changed: list[str], deleted: list[str] = ()) -> None:
        pass

    def close(self) -> None:
        pass


def _dk(*args, timeout=900, inp: bytes | None = None) -> tuple[int, bytes]:
    try:
        r = subprocess.run(["docker", *args], input=inp, capture_output=True, timeout=timeout)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return 124, b"[TIMEOUT]"


class DockerEnv:
    def __init__(self, iid: str, host_root: str, mem: str = "2g", pull_timeout: int = 7200):
        self.iid, self.image = iid, image_for(iid)
        self.name = ("gemnav_" + iid).replace("__", "_")[:100]
        self.timings = {}
        t0 = time.time()
        if _dk("image", "inspect", self.image, timeout=60)[0] != 0:
            rc, o = _dk("pull", self.image, timeout=pull_timeout)
            if rc:
                raise RuntimeError("docker pull failed: " + o[-400:].decode(errors="replace"))
        self.timings["pull_s"] = time.time() - t0
        _dk("rm", "-f", self.name, timeout=120)
        rc, o = _dk("run", "-d", "--name", self.name, "--memory", mem, self.image, "sleep", "infinity", timeout=300)
        if rc:
            raise RuntimeError("docker run failed: " + o[-400:].decode(errors="replace"))
        self._sh("git config --global --add safe.directory /testbed; mkdir -p /gemnav_ext")
        # copy /testbed out as a tar stream (preserves modes/symlinks)
        t0 = time.time()
        self.workdir = os.path.join(host_root, iid)
        shutil.rmtree(self.workdir, ignore_errors=True)
        os.makedirs(self.workdir)
        # streamed (no full in-memory buffer): docker exec tar | tarfile stream mode
        p = subprocess.Popen(["docker", "exec", self.name, "tar", "-C", "/testbed", "-cf", "-", "."], stdout=subprocess.PIPE)
        with tarfile.open(fileobj=p.stdout, mode="r|") as t:
            if hasattr(tarfile, "data_filter"):
                t.extractall(self.workdir, filter="fully_trusted")
            else:
                t.extractall(self.workdir)
        if p.wait(timeout=900):
            raise RuntimeError("copy-out failed")
        self.timings["copy_out_s"] = time.time() - t0

    def _sh(self, cmd: str, timeout: int = 600) -> tuple[int, str]:
        rc, o = _dk("exec", self.name, "bash", "-c", cmd, timeout=timeout)
        return rc, o.decode("utf-8", "replace")

    def run(self, cmd: str, timeout: int = 600) -> tuple[int, str]:
        return self._sh(ACTIVATE + "cd /testbed; " + cmd, timeout)

    def put_external(self, name: str, text: str) -> str:
        p = f"/gemnav_ext/{name}"
        self._sh(f"chmod 644 {p} 2>/dev/null; true")
        _dk("exec", "-i", self.name, "sh", "-c", f"cat > {p} && chmod 444 {p}", inp=text.encode(), timeout=120)
        return p

    def sync(self, changed: list[str], deleted: list[str] = ()) -> None:
        if changed:
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as t:
                for rel in changed:
                    t.add(os.path.join(self.workdir, rel), arcname=rel)
            _dk("exec", "-i", self.name, "tar", "-C", "/testbed", "-xf", "-", inp=buf.getvalue(), timeout=300)
        for rel in deleted:
            self._sh(f"rm -f /testbed/{rel!s}")

    def close(self, remove_image: bool = False) -> None:
        _dk("rm", "-f", self.name, timeout=120)
        shutil.rmtree(self.workdir, ignore_errors=True)
        if remove_image:
            _dk("rmi", "-f", self.image, timeout=300)


class RootfsEnv:
    """Kaggle: `sandbox` is swe_paper's rootfs.Sandbox (has .root and .run(script, timeout))."""

    def __init__(self, sandbox):
        self.sb = sandbox
        self.workdir = os.path.join(str(sandbox.root), "testbed")
        os.makedirs(os.path.join(str(sandbox.root), "gemnav_ext"), exist_ok=True)

    def run(self, cmd: str, timeout: int = 600) -> tuple[int, str]:
        r = self.sb.run(ACTIVATE + "cd /testbed; " + cmd, timeout=timeout)
        if isinstance(r, tuple):
            return r[0], r[1]
        return r.returncode, (r.stdout or "") + (r.stderr or "")

    def put_external(self, name: str, text: str) -> str:
        p = os.path.join(str(self.sb.root), "gemnav_ext", name)
        if os.path.exists(p):
            os.chmod(p, 0o644)
        open(p, "w").write(text)
        os.chmod(p, 0o444)
        return f"/gemnav_ext/{name}"

    def sync(self, changed, deleted=()):
        pass

    def close(self):
        self.sb.close()
