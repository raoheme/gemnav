# Copied from swe_paper/compute/kaggle/common/rootfs.py (same author, unchanged logic): official SWE-bench image -> rootfs, run under chroot/proot.
# ---- rootfs.py: run official SWE-bench instance images WITHOUT Docker (Kaggle has no Docker daemon) ----
# Pulls the image from Docker Hub via the registry v2 API, applies OCI layers (with whiteouts) into a directory,
# and executes commands inside it with chroot (preferred) or proot (fallback). Digest of every blob is verified.
import hashlib as _hl, json as _json, os as _os, shutil as _sh, subprocess as _sp, urllib.request as _ur
from pathlib import Path as _P

_ACC = ("application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, "
        "application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json")
PROOT_URL = "https://github.com/proot-me/proot/releases/download/v5.5.0/proot"
PROOT_SHA = "db185300a9daabbce86aca7632d8c7fdc4249f6f153171954013b70c5f491ec0"


def _get(url, headers, timeout=120):
    return _ur.urlopen(_ur.Request(url, headers=headers), timeout=timeout)


def pull_image(image: str, dest, cache_dir="/kaggle/tmp/blobs"):
    """image like 'swebench/sweb.eval.x86_64.psf_1776_requests-1766:latest'. Returns dict(digest, layers, bytes)."""
    repo, tag = image.rsplit(":", 1)
    tok = _json.load(_get(f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull", {}))["token"]
    h = {"Authorization": "Bearer " + tok, "Accept": _ACC}
    idx = _json.load(_get(f"https://registry-1.docker.io/v2/{repo}/manifests/{tag}", h))
    if "manifests" in idx:
        mdig = [m for m in idx["manifests"] if m.get("platform", {}).get("architecture") == "amd64"][0]["digest"]
        man = _json.load(_get(f"https://registry-1.docker.io/v2/{repo}/manifests/{mdig}", h))
    else:
        mdig, man = None, idx
    dest = _P(dest); dest.mkdir(parents=True, exist_ok=True)
    cache = _P(cache_dir); cache.mkdir(parents=True, exist_ok=True)
    total = 0
    for layer in man["layers"]:
        d = layer["digest"]; fp = cache / d.replace(":", "_")
        if not fp.exists() or fp.stat().st_size != layer["size"]:
            hh = _hl.sha256()
            with _get(f"https://registry-1.docker.io/v2/{repo}/blobs/{d}", h, timeout=600) as r, open(str(fp) + ".part", "wb") as f:
                for b in iter(lambda: r.read(1 << 22), b""):
                    hh.update(b); f.write(b)
            if "sha256:" + hh.hexdigest() != d:
                raise RuntimeError(f"blob digest mismatch {d}")
            _os.replace(str(fp) + ".part", fp)
        total += layer["size"]
        _apply_layer(fp, dest)
    return {"image": image, "manifest_digest": mdig, "layers": len(man["layers"]), "compressed_bytes": total}


def _apply_layer(tgz, dest):
    # list whiteouts first, apply deletions, then extract (whiteout markers removed afterwards)
    names = _sp.run(["tar", "-tzf", str(tgz)], capture_output=True, text=True, check=True).stdout.splitlines()
    for n in names:
        base = _os.path.basename(n.rstrip("/")); parent = _os.path.dirname(n.rstrip("/"))
        if base == ".wh..wh..opq":
            pd = dest / parent
            if pd.is_dir():
                for c in pd.iterdir():
                    _rm(c)
        elif base.startswith(".wh."):
            _rm(dest / parent / base[4:])
    _sp.run(["tar", "--numeric-owner", "-xzpf", str(tgz), "-C", str(dest)], check=True, capture_output=True)
    for n in names:
        base = _os.path.basename(n.rstrip("/"))
        if base.startswith(".wh."):
            _rm(dest / n.rstrip("/"))


def _rm(p):
    p = _P(p)
    if p.is_symlink() or p.is_file():
        p.unlink(missing_ok=True)
    elif p.is_dir():
        _sh.rmtree(p, ignore_errors=True)


class Sandbox:
    """Runs bash inside the rootfs. mode: 'chroot' (needs CAP_SYS_CHROOT) or 'proot' (ptrace)."""

    def __init__(self, root):
        self.root = _P(root); self.mode = None; self.notes = []
        for d in ("proc", "dev", "sys", "tmp"):
            (self.root / d).mkdir(exist_ok=True)
        _os.chmod(self.root / "tmp", 0o1777)
        (self.root / "etc").mkdir(exist_ok=True)
        try:
            _sh.copyfile("/etc/resolv.conf", self.root / "etc/resolv.conf")
        except Exception:
            pass
        hosts = self.root / "etc/hosts"
        if not hosts.exists() or "localhost" not in hosts.read_text(errors="ignore"):
            if hosts.is_symlink():
                hosts.unlink()
            hosts.write_text("127.0.0.1 localhost\n::1 localhost ip6-localhost ip6-loopback\n")
        mounted = _sp.run(["mount", "--bind", "/proc", str(self.root / "proc")], capture_output=True, text=True)
        if mounted.returncode == 0:
            _sp.run(["mount", "--bind", "/dev", str(self.root / "dev")], capture_output=True)
            _sp.run(["mount", "--bind", "/sys", str(self.root / "sys")], capture_output=True)
            self.notes.append("bind mounts ok")
        else:
            self.notes.append("bind mount failed: " + mounted.stderr.strip()[:200])
            for name, maj, mi in (("null", 1, 3), ("zero", 1, 5), ("random", 1, 8), ("urandom", 1, 9), ("tty", 5, 0)):
                p = self.root / "dev" / name
                if not p.exists():
                    _sp.run(["mknod", "-m", "666", str(p), "c", str(maj), str(mi)], capture_output=True)
        self.bind_ok = mounted.returncode == 0
        r = _sp.run(["chroot", str(self.root), "/bin/bash", "-c", "echo ok"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip() == "ok" and self.bind_ok:
            self.mode = "chroot"
        else:
            self.notes.append("chroot: rc=%s %s" % (r.returncode, (r.stderr or "")[:200]))
            pr = _P("/kaggle/tmp/proot")
            if not pr.exists():
                with _get(PROOT_URL, {"User-Agent": "k4"}) as resp, open(pr, "wb") as f:
                    _sh.copyfileobj(resp, f)
                if _hl.sha256(pr.read_bytes()).hexdigest() != PROOT_SHA:
                    raise RuntimeError("proot sha mismatch")
                pr.chmod(0o755)
            r2 = _sp.run([str(pr), "-0", "-r", str(self.root), "-b", "/proc", "-b", "/dev", "-b", "/sys", "/bin/bash", "-c", "echo ok"],
                         capture_output=True, text=True, env=dict(_os.environ, PROOT_NO_SECCOMP="1"))
            if r2.returncode == 0 and "ok" in r2.stdout:
                self.mode = "proot"; self.proot = str(pr)
            elif r.returncode == 0:
                self.mode = "chroot"; self.notes.append("using chroot without /proc")
            else:
                raise RuntimeError("no sandbox: " + "; ".join(self.notes) + " proot: " + r2.stderr[:300])

    def run(self, script: str, timeout=600):
        """Run a bash script inside the image. Returns (rc, combined_output)."""
        full = "export HOME=/root PATH=/opt/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin LANG=C.UTF-8\n" + script
        if self.mode == "chroot":
            cmd = ["chroot", str(self.root), "/bin/bash", "-c", full]; env = None
        else:
            cmd = [self.proot, "-0", "-r", str(self.root), "-b", "/proc", "-b", "/dev", "-b", "/sys", "-w", "/", "/bin/bash", "-c", full]
            env = dict(_os.environ, PROOT_NO_SECCOMP="1")
        try:
            r = _sp.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, errors="replace")
            return r.returncode, r.stdout + r.stderr
        except _sp.TimeoutExpired as e:
            out = (e.stdout or b"") if isinstance(e.stdout, bytes) else (e.stdout or "").encode()
            return 124, out.decode("utf-8", "replace") + "\n[TIMEOUT]"

    def close(self):
        for d in ("sys", "dev", "proc"):
            _sp.run(["umount", "-l", str(self.root / d)], capture_output=True)
# ---- end rootfs.py ----
