#!/usr/bin/env python3
"""Collect the runtime DLL closure of a built Windows library into a flat directory.

Walks the PE import table recursively (pure stdlib, no objdump needed),
resolves each DLL against the given roots (oneAPI prefix, Level Zero SDK,
build output), and copies the closure next to the target so the zip is
self-contained on machines without a toolchain installed.

Usage:
  python collect-sycl-runtime.py --binary artifacts/audiocpp.dll --out artifacts \
      --root "C:/Program Files (x86)/Intel/oneAPI" --root C:/level-zero-sdk

OS-provided DLLs (KnownDLLs, api-ms-*/ext-ms-* sets) are skipped. VC runtime
(msvcp140/vcruntime140*/concrt140) is collected when resolvable and warned
about otherwise - deployments usually ship the VC redist separately.
"""
import argparse
import os
import shutil
import struct
import sys

OS_DLLS = {
    "kernel32.dll", "user32.dll", "gdi32.dll", "comdlg32.dll", "advapi32.dll",
    "shell32.dll", "ole32.dll", "oleaut32.dll", "ws2_32.dll", "shlwapi.dll",
    "psapi.dll", "bcrypt.dll", "ntdll.dll", "secur32.dll", "crypt32.dll",
    "userenv.dll", "dbghelp.dll", "iphlpapi.dll", "imagehlp.dll",
    "wintrust.dll", "setupapi.dll", "winmm.dll", "imm32.dll", "dxgi.dll",
    "d3d11.dll", "opencl.dll", "ze_loader.dll",  # opencl/ze_loader: GPU-driver provided
    "msvcrt.dll", "msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll",
    "concrt140.dll",  # VC redist: collected if found in roots, never fatal
}
OS_PREFIXES = ("api-ms-", "ext-ms-")
WARN_ONLY = {"msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll", "concrt140.dll",
             "msvcp140_codecvt_ids.dll", "ucrtbased.dll", "vcruntime140d.dll"}  # debug CRT: debug dlls must not ship
# The SYCL runtime loads these via LoadLibrary at runtime - invisible to the
# static import walk. Pulled in from the directory that provided sycl*.dll.
# Debug flavors (umfd.dll, tbb12d.dll, ...) are excluded: they pull the debug
# CRT and must never ship in a release zip.
DYNAMIC_SIDECAR_PATTERNS = ("ur_adapter_*.dll", "umf*.dll", "libhwloc*.dll")
DEBUG_SIDECARS = {"umfd.dll", "tbb12d.dll", "ur_win_proxy_loaderd.dll"}


def rva_to_offset(data: bytes, pe_off: int, rva: int) -> int:
    num_sec = struct.unpack_from("<H", data, pe_off + 6)[0]
    sz_opt = struct.unpack_from("<H", data, pe_off + 20)[0]
    sec_off = pe_off + 24 + sz_opt
    for i in range(num_sec):
        b = sec_off + i * 40
        va = struct.unpack_from("<I", data, b + 12)[0]
        vs = struct.unpack_from("<I", data, b + 8)[0]
        raw = struct.unpack_from("<I", data, b + 20)[0]
        rrs = struct.unpack_from("<I", data, b + 16)[0]
        if va <= rva < va + max(vs, rrs):
            return raw + (rva - va)
    return -1


def pe_imports(path: str):
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:2] != b"MZ":
        return set()
    pe_off = struct.unpack_from("<i", data, 0x3C)[0]
    if data[pe_off:pe_off + 4] != b"PE\0\0":
        return set()
    opt_hdr = pe_off + 24
    magic = struct.unpack_from("<H", data, opt_hdr)[0]
    dd_off = opt_hdr + (96 if magic == 0x10B else 112)
    imp_rva, imp_size = struct.unpack_from("<II", data, dd_off + 8)
    if imp_rva == 0 or imp_size == 0:
        return set()
    imports = set()
    desc = rva_to_offset(data, pe_off, imp_rva)
    idx = 0
    while True:
        d = desc + idx * 20
        name_rva = struct.unpack_from("<I", data, d + 12)[0]
        if name_rva == 0:
            break
        name_off = rva_to_offset(data, pe_off, name_rva)
        end = data.find(b"\0", name_off)
        imports.add(data[name_off:end].decode("ascii", "replace").lower())
        idx += 1
    return imports


def index_roots(roots):
    index = {}
    for root in roots:
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if f.lower().endswith(".dll"):
                    index.setdefault(f.lower(), os.path.join(dirpath, f))
    return index


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", action="append", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    index = index_roots([os.path.dirname(os.path.abspath(args.binary))] + list(args.root))
    queue = [os.path.basename(args.binary)]
    copied = {}
    missing = {}
    seen = set()
    while queue:
        name = queue.pop().lower()
        if name in seen:
            continue
        seen.add(name)
        if name.startswith(OS_PREFIXES) or name in OS_DLLS:
            continue
        path = index.get(name)
        if path is None:
            missing.setdefault(name, set())
            continue
        dst = os.path.join(args.out, name)
        if not os.path.exists(dst):
            shutil.copy2(path, dst)
        copied[name] = path
        for dep in pe_imports(path):
            if dep not in seen:
                queue.append(dep)

    for name in sorted(copied):
        print(f"  + {name}  <-  {copied[name]}")
    # Runtime-loaded sidecars (UR adapters, UMF, hwloc) from dirs that host sycl*.dll
    import glob as _glob
    extra_dirs = {os.path.dirname(p) for n, p in copied.items() if n.startswith("sycl")}
    for d in sorted(extra_dirs):
        for pat in DYNAMIC_SIDECAR_PATTERNS:
            for side in _glob.glob(os.path.join(d, pat)):
                name = os.path.basename(side).lower()
                if name in seen or name in OS_DLLS or name.startswith(OS_PREFIXES) or name in DEBUG_SIDECARS:
                    continue
                seen.add(name)
                shutil.copy2(side, os.path.join(args.out, name))
                copied[name] = side
                for dep in pe_imports(side):
                    queue.append(dep)
    # drain any imports the sidecars pulled in
    while queue:
        name = queue.pop().lower()
        if name in seen:
            continue
        seen.add(name)
        if name.startswith(OS_PREFIXES) or name in OS_DLLS:
            continue
        path = index.get(name)
        if path is None:
            missing.setdefault(name, set())
            continue
        if not os.path.exists(os.path.join(args.out, name)):
            shutil.copy2(path, os.path.join(args.out, name))
        copied[name] = path
        for dep in pe_imports(path):
            if dep not in seen:
                queue.append(dep)
    print("  sidecars merged:")
    for name in sorted(copied):
        if "adapter" in name or "umf" in name or "hwloc" in name:
            print(f"  + {name}  <-  {copied[name]}")
    fatal = {m for m in missing if m not in WARN_ONLY}
    if missing:
        print("unresolved imports:")
        for m in sorted(missing):
            tag = "WARN" if m in WARN_ONLY else "MISSING"
            print(f"  [{tag}] {m}")
    if fatal:
        print(f"closure incomplete: {len(fatal)} non-OS DLLs unresolved")
        return 1
    print(f"closure OK: {len(copied)} dlls collected into {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
