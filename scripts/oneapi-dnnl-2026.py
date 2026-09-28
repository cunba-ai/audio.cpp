#!/usr/bin/env python3
"""Fetch the oneDNN payload for oneAPI 2026 without downloading the full toolkit.

Essentials 2026 offline installers no longer bundle intel.oneapi.win.dnnl, but
the ggml SYCL target still links DNNL::dnnl (external/ggml/cmake/
ggml-config.cmake.in). The unified oneAPI toolkit offline .exe (~1.8 GB) does
carry it as packages/intel.oneapi.win.dnnl,v=*/cupPayload.cup; the .exe is a
plain zip appended to a bootstrapper stub, so we range-read its central
directory (4 MB tail), locate the dnnl cupPayload entry, range-fetch just that
entry (~28 MB) and unpack its _installdir/ tree into the oneAPI prefix.

Usage: python oneapi-dnnl-2026.py <oneAPI prefix>   (e.g. "C:/Program Files (x86)/Intel/oneAPI")
"""
import os
import struct
import sys
import urllib.parse
import urllib.request
import zipfile

TOOLKIT_URL = (
    "https://registrationcenter-download.intel.com/akdlm/IRC_NAS/"
    "0cb67a0d-67f6-410b-868b-f4a0a17ff0cf/intel-oneapi-toolkit-2026.1.1.32_offline.exe"
)
TAIL = 4 * 1024 * 1024  # zip EOCD + central directory live in the last few MB


def fetch(start: int, end: int) -> bytes:
    req = urllib.request.Request(TOOLKIT_URL, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return resp.read()


def total_size() -> int:
    req = urllib.request.Request(TOOLKIT_URL, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        content_range = resp.headers["Content-Range"]  # bytes 0-0/NNNN
        return int(content_range.split("/")[-1])


def parse_central_directory(size: int):
    """Return [(name, csize, usize, method, local_header_offset)] for the zip."""
    tail = fetch(size - TAIL, size - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    assert eocd >= 0, "EOCD not found in tail"
    (_, _, _, _, n_total, cd_size, cd_off, _) = struct.unpack("<IHHHHIIH", tail[eocd:eocd + 22])
    if n_total == 0xFFFF or cd_size == 0xFFFFFFFF or cd_off == 0xFFFFFFFF:
        loc = tail.rfind(b"PK\x06\x07", 0, eocd)
        assert loc >= 0, "zip64 locator not found"
        (_, _, _, eocd64_off, _) = struct.unpack("<IIQQ", tail[loc:loc + 20])
        if eocd64_off >= size - TAIL:
            e64 = tail[eocd64_off - (size - TAIL):]
        else:
            e64 = fetch(eocd64_off, eocd64_off + 55)
        (_, _, _, _, n_total, cd_size, cd_off) = struct.unpack("<QHHQQ", e64[:48])
    if cd_off >= size - TAIL:
        cd = tail[cd_off - (size - TAIL): cd_off - (size - TAIL) + cd_size]
    else:
        cd = fetch(cd_off, cd_off + cd_size - 1)
        assert len(cd) == cd_size, "central directory truncated"
    entries = []
    i = 0
    while i + 46 <= len(cd) and cd[i:i + 4] == b"PK\x01\x02":
        (_ver_made, _ver_need, _flags, method, _mtime, _mdate, _crc, csize, usize,
         nlen, elen, clen, _disk, _iattr, _eattr, lho) = struct.unpack("<HHHHHHIIIHHHHHII", cd[i + 4:i + 46])
        name = cd[i + 46:i + 46 + nlen].decode("utf-8", "replace")
        extra = cd[i + 46 + nlen:i + 46 + nlen + elen]
        if csize == 0xFFFFFFFF or usize == 0xFFFFFFFF or lho == 0xFFFFFFFF:
            j = 0
            while j + 4 <= len(extra):
                hid, hsz = struct.unpack("<HH", extra[j:j + 4])
                data = extra[j + 4:j + 4 + hsz]
                if hid == 1:  # zip64 extended information
                    vals = [usize, csize, lho]
                    off = 0
                    for k in range(3):
                        if vals[k] == 0xFFFFFFFF and off + 8 <= len(data):
                            vals[k] = struct.unpack("<Q", data[off:off + 8])[0]
                            off += 8
                    usize, csize, lho = vals
                j += 4 + hsz
        entries.append((name, csize, usize, method, lho))
        i += 46 + nlen + elen + clen
    return entries


def fetch_cup(entry) -> bytes:
    name, csize, _usize, _method, lho = entry
    blob = fetch(lho, lho + 4096 + csize + 64)
    assert blob[:4] == b"PK\x03\x04", "no local file header at entry offset"
    nlen, elen = struct.unpack("<HH", blob[26:30])
    entry_name = blob[30:30 + nlen].decode("utf-8", "replace")
    assert entry_name == name, f"entry name mismatch: {entry_name} != {name}"
    import zlib
    return zlib.decompress(blob[30 + nlen + elen:30 + nlen + elen + csize], -15)


def unpack_cup(cup: bytes, dest: str) -> int:
    import io
    import shutil
    n = 0
    with zipfile.ZipFile(io.BytesIO(cup)) as zf:
        for info in zf.infolist():
            name = urllib.parse.unquote(info.filename)
            if name.endswith("/") or name.endswith("\\"):
                continue
            marker = "_installdir/"
            idx = name.find(marker)
            if idx < 0:
                continue
            rel = name[idx + len(marker):]
            out = os.path.join(dest, rel.replace("/", os.sep))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with zf.open(info) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    return n


def main() -> int:
    prefix = sys.argv[1] if len(sys.argv) > 1 else r"C:\Program Files (x86)\Intel\oneAPI"
    print(f"toolkit: {TOOLKIT_URL}")
    size = total_size()
    print(f"toolkit size: {size:,} bytes (range-reads only, no full download)")
    entries = parse_central_directory(size)
    print(f"central directory entries: {len(entries)}")
    hits = [e for e in entries if "dnnl" in e[0].lower() and e[0].endswith("cupPayload.cup")]
    assert hits, "no dnnl cupPayload entry found in toolkit"
    entry = hits[0]
    print(f"fetching {entry[0]} ({entry[1]:,} bytes compressed)")
    cup = fetch_cup(entry)
    n = unpack_cup(cup, prefix)
    print(f"unpacked {n} files into {prefix}")
    # The cup lands as <prefix>/dnnl/<version>/...
    found = []
    for root, _dirs, files in os.walk(os.path.join(prefix, "dnnl")):
        for f in files:
            if f.lower() == "dnnl.dll":
                found.append(os.path.join(root, f))
    assert found, "dnnl.dll not found after unpack"
    for f in found:
        print(f"  ok: {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
