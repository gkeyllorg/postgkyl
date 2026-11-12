"""Reader for the Gkeyll .gkyl files the viewer opens: 2D float64 fields (whole or ranged) and block topology files,
with their MessagePack metadata. Dependency-free (numpy only), so the viewer needs no postgkyl install; check_gkview.py
compares what it reads with postgkyl."""
from pathlib import Path
import struct

import numpy as np


def unpack_meta(buf):
    """Decode the scalar/map/array MessagePack subset emitted by Gkeyll."""
    def take(off):
        b = buf[off]
        off += 1
        if b <= 0x7f:
            return b, off
        if b >= 0xe0:
            return b - 256, off
        if b in (0xc0, 0xc2, 0xc3):
            return {0xc0: None, 0xc2: False, 0xc3: True}[b], off
        formats = {0xcc: "B", 0xcd: "H", 0xce: "I", 0xcf: "Q",
                   0xd0: "b", 0xd1: "h", 0xd2: "i", 0xd3: "q",
                   0xca: "f", 0xcb: "d"}
        if b in formats:
            fmt = ">" + formats[b]
            return struct.unpack_from(fmt, buf, off)[0], off + struct.calcsize(fmt)
        if 0xa0 <= b <= 0xbf:
            n, kind = b & 31, "str"
        elif 0x90 <= b <= 0x9f:
            n, kind = b & 15, "array"
        elif 0x80 <= b <= 0x8f:
            n, kind = b & 15, "map"
        else:
            spec = {0xd9: ("B", "str"), 0xda: ("H", "str"), 0xdb: ("I", "str"),
                    0xdc: ("H", "array"), 0xdd: ("I", "array"),
                    0xde: ("H", "map"), 0xdf: ("I", "map")}
            if b not in spec:
                raise ValueError("unsupported MessagePack tag 0x%x" % b)
            fmt, kind = spec[b]
            n = struct.unpack_from(">" + fmt, buf, off)[0]
            off += struct.calcsize(fmt)
        if kind == "str":
            if off + n > len(buf):
                raise ValueError("truncated metadata string")
            return buf[off:off+n].decode(), off+n
        out = [] if kind == "array" else {}
        for _ in range(n):
            v, off = take(off)
            if kind == "array":
                out.append(v)
            else:
                value, off = take(off)
                out[v] = value
        return out, off
    value, end = take(0)
    if end != len(buf) or not isinstance(value, dict):
        raise ValueError("invalid metadata map")
    return value


def read_gkyl(path):
    """Read v1 fields/topology; reject partial, overlapping or truncated ranges."""
    buf = Path(path).read_bytes()
    if buf[:5] != b"gkyl0":
        raise ValueError("not a Gkeyll file: %s" % path)
    version, kind, msize = struct.unpack_from("<QQQ", buf, 5)
    if version != 1 or 29 + msize > len(buf):
        raise ValueError("unsupported version or truncated header: %s" % path)
    meta = unpack_meta(buf[29:29+msize]) if msize else {}
    off = 29 + msize
    if kind == 4:
        if off != len(buf):
            raise ValueError("unexpected topology payload")
        return {"meta": meta, "kind": kind}
    if kind not in (1, 3):
        raise ValueError("unsupported field file type %s" % kind)
    real, ndim = struct.unpack_from("<QQ", buf, off)
    off += 16
    if ndim != 2 or real != 2:
        raise ValueError("only 2D float64 fields are supported")
    cells = struct.unpack_from("<2Q", buf, off)
    lower = np.array(struct.unpack_from("<2d", buf, off+16))
    upper = np.array(struct.unpack_from("<2d", buf, off+32))
    esznc, size = struct.unpack_from("<QQ", buf, off+48)
    off += 64
    if min(cells) < 1 or esznc < 8 or esznc % 8 or size != np.prod(cells):
        raise ValueError("invalid field dimensions")
    ncomp = esznc // 8
    if kind == 1:
        data = np.frombuffer(buf, dtype="<f8", count=size*ncomp, offset=off)
        off += size * esznc
        data = data.reshape(cells + (ncomp,))
    else:
        nranges = struct.unpack_from("<Q", buf, off)[0]
        off += 8
        data = np.empty(cells + (ncomp,))
        covered = np.zeros(cells, dtype=bool)
        for _ in range(nranges):
            lo = struct.unpack_from("<2Q", buf, off)
            up = struct.unpack_from("<2Q", buf, off+16)
            rsize = struct.unpack_from("<Q", buf, off+32)[0]
            off += 40
            shape = tuple(u-l+1 for l, u in zip(lo, up))
            if any(l < 1 or u < l or u > c for l, u, c in zip(lo, up, cells)) or rsize != np.prod(shape):
                raise ValueError("invalid field range")
            sl = tuple(slice(l-1, u) for l, u in zip(lo, up))
            if covered[sl].any():
                raise ValueError("overlapping field ranges")
            data[sl] = np.frombuffer(buf, dtype="<f8", count=rsize*ncomp, offset=off).reshape(shape+(ncomp,))
            covered[sl] = True
            off += rsize * esznc
        if not covered.all():
            raise ValueError("incomplete field ranges")
    if off != len(buf):
        raise ValueError("truncated or trailing field payload")
    return {"data": data, "cells": cells, "lower": lower, "upper": upper,
            "ncomp": ncomp, "meta": meta, "kind": kind}
