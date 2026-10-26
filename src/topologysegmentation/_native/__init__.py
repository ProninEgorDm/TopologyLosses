"""Optional C implementation of binary cubical Betti numbers.

The source ships inside the wheel. The first call compiles it with the system
C compiler and caches the library under ``~/.cache/topologysegmentation``.
``TOPOLOGYSEGMENTATION_NATIVE=0`` keeps the NumPy fallback. Install itself
does not need a compiler.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

_LOCK = threading.Lock()
_LIBRARY = None
_UNAVAILABLE = False


def binary_betti(mask: np.ndarray) -> tuple[int, ...] | None:
    """Return Betti numbers, or ``None`` when the compiled routine is unavailable."""
    library = _library()
    if library is None:
        return None
    out = np.zeros(mask.ndim, dtype=np.int32)
    shape = np.ascontiguousarray(mask.shape, dtype=np.int64)
    code = library.cubical_betti(
        mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        int(mask.ndim),
        shape.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    if code != 0:
        return None
    return tuple(int(value) for value in out)


def _library():
    global _LIBRARY, _UNAVAILABLE
    if _UNAVAILABLE:
        return None
    if _LIBRARY is not None:
        return _LIBRARY
    with _LOCK:
        if _LIBRARY is not None or _UNAVAILABLE:
            return _LIBRARY
        if os.environ.get("TOPOLOGYSEGMENTATION_NATIVE") == "0":
            _UNAVAILABLE = True
            return None
        try:
            _LIBRARY = _compile()
        except (OSError, subprocess.SubprocessError):
            _LIBRARY = None
            _UNAVAILABLE = True
        return _LIBRARY


def _compile():
    source = Path(__file__).with_name("cubical_betti.c")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    library_path = cache / f"cubical_betti_{digest}{suffix}"
    if not library_path.exists():
        compiler = _compiler()
        link = "-dynamiclib" if sys.platform == "darwin" else "-shared"
        command = [compiler, "-O3", "-std=c11", "-fPIC", link, "-o", str(library_path), str(source)]
        subprocess.run(command, check=True, capture_output=True)
    library = ctypes.CDLL(str(library_path))
    library.cubical_betti.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int32),
    ]
    library.cubical_betti.restype = ctypes.c_int
    return library


def _compiler() -> str:
    for name in ("clang", "cc", "gcc"):
        if _which(name):
            return name
    raise OSError("No C compiler for the Betti routine")


def _which(name: str) -> bool:
    path = os.environ.get("PATH", "")
    for directory in path.split(os.pathsep):
        if (Path(directory) / name).is_file():
            return True
    return False


def _cache_dir() -> Path:
    override = os.environ.get("TOPOLOGYSEGMENTATION_CACHE")
    if override:
        return Path(override)
    home_cache = Path.home() / ".cache" / "topologysegmentation"
    try:
        home_cache.mkdir(parents=True, exist_ok=True)
        return home_cache
    except OSError:
        return Path(tempfile.gettempdir()) / "topologysegmentation"
