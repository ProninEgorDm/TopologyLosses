"""Compiled vertex grid and union-find for the Betti matching loss.

The routines return cell indices. Filtration values stay in the caller's tensor,
so the loss gradient is unchanged. ``TOPOLOGYSEGMENTATION_NATIVE=0`` or
``TOPOLOGYSEGMENTATION_MATCHING=0`` keeps the Python loops.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

from topologysegmentation._native import _cache_dir, _compiler

_LOCK = threading.Lock()
_LIBRARY = None
_UNAVAILABLE = False


def enabled() -> bool:
    if os.environ.get("TOPOLOGYSEGMENTATION_NATIVE") == "0":
        return False
    if os.environ.get("TOPOLOGYSEGMENTATION_MATCHING") == "0":
        return False
    return _library() is not None


def fill_vertex_grid(order: np.ndarray, keys: np.ndarray, height: int, width: int, rows: int, cols: int, num_edges: int):
    library = _library()
    if library is None:
        return None
    index_map = np.full((rows, cols), -1, dtype=np.int32)
    coordinates = np.zeros((rows * cols, 2), dtype=np.int32)
    edges = np.zeros(num_edges, dtype=np.int32)
    capacity = rows * cols
    fine_rows = np.empty(capacity, dtype=np.int32)
    fine_cols = np.empty(capacity, dtype=np.int32)
    pixel_rows = np.empty(capacity, dtype=np.int32)
    pixel_cols = np.empty(capacity, dtype=np.int32)
    written = ctypes.c_int32(0)
    code = library.fill_vertex_grid(
        np.ascontiguousarray(order, dtype=np.int64).ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        np.ascontiguousarray(keys, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        int(height),
        int(width),
        index_map.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        int(rows),
        int(cols),
        coordinates.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        edges.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        int(num_edges),
        fine_rows.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        fine_cols.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        pixel_rows.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        pixel_cols.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.byref(written),
    )
    if code != 0:
        return None
    count = int(written.value)
    return {
        "index_map": index_map,
        "coordinates": coordinates,
        "edges": edges,
        "fine_rows": fine_rows[:count],
        "fine_cols": fine_cols[:count],
        "pixel_rows": pixel_rows[:count],
        "pixel_cols": pixel_cols[:count],
    }


def cubical_persistence(index_map, coordinates, edges, values, superlevel: bool, mode: int):
    library = _library()
    if library is None:
        return None
    num_cubes = int(coordinates.shape[0])
    num_edges = int(edges.shape[0])
    h0 = np.empty(num_edges * 2, dtype=np.int32)
    h1 = np.empty(num_edges * 2, dtype=np.int32)
    columns = np.empty(num_edges, dtype=np.int32)
    critical = np.empty(num_edges, dtype=np.int32)
    columns2 = np.empty(num_edges, dtype=np.int32)
    n0 = ctypes.c_int32()
    n1 = ctypes.c_int32()
    n_columns = ctypes.c_int32()
    n_critical = ctypes.c_int32()
    n_columns2 = ctypes.c_int32()
    code = library.cubical_persistence(
        np.ascontiguousarray(index_map, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        int(index_map.shape[0]),
        int(index_map.shape[1]),
        np.ascontiguousarray(coordinates, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        np.ascontiguousarray(edges, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        num_edges,
        np.ascontiguousarray(values, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        num_cubes,
        int(superlevel),
        int(mode),
        h0.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.byref(n0),
        h1.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.byref(n1),
        columns.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.byref(n_columns),
        critical.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.byref(n_critical),
        columns2.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.byref(n_columns2),
    )
    if code != 0:
        return None
    return {
        "h0": h0[: int(n0.value) * 2].reshape(-1, 2),
        "h1": h1[: int(n1.value) * 2].reshape(-1, 2),
        "columns": columns[: int(n_columns.value)],
        "critical": critical[: int(n_critical.value)],
        "columns2": columns2[: int(n_columns2.value)],
    }


def image_persistence(index0, coord0, index1, coord1, columns1, critical0):
    library = _library()
    if library is None:
        return None
    capacity = max(int(columns1.shape[0]), int(critical0.shape[0]), 1)
    h0 = np.empty(capacity * 2, dtype=np.int32)
    h1 = np.empty(capacity * 2, dtype=np.int32)
    n0 = ctypes.c_int32()
    n1 = ctypes.c_int32()
    code = library.image_persistence(
        np.ascontiguousarray(index0, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        np.ascontiguousarray(coord0, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        int(coord0.shape[0]),
        np.ascontiguousarray(index1, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        np.ascontiguousarray(coord1, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        int(coord1.shape[0]),
        int(index0.shape[0]),
        int(index0.shape[1]),
        np.ascontiguousarray(columns1, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        int(columns1.shape[0]),
        np.ascontiguousarray(critical0, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        int(critical0.shape[0]),
        h0.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.byref(n0),
        h1.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.byref(n1),
    )
    if code != 0:
        return None
    return {
        "h0": h0[: int(n0.value) * 2].reshape(-1, 2),
        "h1": h1[: int(n1.value) * 2].reshape(-1, 2),
    }


def _library():
    global _LIBRARY, _UNAVAILABLE
    if _UNAVAILABLE:
        return None
    if _LIBRARY is not None:
        return _LIBRARY
    with _LOCK:
        if _LIBRARY is not None or _UNAVAILABLE:
            return _LIBRARY
        if not _allowed():
            _UNAVAILABLE = True
            return None
        try:
            _LIBRARY = _compile()
        except (OSError, subprocess.SubprocessError):
            _LIBRARY = None
            _UNAVAILABLE = True
        return _LIBRARY


def _allowed() -> bool:
    if os.environ.get("TOPOLOGYSEGMENTATION_NATIVE") == "0":
        return False
    return os.environ.get("TOPOLOGYSEGMENTATION_MATCHING") != "0"


def _compile():
    source = Path(__file__).with_name("cubical_matching.c")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    library_path = cache / f"cubical_matching_{digest}{suffix}"
    if not library_path.exists():
        link = "-dynamiclib" if sys.platform == "darwin" else "-shared"
        command = [_compiler(), "-O3", "-std=c11", "-fPIC", link, "-o", str(library_path), str(source)]
        subprocess.run(command, check=True, capture_output=True)
    library = ctypes.CDLL(str(library_path))
    library.fill_vertex_grid.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
    ]
    library.fill_vertex_grid.restype = ctypes.c_int
    library.cubical_persistence.argtypes = [ctypes.POINTER(ctypes.c_int32)] * 0
    _bind_persistence(library)
    _bind_image(library)
    return library


def _bind_persistence(library):
    library.cubical_persistence.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
    ]
    library.cubical_persistence.restype = ctypes.c_int


def _bind_image(library):
    library.image_persistence.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
    ]
    library.image_persistence.restype = ctypes.c_int
