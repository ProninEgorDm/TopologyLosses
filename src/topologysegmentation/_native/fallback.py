"""NumPy cubical Betti numbers used when the C routine is not compiled."""

from __future__ import annotations

import numpy as np


def binary_betti(mask: np.ndarray) -> tuple[int, ...]:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim == 2:
        return _betti2(mask)
    if mask.ndim == 3:
        return _betti3(mask)
    raise ValueError(f"Expected a 2D or 3D mask, got shape {tuple(mask.shape)}")


class _UnionFind:
    def __init__(self, size: int):
        self.parent = np.full(size, -1, dtype=np.int32)

    def add(self, index: int) -> None:
        if self.parent[index] < 0:
            self.parent[index] = index

    def find(self, index: int) -> int:
        parent = self.parent
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(self, left: int, right: int) -> None:
        self.add(left)
        self.add(right)
        left = self.find(left)
        right = self.find(right)
        if left != right:
            self.parent[right] = left

    def components(self) -> int:
        total = 0
        for index, value in enumerate(self.parent):
            if value >= 0 and self.find(index) == index:
                total += 1
        return total


def _betti2(mask: np.ndarray) -> tuple[int, int]:
    height, width = mask.shape
    nodes = _UnionFind(mask.size)
    for row in range(height):
        for col in range(width):
            if not mask[row, col]:
                continue
            index = row * width + col
            nodes.add(index)
            if row:
                if col and mask[row - 1, col - 1]:
                    nodes.union(index, index - width - 1)
                if mask[row - 1, col]:
                    nodes.union(index, index - width)
                if col + 1 < width and mask[row - 1, col + 1]:
                    nodes.union(index, index - width + 1)
            if col and mask[row, col - 1]:
                nodes.union(index, index - 1)
    components = nodes.components()
    chi = _chi2(mask)
    return components, components - chi


def _chi2(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    height, width = mask.shape
    horizontal = np.zeros((height + 1, width), dtype=bool)
    horizontal[1:] |= mask
    horizontal[:-1] |= mask
    vertical = np.zeros((height, width + 1), dtype=bool)
    vertical[:, 1:] |= mask
    vertical[:, :-1] |= mask
    vertices = np.zeros((height + 1, width + 1), dtype=bool)
    vertices[:-1, :-1] |= mask
    vertices[:-1, 1:] |= mask
    vertices[1:, :-1] |= mask
    vertices[1:, 1:] |= mask
    return int(vertices.sum()) - int(horizontal.sum() + vertical.sum()) + int(mask.sum())


def _betti3(mask: np.ndarray) -> tuple[int, int, int]:
    depth, height, width = mask.shape
    nodes = _UnionFind(mask.size)
    neighbors = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dz < 0 or (dz == 0 and (dy < 0 or (dy == 0 and dx < 0)))]
    for z, y, x in np.argwhere(mask):
        index = int((z * height + y) * width + x)
        nodes.add(index)
        for dz, dy, dx in neighbors:
            zz, yy, xx = int(z + dz), int(y + dy), int(x + dx)
            if zz < 0 or yy < 0 or xx < 0 or yy >= height or xx >= width:
                continue
            if mask[zz, yy, xx]:
                nodes.union(index, int((zz * height + yy) * width + xx))
    components = nodes.components()
    voids = _voids(mask)
    return components, components + voids - _chi3(mask), voids


def _voids(mask: np.ndarray) -> int:
    background = ~mask
    depth, height, width = mask.shape
    nodes = _UnionFind(mask.size)
    for z, y, x in np.argwhere(background):
        index = int((z * height + y) * width + x)
        nodes.add(index)
        if z and background[z - 1, y, x]:
            nodes.union(index, index - height * width)
        if y and background[z, y - 1, x]:
            nodes.union(index, index - width)
        if x and background[z, y, x - 1]:
            nodes.union(index, index - 1)
    touched = np.zeros(mask.size, dtype=bool)
    for z, y, x in np.argwhere(background):
        if z == 0 or y == 0 or x == 0 or z + 1 == depth or y + 1 == height or x + 1 == width:
            touched[nodes.find(int((z * height + y) * width + x))] = True
    voids = 0
    for index, parent in enumerate(nodes.parent):
        if parent >= 0 and nodes.find(index) == index and not touched[index]:
            voids += 1
    return voids


def _chi3(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    depth, height, width = mask.shape
    z_faces = np.zeros((depth + 1, height, width), dtype=bool)
    z_faces[:-1] |= mask
    z_faces[1:] |= mask
    y_faces = np.zeros((depth, height + 1, width), dtype=bool)
    y_faces[:, :-1] |= mask
    y_faces[:, 1:] |= mask
    x_faces = np.zeros((depth, height, width + 1), dtype=bool)
    x_faces[:, :, :-1] |= mask
    x_faces[:, :, 1:] |= mask
    edges_x = np.zeros((depth + 1, height + 1, width), dtype=bool)
    edges_y = np.zeros((depth + 1, height, width + 1), dtype=bool)
    edges_z = np.zeros((depth, height + 1, width + 1), dtype=bool)
    for first in (0, 1):
        for second in (0, 1):
            edges_x[first : first + depth, second : second + height] |= mask
            edges_y[first : first + depth, :, second : second + width] |= mask
            edges_z[:, first : first + height, second : second + width] |= mask
    vertices = np.zeros((depth + 1, height + 1, width + 1), dtype=bool)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                vertices[dz : dz + depth, dy : dy + height, dx : dx + width] |= mask
    faces = int(z_faces.sum() + y_faces.sum() + x_faces.sum())
    edges = int(edges_x.sum() + edges_y.sum() + edges_z.sum())
    return int(vertices.sum()) - edges + faces - int(mask.sum())
