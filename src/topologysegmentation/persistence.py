"""Cubical persistence on a 2D or 3D probability map.

Closed cubes that touch at a corner form one piece, so foreground connectivity
includes diagonal neighbors. A one-voxel frame of zeros makes the exterior a
single background. Gudhi reads cell indices in column-major order and is
imported only when a diagram is requested.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Bar:
    dimension: int
    birth: tuple[int, ...] | None
    death: tuple[int, ...] | None
    birth_value: float
    death_value: float

    @property
    def persistence(self) -> float:
        return float(self.birth_value - self.death_value)

    def alive(self, threshold: float) -> bool:
        return self.birth_value >= threshold and self.death_value < threshold


def cubical_bars(probability, epsilon: float = 1e-6) -> list[Bar]:
    """Persistence bars of a superlevel filtration, in probability coordinates.

    ``probability`` is a 2D or 3D array. Birth is the probability at which a
    feature appears as the threshold falls, and death is where it disappears.
    An essential class has no death voxel and is recorded with death value 0.
    """
    import gudhi

    image = _spatial(probability, dtype=np.float64)
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D map, got shape {tuple(image.shape)}")
    original_shape = image.shape
    padded = np.pad(image, 1, mode="constant", constant_values=0.0)
    filtration = np.asfortranarray(1.0 - padded)
    complex_ = gudhi.CubicalComplex(top_dimensional_cells=filtration)
    complex_.persistence()
    finite, essential = complex_.cofaces_of_persistence_pairs()
    bars: list[Bar] = []
    for dimension in range(image.ndim):
        for birth_index, death_index in _pairs(finite, dimension):
            bar = _make_bar(dimension, int(birth_index), int(death_index), filtration, original_shape)
            if bar is not None and bar.persistence > epsilon:
                bars.append(bar)
        for birth_index in _indices(essential, dimension):
            bar = _make_bar(dimension, int(birth_index), None, filtration, original_shape)
            if bar is not None and bar.persistence > epsilon:
                bars.append(bar)
    return bars


def betti_numbers(binary_mask: np.ndarray, threshold: float = 0.5) -> tuple[int, ...]:
    """Betti numbers of the features alive at ``threshold``.

    A 2D mask returns ``(β0, β1)``. A volume returns ``(β0, β1, β2)``.
    Voxels that share a corner are one component. An empty mask is all zeros.
    Values above zero are the foreground. On a 0/1 mask every foreground bar is
    born at 1 and dies at 0, so a threshold outside ``(0, 1]`` counts nothing.
    """
    image = _spatial(binary_mask)
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D mask, got shape {tuple(image.shape)}")
    if not 0.0 < float(threshold) <= 1.0:
        return tuple(0 for _ in range(image.ndim))
    mask = np.ascontiguousarray(image > 0, dtype=np.uint8)
    from topologysegmentation._native import binary_betti

    counted = binary_betti(mask)
    if counted is None:
        from topologysegmentation._native.fallback import binary_betti as betti_fallback

        counted = betti_fallback(mask.astype(bool))
    return counted


def _spatial(array, dtype=None) -> np.ndarray:
    """Keep every spatial axis, including sides of length 1.

    A leading singleton on a 4D array is the channel that ``topology_errors``
    used to leave attached. Squeezing the whole array would turn a 1×N frame
    into a vector and a one-slice volume into a frame.
    """
    image = np.asarray(array, dtype=dtype)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    return image


def _pairs(groups, dimension: int) -> np.ndarray:
    array = _indices(groups, dimension)
    if array.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return array.reshape(-1, 2)


def _indices(groups, dimension: int) -> np.ndarray:
    if dimension >= len(groups):
        return np.zeros((0,), dtype=np.int64)
    return np.asarray(groups[dimension])


def _make_bar(dimension, birth_index, death_index, filtration, original_shape):
    values = filtration.ravel(order="F")
    limit = values.size
    if birth_index < 0 or birth_index >= limit:
        return None
    if death_index is not None and (death_index < 0 or death_index >= limit):
        return None
    birth_value = 1.0 - float(values[birth_index])
    if death_index is None:
        death_value = 0.0
    else:
        death_value = 1.0 - float(values[death_index])
    if birth_value < death_value:
        birth_index, death_index = death_index, birth_index
        birth_value, death_value = death_value, birth_value
    return Bar(
        dimension=dimension,
        birth=_coordinate(birth_index, filtration.shape, original_shape),
        death=None if death_index is None else _coordinate(death_index, filtration.shape, original_shape),
        birth_value=birth_value,
        death_value=death_value,
    )


def _coordinate(index: int, padded_shape, original_shape):
    coords = np.unravel_index(index, padded_shape, order="F")
    point = tuple(int(coord) - 1 for coord in coords)
    if any(coord < 0 or coord >= size for coord, size in zip(point, original_shape)):
        return None
    return point
