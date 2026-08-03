"""Structured tensor and partition views over flat buffers.

CUDA Tile IR has two ways to touch memory. The pointer style computes an explicit
index tile and gathers through it (see the load/store opcodes). The view style is
higher level: make_tensor_view wraps a buffer as a strided N-D matrix, and
make_partition_view slices that matrix into fixed-size tiles, optionally permuting
the axes so a tile can be read transposed.

Both view kinds lower to the SAME gather the pointer style uses: a partition view
turns a block coordinate into a flat index tile, and the interpreter then reads it
through the ordinary Memory.load. Keeping one gather path means the two styles can
never disagree about what a load means.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .values import Tile


@dataclass(slots=True)
class TensorView:
    """A flat buffer seen as a strided N-D tensor: element[c] lives at sum(c * strides)."""

    buffer: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.shape) != len(self.strides):
            raise ValueError(
                f"tensor view {self.buffer!r}: shape {self.shape} and strides "
                f"{self.strides} have different ranks"
            )


@dataclass(slots=True)
class PartitionView:
    """A tensor view tiled into fixed-size blocks, with an axis permutation.

    dim_map[d] is the tensor axis that partition axis d runs along. dim_map=[1,0]
    over a (K, M) tensor yields tiles indexed as (m-block, k-block) whose elements
    are read transposed out of the (K, M) storage.
    """

    tensor: TensorView
    tile: tuple[int, ...]
    dim_map: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.tile) != len(self.dim_map):
            raise ValueError(
                f"partition view: tile {self.tile} and dim_map {self.dim_map} "
                f"have different ranks"
            )

    def index_space(self) -> tuple[int, ...]:
        """The number of tiles along each partition axis, rounding up."""
        counts = []
        for axis, extent in enumerate(self.tile):
            logical = self.tensor.shape[self.dim_map[axis]]
            counts.append((logical + extent - 1) // extent)
        return tuple(counts)

    def index_tile(self, block: tuple[int, ...]) -> Tile:
        """Flat buffer indices for the tile at the given block coordinate.

        The result has the tile's shape; interpreting it against the buffer with a
        gather reproduces the block, transpose and strides included.
        """
        if len(block) != len(self.tile):
            raise ValueError(
                f"block coordinate {block} does not match tile rank {len(self.tile)}"
            )
        tile = self.tile
        strides = self.tensor.strides
        dim_map = self.dim_map
        rank = len(tile)
        tensor_rank = len(self.tensor.shape)
        data: list[int] = []
        for position in range(math.prod(tile)):
            coords = _unravel(position, tile)
            tensor_coord = [0] * tensor_rank
            for axis in range(rank):
                tensor_coord[dim_map[axis]] = block[axis] * tile[axis] + coords[axis]
            flat = sum(tensor_coord[t] * strides[t] for t in range(tensor_rank))
            data.append(flat)
        return Tile(tile, "i64", data)


def _unravel(position: int, shape: tuple[int, ...]) -> list[int]:
    """Row-major coordinate of a flat position within shape."""
    coords = [0] * len(shape)
    remainder = position
    for axis in range(len(shape) - 1, -1, -1):
        coords[axis] = remainder % shape[axis]
        remainder //= shape[axis]
    return coords
