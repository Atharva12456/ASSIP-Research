from __future__ import annotations

import math
from dataclasses import dataclass

from .values import ShapeError, Tile, cast_value


class MemoryError_(Exception):
    """Raised for unknown buffers and out-of-range buffer indices."""


@dataclass(slots=True)
class Buffer:
    """A named tile living in a named address space."""

    name: str
    tile: Tile
    space: str = "global"


class Memory:
    """Named buffers in named address spaces; the target of load and store."""

    __slots__ = ("_buffers",)

    def __init__(self) -> None:
        self._buffers: dict[str, Buffer] = {}

    def declare(self, name: str, tile: Tile, space: str = "global") -> Buffer:
        """Install a private copy of tile under name, replacing any previous buffer."""
        if not isinstance(tile, Tile):
            raise MemoryError_(f"buffer {name!r} needs a Tile, got {type(tile).__name__}")
        buffer = Buffer(name, tile.copy(), space)
        self._buffers[name] = buffer
        return buffer

    def ensure(self, name: str, shape: object, dtype: str = "f32", space: str = "global") -> Buffer:
        """Return the existing buffer, or declare a zero-filled one."""
        existing = self._buffers.get(name)
        if existing is not None:
            return existing
        return self.declare(name, Tile.zeros(shape, dtype), space)

    def has(self, name: str) -> bool:
        """Whether a buffer with this name exists."""
        return name in self._buffers

    def buffer(self, name: str) -> Buffer:
        """Return the named buffer, raising MemoryError_ when absent."""
        try:
            return self._buffers[name]
        except KeyError:
            known = ", ".join(sorted(self._buffers)) or "<none>"
            raise MemoryError_(f"unknown buffer {name!r}; declared: {known}") from None

    def load(
        self,
        name: str,
        index: Tile | None = None,
        mask: Tile | None = None,
        other: object = 0.0,
    ) -> Tile:
        """Read a buffer whole, or gather elementwise through an integer index tile."""
        buffer = self.buffer(name)
        tile = buffer.tile
        shape = tile.shape if index is None else _index_tile(name, index).shape
        mask_data = _broadcast_optional(mask, shape, name)
        other_data = None if mask_data is None else _broadcast_optional(_as_tile(other), shape, name)
        data: list[object] = []
        for position in range(math.prod(shape)):
            if mask_data is not None and not mask_data[position]:
                data.append(cast_value(other_data[position], tile.dtype))
                continue
            flat = position if index is None else int(index.data[position])
            if not 0 <= flat < tile.size:
                raise MemoryError_(
                    f"load from {name!r} at flat index {flat} is out of range 0..{tile.size - 1}"
                )
            data.append(tile.data[flat])
        return Tile(shape, tile.dtype, [cast_value(value, tile.dtype) for value in data])

    def store(
        self,
        name: str,
        value: Tile,
        index: Tile | None = None,
        mask: Tile | None = None,
    ) -> None:
        """Write a buffer whole, or scatter elementwise through an integer index tile."""
        buffer = self.buffer(name)
        tile = buffer.tile
        shape = tile.shape if index is None else _index_tile(name, index).shape
        values = _as_tile(value).broadcast_to(shape)
        mask_data = _broadcast_optional(mask, shape, name)
        for position in range(math.prod(shape)):
            if mask_data is not None and not mask_data[position]:
                continue
            flat = position if index is None else int(index.data[position])
            if not 0 <= flat < tile.size:
                raise MemoryError_(
                    f"store to {name!r} at flat index {flat} is out of range 0..{tile.size - 1}"
                )
            tile.data[flat] = cast_value(values.data[position], tile.dtype)

    def snapshot(self) -> dict[str, Tile]:
        """Deep copy of every buffer's tile, keyed by name."""
        return {name: buffer.tile.copy() for name, buffer in self._buffers.items()}

    def names(self) -> list[str]:
        """Buffer names, sorted for deterministic reports."""
        return sorted(self._buffers)

    def buffers(self) -> list[Buffer]:
        """Every buffer, ordered by name."""
        return [self._buffers[name] for name in sorted(self._buffers)]

    def copy(self) -> Memory:
        """An independent Memory with copies of every buffer."""
        clone = Memory()
        for name, buffer in self._buffers.items():
            clone.declare(name, buffer.tile, buffer.space)
        return clone

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._buffers

    def __len__(self) -> int:
        return len(self._buffers)

    def __repr__(self) -> str:
        parts = ", ".join(f"{name}:{buf.tile.describe()}" for name, buf in self._buffers.items())
        return f"Memory({parts})"


def _as_tile(value: object) -> Tile:
    if isinstance(value, Tile):
        return value
    return Tile.scalar(value)


def _index_tile(name: str, index: Tile) -> Tile:
    if not isinstance(index, Tile):
        raise MemoryError_(f"index for buffer {name!r} must be a Tile")
    return index


def _broadcast_optional(tile: Tile | None, shape: tuple[int, ...], name: str) -> list | None:
    if tile is None:
        return None
    try:
        return _as_tile(tile).broadcast_to(shape).data
    except ShapeError as exc:
        raise MemoryError_(f"cannot apply operand to buffer {name!r}: {exc}") from exc
