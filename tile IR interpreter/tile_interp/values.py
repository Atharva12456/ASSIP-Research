from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

Scalar = float | int | bool

DTYPES: tuple[str, ...] = ("bool", "i32", "i64", "f32", "f64")
FLOAT_DTYPES = frozenset({"f32", "f64"})
INT_DTYPES = frozenset({"i32", "i64"})

_RANK = {name: rank for rank, name in enumerate(DTYPES)}


class ShapeError(Exception):
    """Raised for shape, broadcast, index, and dtype errors on tiles."""


def check_dtype(dtype: str) -> str:
    """Validate a dtype name and return it."""
    if dtype not in _RANK:
        raise ShapeError(f"unknown dtype: {dtype!r}")
    return dtype


def result_dtype(a: str, b: str) -> str:
    """Promote two dtypes under bool < i32 < i64 < f32 < f64."""
    return a if _RANK[check_dtype(a)] >= _RANK[check_dtype(b)] else b


def arith_dtype(a: str, b: str) -> str:
    """Promotion for add/sub/mul/floordiv/mod/pow; bool operands become i32."""
    promoted = result_dtype(a, b)
    return "i32" if promoted == "bool" else promoted


def float_dtype(dtype: str) -> str:
    """Promotion for true division: integral dtypes become f64."""
    return dtype if check_dtype(dtype) in FLOAT_DTYPES else "f64"


def infer_dtype(value: object) -> str:
    """Infer a dtype from a Python scalar: bool, int -> i64, float -> f64."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "i64"
    if isinstance(value, float):
        return "f64"
    raise ShapeError(f"cannot infer dtype from {type(value).__name__}")


def cast_value(value: object, dtype: str) -> Scalar:
    """Coerce a Python scalar into the storage form for dtype."""
    check_dtype(dtype)
    try:
        if dtype == "bool":
            return bool(value)
        if dtype in INT_DTYPES:
            return int(value)  # type: ignore[arg-type]
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ShapeError(f"cannot store {value!r} as {dtype}: {exc}") from exc


def normalize_shape(shape: object) -> tuple[int, ...]:
    """Coerce an int or an iterable of ints into a shape tuple."""
    if isinstance(shape, int) and not isinstance(shape, bool):
        return (int(shape),)
    if isinstance(shape, (list, tuple)):
        dims: list[int] = []
        for dim in shape:
            if isinstance(dim, bool) or not isinstance(dim, int):
                raise ShapeError(f"shape dimensions must be ints, got {dim!r}")
            if dim < 0:
                raise ShapeError(f"shape dimensions must be non-negative, got {dim}")
            dims.append(int(dim))
        return tuple(dims)
    raise ShapeError(f"invalid shape: {shape!r}")


def broadcast_shapes(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    """Numpy broadcasting: right-align, stretch 1s, treat missing dims as 1."""
    left = normalize_shape(a)
    right = normalize_shape(b)
    ndim = max(len(left), len(right))
    left = (1,) * (ndim - len(left)) + left
    right = (1,) * (ndim - len(right)) + right
    out: list[int] = []
    for axis, (x, y) in enumerate(zip(left, right)):
        if x == y:
            out.append(x)
        elif x == 1:
            out.append(y)
        elif y == 1:
            out.append(x)
        else:
            raise ShapeError(
                f"cannot broadcast {tuple(a)} with {tuple(b)}: axis {axis} is {x} vs {y}"
            )
    return tuple(out)


def _strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [0] * len(shape)
    acc = 1
    for axis in range(len(shape) - 1, -1, -1):
        strides[axis] = acc
        acc *= shape[axis]
    return tuple(strides)


def _offsets(shape: tuple[int, ...], strides: tuple[int, ...]) -> Iterator[int]:
    total = math.prod(shape)
    if total == 0:
        return
    ndim = len(shape)
    if ndim == 0:
        yield 0
        return
    index = [0] * ndim
    offset = 0
    for _ in range(total):
        yield offset
        for axis in range(ndim - 1, -1, -1):
            index[axis] += 1
            offset += strides[axis]
            if index[axis] < shape[axis]:
                break
            offset -= strides[axis] * shape[axis]
            index[axis] = 0


def _unflatten(flat: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    index = [0] * len(shape)
    for axis in range(len(shape) - 1, -1, -1):
        dim = shape[axis]
        index[axis] = flat % dim if dim else 0
        flat //= dim if dim else 1
    return tuple(index)


def _digest_token(value: object, dtype: str) -> str:
    if dtype == "bool":
        return "T" if value else "F"
    if dtype in INT_DTYPES:
        return str(int(value))  # type: ignore[arg-type]
    number = float(value)  # type: ignore[arg-type]
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    if number == 0.0:
        return "0.000000000000e+00"
    return f"{number:.12e}"


@dataclass(slots=True)
class Tile:
    """Dense row-major tile of scalars with a fixed shape and dtype."""

    shape: tuple[int, ...]
    dtype: str
    data: list[Scalar]

    def __post_init__(self) -> None:
        self.shape = normalize_shape(self.shape)
        check_dtype(self.dtype)
        if not isinstance(self.data, list):
            self.data = list(self.data)
        expected = math.prod(self.shape)
        if len(self.data) != expected:
            raise ShapeError(
                f"tile of shape {self.shape} needs {expected} values, got {len(self.data)}"
            )

    @property
    def size(self) -> int:
        """Number of elements."""
        return math.prod(self.shape)

    @property
    def ndim(self) -> int:
        """Number of axes."""
        return len(self.shape)

    @staticmethod
    def scalar(value: object, dtype: str | None = None) -> Tile:
        """A 0-d tile holding one value."""
        resolved = infer_dtype(value) if dtype is None else check_dtype(dtype)
        return Tile((), resolved, [cast_value(value, resolved)])

    @staticmethod
    def zeros(shape: object, dtype: str = "f32") -> Tile:
        """A tile of zeros."""
        resolved = check_dtype(dtype)
        dims = normalize_shape(shape)
        return Tile(dims, resolved, [cast_value(0, resolved)] * math.prod(dims))

    @staticmethod
    def full(shape: object, value: object, dtype: str | None = None) -> Tile:
        """A tile filled with one value."""
        resolved = infer_dtype(value) if dtype is None else check_dtype(dtype)
        dims = normalize_shape(shape)
        return Tile(dims, resolved, [cast_value(value, resolved)] * math.prod(dims))

    @staticmethod
    def arange(start: int, stop: int, step: int = 1, dtype: str = "i32") -> Tile:
        """A 1-d ramp, half-open like Python's range."""
        resolved = check_dtype(dtype)
        if step == 0:
            raise ShapeError("arange step must be non-zero")
        values = [cast_value(value, resolved) for value in range(int(start), int(stop), int(step))]
        return Tile((len(values),), resolved, values)

    @staticmethod
    def from_nested(nested: object, dtype: str | None = None) -> Tile:
        """Build a tile from nested lists or a bare scalar."""
        shape = _nested_shape(nested)
        flat: list[object] = []
        _flatten_nested(nested, shape, 0, flat)
        resolved = _infer_many(flat) if dtype is None else check_dtype(dtype)
        return Tile(shape, resolved, [cast_value(value, resolved) for value in flat])

    @staticmethod
    def from_flat(data: Sequence[object], shape: object, dtype: str | None = None) -> Tile:
        """Build a tile from a flat row-major sequence."""
        flat = list(data)
        dims = normalize_shape(shape)
        expected = math.prod(dims)
        if len(flat) != expected:
            raise ShapeError(f"shape {dims} needs {expected} values, got {len(flat)}")
        resolved = _infer_many(flat) if dtype is None else check_dtype(dtype)
        return Tile(dims, resolved, [cast_value(value, resolved) for value in flat])

    def copy(self) -> Tile:
        """An independent tile with the same contents."""
        return Tile(self.shape, self.dtype, list(self.data))

    def astype(self, dtype: str) -> Tile:
        """A tile with the same shape, values coerced to dtype."""
        resolved = check_dtype(dtype)
        return Tile(self.shape, resolved, [cast_value(value, resolved) for value in self.data])

    def to_nested(self) -> object:
        """Nested lists, or the bare value for a 0-d tile."""
        if self.ndim == 0:
            return self.data[0]
        strides = _strides(self.shape)

        def build(axis: int, offset: int) -> list[object]:
            if axis == self.ndim - 1:
                return [self.data[offset + i * strides[axis]] for i in range(self.shape[axis])]
            return [build(axis + 1, offset + i * strides[axis]) for i in range(self.shape[axis])]

        return build(0, 0)

    def item(self) -> Scalar:
        """The single value of a size-1 tile."""
        if self.size != 1:
            raise ShapeError(f"item() needs a size-1 tile, got shape {self.shape}")
        return self.data[0]

    def reshape(self, shape: object) -> Tile:
        """A tile with the same row-major data under a new shape; one dim may be -1."""
        dims = _resolve_reshape(shape, self.size)
        return Tile(dims, self.dtype, list(self.data))

    def broadcast_to(self, shape: object) -> Tile:
        """Stretch this tile to a compatible larger shape under numpy rules."""
        target = normalize_shape(shape)
        if len(target) < self.ndim:
            raise ShapeError(f"cannot broadcast shape {self.shape} to {target}")
        source = (1,) * (len(target) - self.ndim) + self.shape
        strides = _strides(source)
        effective: list[int] = []
        for axis, (have, want) in enumerate(zip(source, target)):
            if have == want:
                effective.append(strides[axis])
            elif have == 1:
                effective.append(0)
            else:
                raise ShapeError(
                    f"cannot broadcast shape {self.shape} to {target}: axis {axis} is {have} vs {want}"
                )
        data = [self.data[offset] for offset in _offsets(target, tuple(effective))]
        return Tile(target, self.dtype, data)

    def map(self, fn: Callable[[Scalar], object], dtype: str | None = None) -> Tile:
        """Apply fn elementwise."""
        resolved = self.dtype if dtype is None else check_dtype(dtype)
        return Tile(self.shape, resolved, [cast_value(fn(value), resolved) for value in self.data])

    def zip_with(
        self,
        other: Tile,
        fn: Callable[[Scalar, Scalar], object],
        dtype: str | None = None,
    ) -> Tile:
        """Apply fn elementwise against another tile, broadcasting both sides."""
        if not isinstance(other, Tile):
            raise ShapeError(f"zip_with expects a Tile, got {type(other).__name__}")
        shape = broadcast_shapes(self.shape, other.shape)
        resolved = result_dtype(self.dtype, other.dtype) if dtype is None else check_dtype(dtype)
        left = self.broadcast_to(shape)
        right = other.broadcast_to(shape)
        data = [
            cast_value(fn(x, y), resolved) for x, y in zip(left.data, right.data)
        ]
        return Tile(shape, resolved, data)

    def reduce(
        self,
        fn: Callable[[object, Scalar], object],
        axis: int | None = None,
        init: object = None,
    ) -> Tile:
        """Fold fn over every element, or along one axis."""
        if axis is None:
            values = self.data
            if init is None:
                if not values:
                    raise ShapeError("reduce over an empty tile needs an init value")
                acc: object = values[0]
                rest: Sequence[Scalar] = values[1:]
            else:
                acc = init
                rest = values
            for value in rest:
                acc = fn(acc, value)
            resolved = _reduced_dtype(self.dtype, acc)
            return Tile((), resolved, [cast_value(acc, resolved)])

        normalized = axis + self.ndim if axis < 0 else axis
        if not 0 <= normalized < self.ndim:
            raise ShapeError(f"axis {axis} is out of range for shape {self.shape}")
        length = self.shape[normalized]
        if length == 0 and init is None:
            raise ShapeError("reduce over an empty axis needs an init value")
        out_shape = self.shape[:normalized] + self.shape[normalized + 1 :]
        strides = _strides(self.shape)
        step = strides[normalized]
        results: list[object] = []
        for flat in range(math.prod(out_shape)):
            index = _unflatten(flat, out_shape)
            full = index[:normalized] + (0,) + index[normalized:]
            base = sum(i * s for i, s in zip(full, strides))
            if init is None:
                acc = self.data[base]
                span = range(1, length)
            else:
                acc = init
                span = range(length)
            for k in span:
                acc = fn(acc, self.data[base + k * step])
            results.append(acc)
        resolved = _reduced_dtype(self.dtype, results[0] if results else 0)
        return Tile(out_shape, resolved, [cast_value(value, resolved) for value in results])

    def matmul(self, other: Tile) -> Tile:
        """Matrix product; 1-d operands follow numpy's promote-then-drop rule."""
        if not isinstance(other, Tile):
            raise ShapeError(f"matmul expects a Tile, got {type(other).__name__}")
        if self.ndim == 0 or other.ndim == 0:
            raise ShapeError("matmul needs at least 1-d operands")
        if self.ndim > 2 or other.ndim > 2:
            raise ShapeError("matmul supports 1-d and 2-d operands only")
        left = self.reshape((1, self.shape[0])) if self.ndim == 1 else self
        right = other.reshape((other.shape[0], 1)) if other.ndim == 1 else other
        rows, inner = left.shape
        inner_other, cols = right.shape
        if inner != inner_other:
            raise ShapeError(f"matmul shape mismatch: {self.shape} @ {other.shape}")
        resolved = arith_dtype(self.dtype, other.dtype)
        data: list[Scalar] = []
        for i in range(rows):
            row_base = i * inner
            for j in range(cols):
                total: object = 0
                for k in range(inner):
                    total = total + left.data[row_base + k] * right.data[k * cols + j]
                data.append(cast_value(total, resolved))
        if self.ndim == 1 and other.ndim == 1:
            return Tile((), resolved, data)
        if self.ndim == 1:
            return Tile((cols,), resolved, data)
        if other.ndim == 1:
            return Tile((rows,), resolved, data)
        return Tile((rows, cols), resolved, data)

    def transpose(self) -> Tile:
        """Reverse every axis; a no-op copy below 2 dimensions."""
        if self.ndim < 2:
            return self.copy()
        target = tuple(reversed(self.shape))
        strides = tuple(reversed(_strides(self.shape)))
        return Tile(target, self.dtype, [self.data[offset] for offset in _offsets(target, strides)])

    def digest(self) -> str:
        """Stable 8-hex-char content hash; floats are rounded before hashing."""
        hasher = hashlib.sha256()
        hasher.update(f"{self.dtype}|{','.join(str(dim) for dim in self.shape)}|".encode("utf-8"))
        for value in self.data:
            hasher.update(_digest_token(value, self.dtype).encode("utf-8"))
            hasher.update(b";")
        return hasher.hexdigest()[:8]

    def allclose(self, other: Tile, rtol: float = 1e-9, atol: float = 1e-12) -> bool:
        """Numeric comparison with broadcasting; NaN never compares equal."""
        if not isinstance(other, Tile):
            return False
        try:
            shape = broadcast_shapes(self.shape, other.shape)
            left = self.broadcast_to(shape)
            right = other.broadcast_to(shape)
        except ShapeError:
            return False
        for x, y in zip(left.data, right.data):
            if isinstance(x, bool) or isinstance(y, bool):
                if bool(x) != bool(y):
                    return False
                continue
            fx = float(x)
            fy = float(y)
            if math.isnan(fx) or math.isnan(fy):
                return False
            if math.isinf(fx) or math.isinf(fy):
                if fx != fy:
                    return False
                continue
            if abs(fx - fy) > atol + rtol * abs(fy):
                return False
        return True

    def describe(self) -> str:
        """Compact 'shape:dtype@digest' label used by trace lines."""
        dims = "x".join(str(dim) for dim in self.shape) or "scalar"
        return f"{dims}:{self.dtype}@{self.digest()}"


def _resolve_reshape(shape: object, size: int) -> tuple[int, ...]:
    if isinstance(shape, int) and not isinstance(shape, bool):
        raw: tuple[int, ...] = (int(shape),)
    elif isinstance(shape, (list, tuple)):
        raw = tuple(int(dim) for dim in shape)
    else:
        raise ShapeError(f"invalid shape: {shape!r}")
    if raw.count(-1) > 1:
        raise ShapeError("reshape allows at most one -1 dimension")
    if -1 in raw:
        known = math.prod(dim for dim in raw if dim != -1)
        if known == 0 or size % known:
            raise ShapeError(f"cannot reshape size {size} into {raw}")
        raw = tuple(size // known if dim == -1 else dim for dim in raw)
    dims = normalize_shape(raw)
    if math.prod(dims) != size:
        raise ShapeError(f"cannot reshape size {size} into {dims}")
    return dims


def _nested_shape(nested: object) -> tuple[int, ...]:
    shape: list[int] = []
    node = nested
    while isinstance(node, (list, tuple)):
        shape.append(len(node))
        if not node:
            break
        node = node[0]
    return tuple(shape)


def _flatten_nested(nested: object, shape: tuple[int, ...], depth: int, out: list[object]) -> None:
    if depth == len(shape):
        if isinstance(nested, (list, tuple)):
            raise ShapeError("nested data is ragged")
        out.append(nested)
        return
    if not isinstance(nested, (list, tuple)):
        raise ShapeError("nested data is ragged")
    if len(nested) != shape[depth]:
        raise ShapeError(f"nested data is ragged at depth {depth}")
    for item in nested:
        _flatten_nested(item, shape, depth + 1, out)


def _infer_many(values: Sequence[object]) -> str:
    dtype = "bool"
    for value in values:
        dtype = result_dtype(dtype, infer_dtype(value))
    return dtype


def _reduced_dtype(dtype: str, sample: object) -> str:
    if isinstance(sample, bool):
        return dtype
    if isinstance(sample, float):
        return dtype if dtype in FLOAT_DTYPES else "f64"
    if isinstance(sample, int) and dtype == "bool":
        return "i32"
    return dtype
