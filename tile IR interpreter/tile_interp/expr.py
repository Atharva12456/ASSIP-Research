from __future__ import annotations

import ast
import math
import operator
from typing import Callable, Mapping

from .values import (
    ShapeError,
    Tile,
    arith_dtype,
    float_dtype,
    infer_dtype,
    result_dtype,
)


class ExprError(Exception):
    """Raised when an attribute expression is unsupported or cannot be evaluated."""


LITERALS: dict[str, object] = {
    "true": True,
    "false": False,
    "True": True,
    "False": False,
    "none": None,
    "None": None,
}

_ARITH: dict[str, Callable[[object, object], object]] = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": operator.truediv,
    "floordiv": operator.floordiv,
    "mod": operator.mod,
    "pow": operator.pow,
}

_COMPARE: dict[str, Callable[[object, object], bool]] = {
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
    "eq": operator.eq,
    "ne": operator.ne,
}

_BINOP_NODES: dict[type, str] = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.Div: "div",
    ast.FloorDiv: "floordiv",
    ast.Mod: "mod",
    ast.Pow: "pow",
}

_COMPARE_NODES: dict[type, str] = {
    ast.Lt: "lt",
    ast.LtE: "le",
    ast.Gt: "gt",
    ast.GtE: "ge",
    ast.Eq: "eq",
    ast.NotEq: "ne",
}

_BANNED_NODES: tuple[type, ...] = (
    ast.Attribute,
    ast.Call,
    ast.Lambda,
    ast.NamedExpr,
    ast.Starred,
    ast.JoinedStr,
    ast.FormattedValue,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
    ast.IfExp,
    ast.Dict,
    ast.Set,
)


def eval_attr(text: str, env: Mapping[str, object]) -> object:
    """Evaluate an IR attribute value against the live environment; never uses eval()."""
    if not isinstance(text, str):
        raise ExprError(f"expression must be text, got {type(text).__name__}")
    stripped = text.strip()
    if not stripped:
        raise ExprError("empty expression")
    if stripped in LITERALS and stripped not in env:
        return LITERALS[stripped]
    tree = _parse(stripped)
    return _Evaluator(env).visit(tree.body)


def parse_names(text: str) -> list[str]:
    """Identifiers an expression reads, in source order, without duplicates."""
    if not isinstance(text, str) or not text.strip():
        return []
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError:
        return []
    found = [node for node in ast.walk(tree) if isinstance(node, ast.Name)]
    found.sort(key=lambda node: (node.lineno, node.col_offset))
    seen: dict[str, None] = {}
    for node in found:
        if node.id not in LITERALS:
            seen.setdefault(node.id, None)
    return list(seen)


def apply_binary(op: str, lhs: object, rhs: object) -> object:
    """Elementwise arithmetic for the IR arithmetic opcodes, with broadcasting."""
    fn = _ARITH.get(op)
    if fn is None:
        raise ExprError(f"unsupported binary operator: {op!r}")
    if not isinstance(lhs, Tile) and not isinstance(rhs, Tile):
        return _guard(fn, lhs, rhs)
    left = _tile(lhs)
    right = _tile(rhs)
    dtype = _binary_dtype(op, left.dtype, right.dtype)
    return left.zip_with(right, lambda a, b: _guard(fn, a, b), dtype)


def apply_compare(op: str, lhs: object, rhs: object) -> object:
    """Elementwise comparison producing bools or a bool tile."""
    fn = _COMPARE.get(op)
    if fn is None:
        raise ExprError(f"unsupported comparison: {op!r}")
    if not isinstance(lhs, Tile) and not isinstance(rhs, Tile):
        return bool(_guard(fn, lhs, rhs))
    left = _tile(lhs)
    right = _tile(rhs)
    return left.zip_with(right, lambda a, b: bool(_guard(fn, a, b)), "bool")


def apply_boolean(op: str, values: list[object]) -> object:
    """Elementwise 'and'/'or' over one or more operands."""
    if op not in {"and", "or"}:
        raise ExprError(f"unsupported boolean operator: {op!r}")
    if not values:
        raise ExprError(f"'{op}' needs at least one operand")
    combine = (lambda a, b: bool(a) and bool(b)) if op == "and" else (lambda a, b: bool(a) or bool(b))
    acc = values[0]
    if len(values) == 1:
        return acc.map(bool, "bool") if isinstance(acc, Tile) else bool(acc)
    for value in values[1:]:
        if isinstance(acc, Tile) or isinstance(value, Tile):
            acc = _tile(acc).zip_with(_tile(value), combine, "bool")
        else:
            acc = combine(acc, value)
    return acc


def apply_unary(op: str, value: object) -> object:
    """Unary '+', '-' and logical 'not'."""
    if op == "not":
        if isinstance(value, Tile):
            return value.map(lambda item: not item, "bool")
        return not truthy(value)
    if op not in {"pos", "neg"}:
        raise ExprError(f"unsupported unary operator: {op!r}")
    fn = operator.pos if op == "pos" else operator.neg
    if isinstance(value, Tile):
        dtype = "i32" if value.dtype == "bool" else value.dtype
        return value.map(lambda item: _guard(fn, item), dtype)
    return _guard(fn, value)


def truthy(value: object) -> bool:
    """Condition test: a size-1 tile uses its element, larger tiles require all lanes."""
    if isinstance(value, Tile):
        if value.size == 0:
            return False
        if value.size == 1:
            return bool(value.data[0])
        return all(bool(item) for item in value.data)
    return bool(value)


def subscript(value: object, key: object) -> object:
    """Index a tile along its leading axes, or a plain Python sequence."""
    if isinstance(value, Tile):
        return _tile_subscript(value, key)
    if isinstance(value, (list, tuple, str)):
        try:
            return value[key]  # type: ignore[index]
        except (IndexError, KeyError, TypeError) as exc:
            raise ExprError(f"bad subscript {key!r}: {exc}") from exc
    raise ExprError(f"cannot subscript {type(value).__name__}")


def _parse(text: str) -> ast.Expression:
    try:
        return ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ExprError(f"cannot parse expression {text!r}: {exc.msg}") from exc


class _Evaluator:
    """Whitelist visitor over a parsed attribute expression."""

    __slots__ = ("env",)

    def __init__(self, env: Mapping[str, object]) -> None:
        self.env = env

    def visit(self, node: ast.AST) -> object:
        if isinstance(node, _BANNED_NODES):
            raise ExprError(f"{type(node).__name__} is not allowed in IR expressions")
        method = getattr(self, f"_on_{type(node).__name__}", None)
        if method is None:
            raise ExprError(f"{type(node).__name__} is not allowed in IR expressions")
        return method(node)

    def _on_Name(self, node: ast.Name) -> object:
        if node.id.startswith("__") or node.id.endswith("__"):
            raise ExprError(f"dunder names are not allowed: {node.id!r}")
        if node.id in self.env:
            return self.env[node.id]
        if node.id in LITERALS:
            return LITERALS[node.id]
        raise ExprError(f"unknown name: {node.id!r}")

    def _on_Constant(self, node: ast.Constant) -> object:
        value = node.value
        if isinstance(value, (bool, int, float, str)) or value is None:
            return value
        raise ExprError(f"unsupported constant: {value!r}")

    def _on_UnaryOp(self, node: ast.UnaryOp) -> object:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return apply_unary("neg", operand)
        if isinstance(node.op, ast.UAdd):
            return apply_unary("pos", operand)
        if isinstance(node.op, ast.Not):
            return apply_unary("not", operand)
        raise ExprError(f"unsupported unary operator: {type(node.op).__name__}")

    def _on_BinOp(self, node: ast.BinOp) -> object:
        name = _BINOP_NODES.get(type(node.op))
        if name is None:
            raise ExprError(f"unsupported binary operator: {type(node.op).__name__}")
        return apply_binary(name, self.visit(node.left), self.visit(node.right))

    def _on_BoolOp(self, node: ast.BoolOp) -> object:
        name = "and" if isinstance(node.op, ast.And) else "or"
        return apply_boolean(name, [self.visit(value) for value in node.values])

    def _on_Compare(self, node: ast.Compare) -> object:
        left = self.visit(node.left)
        result: object = None
        for op, comparator in zip(node.ops, node.comparators):
            name = _COMPARE_NODES.get(type(op))
            if name is None:
                raise ExprError(f"unsupported comparison: {type(op).__name__}")
            right = self.visit(comparator)
            current = apply_compare(name, left, right)
            result = current if result is None else apply_boolean("and", [result, current])
            left = right
        return result

    def _on_Subscript(self, node: ast.Subscript) -> object:
        return subscript(self.visit(node.value), self.visit(node.slice))

    def _on_Slice(self, node: ast.Slice) -> object:
        return slice(
            self.visit(node.lower) if node.lower is not None else None,
            self.visit(node.upper) if node.upper is not None else None,
            self.visit(node.step) if node.step is not None else None,
        )

    def _on_Tuple(self, node: ast.Tuple) -> object:
        return tuple(self.visit(item) for item in node.elts)

    def _on_List(self, node: ast.List) -> object:
        return [self.visit(item) for item in node.elts]


def _tile(value: object) -> Tile:
    if isinstance(value, Tile):
        return value
    if isinstance(value, (list, tuple)):
        return Tile.from_nested(list(value))
    try:
        return Tile.scalar(value, infer_dtype(value))
    except ShapeError as exc:
        raise ExprError(f"cannot use {value!r} as a tile operand: {exc}") from exc


def _binary_dtype(op: str, left: str, right: str) -> str:
    if op == "div":
        return float_dtype(result_dtype(left, right))
    return arith_dtype(left, right)


def _guard(fn: Callable[..., object], *args: object) -> object:
    try:
        return fn(*args)
    except ZeroDivisionError as exc:
        raise ExprError(f"division by zero: {exc}") from exc
    except (TypeError, ValueError, OverflowError) as exc:
        raise ExprError(f"cannot apply operator: {exc}") from exc


def _tile_subscript(tile: Tile, key: object) -> object:
    keys = list(key) if isinstance(key, tuple) else [key]
    if len(keys) > tile.ndim:
        raise ExprError(f"too many indices for shape {tile.shape}")
    current = tile
    consumed = 0
    for item in keys:
        current = _take_axis(current, consumed, item)
        if isinstance(item, slice):
            consumed += 1
    if current.ndim == 0:
        return current.data[0]
    return current


def _take_axis(tile: Tile, axis: int, key: object) -> Tile:
    length = tile.shape[axis]
    if isinstance(key, slice):
        picks = list(range(*key.indices(length)))
        out_shape = tile.shape[:axis] + (len(picks),) + tile.shape[axis + 1 :]
    elif isinstance(key, bool) or not isinstance(key, int):
        if isinstance(key, Tile) and key.size == 1:
            picks = [_normalize_index(int(key.data[0]), length, tile.shape)]
        else:
            raise ExprError(f"unsupported index {key!r}")
        out_shape = tile.shape[:axis] + tile.shape[axis + 1 :]
    else:
        picks = [_normalize_index(int(key), length, tile.shape)]
        out_shape = tile.shape[:axis] + tile.shape[axis + 1 :]
    outer = math.prod(tile.shape[:axis])
    inner = math.prod(tile.shape[axis + 1 :])
    data: list[object] = []
    for block in range(outer):
        base = block * length * inner
        for pick in picks:
            start = base + pick * inner
            data.extend(tile.data[start : start + inner])
    return Tile(out_shape, tile.dtype, data)


def _normalize_index(index: int, length: int, shape: tuple[int, ...]) -> int:
    resolved = index + length if index < 0 else index
    if not 0 <= resolved < length:
        raise ExprError(f"index {index} is out of range for shape {shape}")
    return resolved
