from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Iterable

from .expr import (
    ExprError,
    apply_binary,
    apply_boolean,
    apply_compare,
    eval_attr,
    parse_names,
    truthy,
)
from .ir import IRError, Op, split_list
from .values import (
    INT_DTYPES,
    ShapeError,
    Tile,
    broadcast_shapes,
    check_dtype,
    float_dtype,
    infer_dtype,
    result_dtype,
)
from .views import PartitionView, TensorView

if TYPE_CHECKING:
    from .interpreter import ExecContext


class UnsupportedOpcode(Exception):
    """Raised for an opcode, or a 'call' callee, the interpreter has no semantics for."""


@dataclass(slots=True)
class Effects:
    """What one op reads and writes, in the environment and in memory."""

    reads: list[str] = field(default_factory=list)
    writes: list[str] = field(default_factory=list)
    mem_reads: list[str] = field(default_factory=list)
    mem_writes: list[str] = field(default_factory=list)
    is_barrier: bool = False


@dataclass(slots=True)
class OpSpec:
    """The complete definition of an opcode: its cost, its effects, and its behaviour."""

    opcode: str
    kind: str
    latency: int
    effects: Callable[[Op], Effects]
    execute: Callable[["ExecContext", Op], object]


DTYPE_ALIASES: dict[str, str] = {
    "bool": "bool",
    "tl.int1": "bool",
    "int1": "bool",
    "i32": "i32",
    "int32": "i32",
    "tl.int32": "i32",
    "int": "i32",
    "i64": "i64",
    "int64": "i64",
    "tl.int64": "i64",
    "long": "i64",
    "f32": "f32",
    "float32": "f32",
    "tl.float32": "f32",
    "fp32": "f32",
    "float": "f32",
    "f64": "f64",
    "float64": "f64",
    "tl.float64": "f64",
    "fp64": "f64",
    "double": "f64",
}

_ARITH_OPS = ("add", "sub", "mul", "div", "floordiv", "mod", "pow")
_COMPARE_OPS = ("lt", "le", "gt", "ge", "eq", "ne")
_LOGIC_OPS = ("and", "or")

_OPEN_BRACKETS = "([{"
_CLOSE_BRACKETS = ")]}"


def split_args(text: str | None) -> list[str]:
    """Split a comma-joined attribute value at bracket depth zero."""
    if not text or not text.strip():
        return []
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    quote = ""
    for char in text:
        if quote:
            current.append(char)
            if char == quote:
                quote = ""
            continue
        if char in "\"'":
            quote = char
            current.append(char)
            continue
        if char == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        if char in _OPEN_BRACKETS:
            depth += 1
        elif char in _CLOSE_BRACKETS:
            depth -= 1
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def target_names(text: str | None) -> list[str]:
    """Names an assignment or loop target binds, whether 'a', 'a,b' or '(a, b)'."""
    if text is None:
        return []
    stripped = text.strip()
    if not stripped:
        return []
    if stripped[0] in "([":
        return parse_names(stripped)
    return split_list(stripped)


def unpacks(text: str | None) -> bool:
    """Whether a target spells a tuple pattern that consumes a sequence."""
    stripped = (text or "").strip()
    return bool(stripped) and stripped[0] in "(["


def output_names(op: Op) -> list[str]:
    """Environment names an op binds through its out attr."""
    return target_names(op.get("out"))


def read_names(op: Op, *keys: str) -> list[str]:
    """Identifiers read by the given attributes, de-duplicated in source order."""
    seen: dict[str, None] = {}
    for key in keys:
        for name in parse_names(op.get(key)):
            seen.setdefault(name, None)
    return list(seen)


def memory_target(op: Op) -> tuple[str, str | None]:
    """Buffer name and index expression for a load or store, in buf= or ptr= form."""
    buf = op.get("buf")
    if buf is not None:
        name = buf.strip()
        if not name.isidentifier():
            raise IRError(f"{_where_op(op)}: buf={name!r} is not a buffer name")
        return name, op.get("index")
    ptr = op.get("ptr")
    if ptr is None:
        raise IRError(f"{_where_op(op)} needs a 'buf' or 'ptr' attribute")
    names = parse_names(ptr)
    if not names:
        raise IRError(f"{_where_op(op)}: pointer {ptr!r} names no buffer")
    base = names[0]
    stripped = ptr.strip()
    if stripped == base:
        return base, op.get("index")
    return base, stripped


def spec_for(op: Op) -> OpSpec:
    """The OpSpec for an op, raising UnsupportedOpcode when the opcode is unknown."""
    spec = OPCODES.get(op.opcode)
    if spec is None:
        known = ", ".join(sorted(OPCODES))
        raise UnsupportedOpcode(f"{_where_op(op)}: unknown opcode; supported: {known}")
    return spec


def effects_of(op: Op) -> Effects:
    """Reads and writes for an op, derived from the same attrs execute() consumes."""
    return spec_for(op).effects(op)


def bind_output(ctx: "ExecContext", op: Op, value: object) -> object:
    """Bind a computed value to the op's out target(s) and return it."""
    text = op.get("out")
    names = target_names(text)
    if not names:
        return value
    if unpacks(text):
        items = _sequence(value)
        if items is None or len(items) != len(names):
            raise IRError(f"{_where_op(op)}: cannot unpack result into {names}")
        for name, item in zip(names, items):
            ctx.bind(name, item)
        return value
    for name in names:
        ctx.bind(name, value)
    return value


def bind_loop_value(ctx: "ExecContext", op: Op, value: object) -> None:
    """Bind one iteration value to a for-loop target."""
    names = loop_targets(op)
    if not names:
        raise IRError(f"{_where_op(op)} has no loop target")
    if unpacks(op.get("target")) or len(names) > 1:
        items = _sequence(value)
        if items is None or len(items) != len(names):
            raise IRError(f"{_where_op(op)}: cannot unpack {value!r} into {names}")
        for name, item in zip(names, items):
            ctx.bind(name, item)
        return
    ctx.bind(names[0], value)


def loop_targets(op: Op) -> list[str]:
    """Names a for-loop binds each iteration."""
    text = op.get("target")
    if text is None:
        plan = _c_style_for(op.get("iter") or "")
        text = plan[0] if plan else None
    return target_names(text)


def as_tile(value: object) -> Tile:
    """Coerce a scalar, list or tile into a Tile."""
    if isinstance(value, Tile):
        return value
    if isinstance(value, (list, tuple)):
        return Tile.from_nested(list(value))
    try:
        return Tile.scalar(value, infer_dtype(value))
    except ShapeError as exc:
        raise ExprError(f"cannot use {value!r} as a tile: {exc}") from exc


def _where_op(op: Op) -> str:
    return f"op {op.index:04d} ({op.opcode})"


def _sequence(value: object) -> list[object] | None:
    if isinstance(value, Tile):
        return None if value.ndim == 0 else list(value.data)
    if isinstance(value, (list, tuple)):
        return list(value)
    return None


def _as_int(value: object) -> int:
    if isinstance(value, Tile):
        if value.size != 1:
            raise ShapeError(f"expected a scalar, got a tile of shape {value.shape}")
        value = value.data[0]
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    raise ShapeError(f"expected an integer, got {value!r}")


def _as_shape(value: object) -> tuple[int, ...]:
    if isinstance(value, Tile):
        if value.ndim == 0:
            return (_as_int(value),)
        return tuple(int(item) for item in value.data)
    if isinstance(value, (list, tuple)):
        return tuple(_as_int(item) for item in value)
    return (_as_int(value),)


def _dtype_attr(op: Op, default: str | None) -> str | None:
    text = op.get("dtype")
    if text is None:
        return default
    key = text.strip()
    resolved = DTYPE_ALIASES.get(key) or DTYPE_ALIASES.get(key.rsplit(".", 1)[-1])
    if resolved is None:
        raise IRError(f"{_where_op(op)}: unknown dtype {text!r}")
    return check_dtype(resolved)


def _guard_math(fn: Callable[[float], float], value: object) -> float:
    try:
        return fn(float(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ExprError(f"cannot apply {fn.__name__} to {value!r}: {exc}") from exc


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    return ""


def _c_style_for(text: str) -> tuple[str, str, str, str, int, str] | None:
    parts = [part.strip() for part in text.split(";")]
    if len(parts) != 3 or not all(parts):
        return None
    init, test, step = parts
    m_init = re.match(r"^(?:[A-Za-z_]\w*[\s*&]+)*(?P<var>[A-Za-z_]\w*)\s*=\s*(?P<start>.+)$", init)
    if not m_init:
        return None
    var = m_init.group("var")
    m_test = re.match(rf"^{re.escape(var)}\s*(?P<cmp><=|<|>=|>)\s*(?P<stop>.+)$", test)
    if not m_test:
        return None
    if step in (f"++{var}", f"{var}++"):
        sign, delta = 1, "1"
    elif step in (f"--{var}", f"{var}--"):
        sign, delta = -1, "1"
    else:
        m_step = re.match(rf"^{re.escape(var)}\s*(?P<aug>\+=|-=)\s*(?P<delta>.+)$", step)
        if not m_step:
            return None
        sign = 1 if m_step.group("aug") == "+=" else -1
        delta = m_step.group("delta").strip()
    return var, m_init.group("start").strip(), m_test.group("cmp"), m_test.group("stop").strip(), sign, delta


def _index_value(ctx: "ExecContext", op: Op, buffer: str, index_text: str | None) -> Tile | None:
    if index_text is None:
        return None
    scope = dict(ctx.env)
    scope[buffer] = 0
    value = eval_attr(index_text, scope)
    if isinstance(value, (list, tuple)):
        value = as_tile(value)
    if isinstance(value, Tile):
        return value if value.dtype in INT_DTYPES else value.astype("i64")
    return Tile.scalar(_as_int(value), "i64")


def _mask_value(ctx: "ExecContext", op: Op) -> Tile | None:
    text = op.get("mask")
    if text is None:
        return None
    value = ctx.value(text)
    if value is None:
        return None
    if isinstance(value, Tile):
        return value if value.dtype == "bool" else value.astype("bool")
    return Tile.scalar(bool(value), "bool")


def _where_select(cond: object, on_true: object, on_false: object) -> object:
    if not any(isinstance(item, Tile) for item in (cond, on_true, on_false)):
        return on_true if truthy(cond) else on_false
    mask = as_tile(cond)
    left = as_tile(on_true)
    right = as_tile(on_false)
    shape = broadcast_shapes(broadcast_shapes(mask.shape, left.shape), right.shape)
    dtype = result_dtype(left.dtype, right.dtype)
    mask_data = mask.broadcast_to(shape).data
    left_data = left.broadcast_to(shape).astype(dtype).data
    right_data = right.broadcast_to(shape).astype(dtype).data
    data = [left_data[i] if mask_data[i] else right_data[i] for i in range(len(mask_data))]
    return Tile(shape, dtype, data)


def _exec_binary(ctx: "ExecContext", op: Op) -> object:
    lhs = ctx.value(op.require("lhs"))
    rhs = ctx.value(op.require("rhs"))
    return bind_output(ctx, op, apply_binary(op.opcode, lhs, rhs))


def _exec_compare(ctx: "ExecContext", op: Op) -> object:
    lhs = ctx.value(op.require("lhs"))
    rhs = ctx.value(op.require("rhs"))
    return bind_output(ctx, op, apply_compare(op.opcode, lhs, rhs))


def _exec_logic(ctx: "ExecContext", op: Op) -> object:
    parts = split_args(op.get("args"))
    if not parts:
        parts = [op.require("lhs"), op.require("rhs")]
    values = [ctx.value(part) for part in parts]
    return bind_output(ctx, op, apply_boolean(op.opcode, values))


def _exec_load(ctx: "ExecContext", op: Op) -> object:
    name, index_text = memory_target(op)
    index = _index_value(ctx, op, name, index_text)
    mask = _mask_value(ctx, op)
    other = ctx.value(op.get("other"), 0.0)
    return bind_output(ctx, op, ctx.memory.load(name, index, mask, other))


def _exec_store(ctx: "ExecContext", op: Op) -> object:
    name, index_text = memory_target(op)
    index = _index_value(ctx, op, name, index_text)
    mask = _mask_value(ctx, op)
    value = ctx.value(op.require("value"))
    ctx.memory.store(name, as_tile(value), index, mask)
    return None


def _exec_arange(ctx: "ExecContext", op: Op) -> object:
    start = _as_int(ctx.value(op.get("start"), 0))
    stop = _as_int(ctx.value(op.require("stop")))
    step = _as_int(ctx.value(op.get("step"), 1))
    dtype = _dtype_attr(op, "i32") or "i32"
    return bind_output(ctx, op, Tile.arange(start, stop, step, dtype))


def _exec_program_id(ctx: "ExecContext", op: Op) -> object:
    axis = _as_int(ctx.value(op.get("axis"), 0))
    if not 0 <= axis < len(ctx.program_ids):
        raise IRError(
            f"{_where_op(op)}: axis {axis} is outside a grid of rank {len(ctx.program_ids)}"
        )
    return bind_output(ctx, op, int(ctx.program_ids[axis]))


def _exec_fill(ctx: "ExecContext", op: Op) -> object:
    parts = split_args(op.get("args"))
    if not parts:
        raise IRError(f"{_where_op(op)} needs an 'args' attribute holding a shape")
    shape = _as_shape(ctx.value(parts[0]))
    explicit = op.get("value")
    if explicit is not None:
        value: object = ctx.value(explicit)
    elif len(parts) > 1:
        value = ctx.value(parts[1])
    else:
        value = None
    dtype = _dtype_attr(op, None)
    if value is None:
        value = 0.0
        dtype = dtype or "f32"
    if isinstance(value, Tile):
        value = value.item()
    return bind_output(ctx, op, Tile.full(shape, value, dtype or infer_dtype(value)))


def _exec_select(ctx: "ExecContext", op: Op) -> object:
    cond = ctx.value(op.require("cond"))
    on_true = ctx.value(op.require("true"))
    on_false = ctx.value(op.require("false"))
    return bind_output(ctx, op, _where_select(cond, on_true, on_false))


# Elementwise unary ops. The first group always returns a float; the second
# keeps the operand's dtype so abs and neg of an integer stay integer.
_FLOAT_UNARY: dict[str, Callable[[float], float]] = {
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tanh": math.tanh,
    "sigmoid": lambda x: 1.0 / (1.0 + math.exp(-x)),
    "rsqrt": lambda x: 1.0 / math.sqrt(x),
    "recip": lambda x: 1.0 / x,
}
_KEEP_UNARY: dict[str, Callable[[float], float]] = {
    "abs": abs,
    "neg": lambda x: -x,
    "floor": math.floor,
    "ceil": math.ceil,
    "sign": lambda x: (x > 0) - (x < 0),
    "relu": lambda x: x if x > 0 else 0,
}


def _exec_unary(ctx: "ExecContext", op: Op) -> object:
    keep = op.opcode in _KEEP_UNARY
    fn = _KEEP_UNARY[op.opcode] if keep else _FLOAT_UNARY[op.opcode]
    value = ctx.value(op.require("value"))
    if isinstance(value, Tile):
        dtype = value.dtype if keep else float_dtype(value.dtype)
        return bind_output(ctx, op, value.map(lambda item: _guard_math(fn, item), dtype))
    return bind_output(ctx, op, _guard_math(fn, value))


def _exec_minmax(ctx: "ExecContext", op: Op) -> object:
    pick = max if op.opcode == "max" else min
    lhs = ctx.value(op.require("lhs"))
    rhs = ctx.value(op.require("rhs"))
    if not isinstance(lhs, Tile) and not isinstance(rhs, Tile):
        return bind_output(ctx, op, pick(lhs, rhs))
    left = as_tile(lhs)
    right = as_tile(rhs)
    dtype = result_dtype(left.dtype, right.dtype)
    return bind_output(ctx, op, left.zip_with(right, lambda a, b: pick(a, b), dtype))


def _exec_dot(ctx: "ExecContext", op: Op) -> object:
    lhs = ctx.value(op.require("lhs"))
    rhs = ctx.value(op.require("rhs"))
    return bind_output(ctx, op, as_tile(lhs).matmul(as_tile(rhs)))


REDUCE_COMBINERS: dict[str, tuple[Callable[[object, object], object], object]] = {
    "sum": (lambda acc, item: acc + item, 0),
    "prod": (lambda acc, item: acc * item, 1),
    "max": (lambda acc, item: item if item > acc else acc, None),
    "min": (lambda acc, item: item if item < acc else acc, None),
}


def _reduce_axis(ctx: "ExecContext", op: Op) -> int | None:
    text = op.get("axis")
    if text is None or text.strip() in {"", "all", "none"}:
        return None
    return _as_int(ctx.value(text))


def _kept_shape(shape: tuple[int, ...], axis: int | None) -> tuple[int, ...]:
    """The reduced shape with the folded axis put back as a length-1 dimension."""
    if axis is None:
        return tuple(1 for _ in shape)
    normalized = axis + len(shape) if axis < 0 else axis
    return shape[:normalized] + (1,) + shape[normalized + 1 :]


def _exec_reduce(ctx: "ExecContext", op: Op) -> object:
    tile = as_tile(ctx.value(op.require("value")))
    how = (op.get("op") or "sum").strip()
    combiner = REDUCE_COMBINERS.get(how)
    if combiner is None:
        known = ", ".join(sorted(REDUCE_COMBINERS))
        raise UnsupportedOpcode(f"{_where_op(op)}: unknown reduce op {how!r}; supported: {known}")
    fn, init = combiner
    axis = _reduce_axis(ctx, op)
    result = tile.reduce(fn, axis, init)
    if truthy(ctx.value(op.get("keepdims"), False)):
        result = result.reshape(_kept_shape(tile.shape, axis))
    return bind_output(ctx, op, result)


def _exec_transpose(ctx: "ExecContext", op: Op) -> object:
    return bind_output(ctx, op, as_tile(ctx.value(op.require("value"))).transpose())


def _shape_attr(ctx: "ExecContext", op: Op) -> object:
    """The 'shape' attribute, accepting either 64x64 or a list expression."""
    text = op.require("shape")
    if "x" in text and not text.strip().startswith("["):
        return tuple(int(part) for part in text.strip().split("x") if part)
    return _as_shape(ctx.value(text))


def _exec_reshape(ctx: "ExecContext", op: Op) -> object:
    tile = as_tile(ctx.value(op.require("value")))
    return bind_output(ctx, op, tile.reshape(_shape_attr(ctx, op)))


def _exec_broadcast(ctx: "ExecContext", op: Op) -> object:
    tile = as_tile(ctx.value(op.require("value")))
    return bind_output(ctx, op, tile.broadcast_to(_shape_attr(ctx, op)))


def _exec_mma(ctx: "ExecContext", op: Op) -> object:
    """Fused tile multiply-accumulate: acc + (lhs @ rhs)."""
    lhs = as_tile(ctx.value(op.require("lhs")))
    rhs = as_tile(ctx.value(op.require("rhs")))
    product = lhs.matmul(rhs)
    acc_text = op.get("acc")
    if acc_text is None:
        return bind_output(ctx, op, product)
    acc = as_tile(ctx.value(acc_text))
    dtype = result_dtype(acc.dtype, product.dtype)
    return bind_output(ctx, op, acc.zip_with(product, lambda a, b: a + b, dtype))


def _literal_ints(text: str) -> tuple[int, ...]:
    """A comma list of integer literals, e.g. tile="128,64"."""
    return tuple(int(part) for part in split_list(text))


def _evaluated_ints(ctx: "ExecContext", text: str) -> tuple[int, ...]:
    """A comma list of integer expressions evaluated against the env, e.g. "K,M"."""
    return tuple(_as_int(ctx.value(part)) for part in split_list(text))


def _exec_tensor_view(ctx: "ExecContext", op: Op) -> object:
    view = TensorView(
        op.require("buf"),
        _evaluated_ints(ctx, op.require("shape")),
        _evaluated_ints(ctx, op.require("strides")),
    )
    return bind_output(ctx, op, view)


def _exec_partition_view(ctx: "ExecContext", op: Op) -> object:
    tensor = ctx.value(op.require("view"))
    if not isinstance(tensor, TensorView):
        raise IRError(f"{_where_op(op)}: partition_view needs a tensor view, got {tensor!r}")
    view = PartitionView(tensor, _literal_ints(op.require("tile")), _literal_ints(op.require("dim_map")))
    return bind_output(ctx, op, view)


def _exec_index_space(ctx: "ExecContext", op: Op) -> object:
    view = ctx.value(op.require("view"))
    if not isinstance(view, PartitionView):
        raise IRError(f"{_where_op(op)}: index_space needs a partition view, got {view!r}")
    return bind_output(ctx, op, [int(count) for count in view.index_space()])


def _partition(ctx: "ExecContext", op: Op) -> PartitionView:
    view = ctx.value(op.require("view"))
    if not isinstance(view, PartitionView):
        raise IRError(f"{_where_op(op)}: expected a partition view, got {view!r}")
    return view


def _exec_load_view(ctx: "ExecContext", op: Op) -> object:
    view = _partition(ctx, op)
    index = view.index_tile(_evaluated_ints(ctx, op.require("index")))
    return bind_output(ctx, op, ctx.memory.load(view.tensor.buffer, index))


def _exec_store_view(ctx: "ExecContext", op: Op) -> object:
    view = _partition(ctx, op)
    index = view.index_tile(_evaluated_ints(ctx, op.require("index")))
    ctx.memory.store(view.tensor.buffer, as_tile(ctx.value(op.require("value"))), index)
    return None


def _effects_tensor_view(op: Op) -> Effects:
    return Effects(read_names(op, "shape", "strides"), output_names(op), [], [])


def _effects_load_view(op: Op) -> Effects:
    buf = op.get("buf")
    return Effects(read_names(op, "view", "index"), output_names(op), [buf] if buf else [], [])


def _effects_store_view(op: Op) -> Effects:
    buf = op.get("buf")
    return Effects(read_names(op, "view", "value", "index"), [], [], [buf] if buf else [])


def _exec_assign(ctx: "ExecContext", op: Op) -> object:
    text = op.get("value")
    value = None if text is None else ctx.value(text)
    return bind_output(ctx, op, value)


def _exec_kernel(ctx: "ExecContext", op: Op) -> object:
    return None


def _exec_return(ctx: "ExecContext", op: Op) -> object:
    text = op.get("value")
    return None if text is None else ctx.value(text)


def _exec_marker(ctx: "ExecContext", op: Op) -> object:
    return None


def _exec_cond(ctx: "ExecContext", op: Op) -> object:
    return truthy(ctx.value(op.require("cond")))


def _exec_for(ctx: "ExecContext", op: Op) -> object:
    text = (op.get("iter") or "").strip()
    if not text:
        raise IRError(f"{_where_op(op)} needs an 'iter' attribute")
    plan = _c_style_for(text)
    if plan is not None:
        _, start_text, comparison, stop_text, sign, delta_text = plan
        start = _as_int(ctx.value(start_text))
        stop = _as_int(ctx.value(stop_text))
        step = sign * _as_int(ctx.value(delta_text))
        if step == 0:
            raise IRError(f"{_where_op(op)}: loop step is zero")
        if comparison == "<=":
            stop += 1
        elif comparison == ">=":
            stop -= 1
        return list(range(start, stop, step))
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ExprError(f"cannot parse iter {text!r}: {exc.msg}") from exc
    node = tree.body
    if isinstance(node, ast.Call) and _dotted(node.func).rsplit(".", 1)[-1] == "range":
        if node.keywords or not 1 <= len(node.args) <= 3:
            raise IRError(f"{_where_op(op)}: range takes 1 to 3 positional arguments")
        bounds = [_as_int(ctx.value(ast.unparse(arg))) for arg in node.args]
        if len(bounds) == 3 and bounds[2] == 0:
            raise IRError(f"{_where_op(op)}: loop step is zero in iter {text!r}")
        return list(range(*bounds))
    value = ctx.value(text)
    if isinstance(value, Tile):
        return list(value.data)
    if isinstance(value, (list, tuple)):
        return list(value)
    raise IRError(f"{_where_op(op)}: cannot iterate over {text!r}")


def _call_abs(value: object) -> object:
    if isinstance(value, Tile):
        return value.map(abs)
    return abs(value)  # type: ignore[arg-type]


def _call_sum(value: object) -> object:
    if isinstance(value, Tile):
        return value.reduce(lambda acc, item: acc + item, None, 0)
    return sum(value)  # type: ignore[arg-type]


def _call_reshape(value: object, shape: object) -> object:
    return as_tile(value).reshape(_as_shape(shape))


def _call_transpose(value: object) -> object:
    return as_tile(value).transpose()


def _call_cast(dtype: str) -> Callable[[object], object]:
    def cast(value: object) -> object:
        if isinstance(value, Tile):
            return value.astype(dtype)
        if dtype == "bool":
            return truthy(value)
        return int(value) if dtype in INT_DTYPES else float(value)  # type: ignore[arg-type]

    return cast


CALL_BUILTINS: dict[str, Callable[..., object]] = {
    "abs": _call_abs,
    "sum": _call_sum,
    "reshape": _call_reshape,
    "view": _call_reshape,
    "transpose": _call_transpose,
    "trans": _call_transpose,
    "int": _call_cast("i64"),
    "float": _call_cast("f64"),
    "bool": _call_cast("bool"),
    "len": lambda value: value.shape[0] if isinstance(value, Tile) else len(value),
    "range": lambda *bounds: list(range(*bounds)),
}


def _exec_call(ctx: "ExecContext", op: Op) -> object:
    callee = op.get("callee")
    if callee is None:
        raise UnsupportedOpcode(
            f"{_where_op(op)}: statement {op.get('expr')!r} has no interpreter semantics"
        )
    key = callee.strip()
    fn = CALL_BUILTINS.get(key) or CALL_BUILTINS.get(key.rsplit(".", 1)[-1])
    if fn is None:
        known = ", ".join(sorted(CALL_BUILTINS))
        raise UnsupportedOpcode(f"{_where_op(op)}: unknown callee {callee!r}; supported: {known}")
    args = [ctx.value(part) for part in split_args(op.get("args"))]
    try:
        result = fn(*args)
    except (TypeError, ValueError, IndexError) as exc:
        raise ExprError(f"{_where_op(op)}: call to {callee!r} failed: {exc}") from exc
    return bind_output(ctx, op, result)


def _effects_value(*keys: str, barrier: bool = False) -> Callable[[Op], Effects]:
    def build(op: Op) -> Effects:
        return Effects(read_names(op, *keys), output_names(op), [], [], barrier)

    return build


def _effects_logic(op: Op) -> Effects:
    return Effects(read_names(op, "args", "lhs", "rhs"), output_names(op), [], [])


def _effects_load(op: Op) -> Effects:
    buffer, index_text = memory_target(op)
    reads = [name for name in _names_of(index_text) if name != buffer]
    for name in read_names(op, "mask", "other"):
        if name not in reads and name != buffer:
            reads.append(name)
    return Effects(reads, output_names(op), [buffer], [])


def _effects_store(op: Op) -> Effects:
    buffer, index_text = memory_target(op)
    reads = [name for name in _names_of(index_text) if name != buffer]
    for name in read_names(op, "value", "mask"):
        if name not in reads and name != buffer:
            reads.append(name)
    return Effects(reads, [], [], [buffer])


def _effects_for(op: Op) -> Effects:
    targets = loop_targets(op)
    plan = _c_style_for(op.get("iter") or "")
    if plan is None:
        found = read_names(op, "iter")
    else:
        found = []
        for text in (plan[1], plan[3], plan[5]):
            for name in parse_names(text):
                if name not in found:
                    found.append(name)
    reads = [name for name in found if name not in targets]
    return Effects(reads, targets, [], [], True)


def _effects_kernel(op: Op) -> Effects:
    """A kernel op declares a signature; it reads nothing, writes nothing, orders nothing."""
    return Effects([], [], [], [], False)


def _names_of(text: str | None) -> list[str]:
    return parse_names(text) if text else []


def _spec(
    opcode: str,
    kind: str,
    latency: int,
    effects: Callable[[Op], Effects],
    execute: Callable[["ExecContext", Op], object],
) -> tuple[str, OpSpec]:
    return opcode, OpSpec(opcode, kind, latency, effects, execute)


OPCODES: dict[str, OpSpec] = dict(
    [
        *(
            _spec(name, "arith", 1, _effects_value("lhs", "rhs"), _exec_binary)
            for name in _ARITH_OPS
        ),
        *(
            _spec(name, "compare", 1, _effects_value("lhs", "rhs"), _exec_compare)
            for name in _COMPARE_OPS
        ),
        *(_spec(name, "logic", 1, _effects_logic, _exec_logic) for name in _LOGIC_OPS),
        _spec("load", "memory", 4, _effects_load, _exec_load),
        _spec("store", "memory", 4, _effects_store, _exec_store),
        _spec("arange", "memory", 2, _effects_value("start", "stop", "step"), _exec_arange),
        _spec("program_id", "memory", 1, _effects_value("axis"), _exec_program_id),
        _spec("fill", "memory", 2, _effects_value("args", "value"), _exec_fill),
        _spec("dot", "math", 6, _effects_value("lhs", "rhs"), _exec_dot),
        _spec(
            "reduce",
            "math",
            4,
            _effects_value("value", "axis", "keepdims"),
            _exec_reduce,
        ),
        _spec("transpose", "math", 3, _effects_value("value"), _exec_transpose),
        _spec("reshape", "math", 1, _effects_value("value", "shape"), _exec_reshape),
        _spec("broadcast", "math", 1, _effects_value("value", "shape"), _exec_broadcast),
        _spec("mma", "math", 6, _effects_value("lhs", "rhs", "acc"), _exec_mma),
        _spec("tensor_view", "memory", 1, _effects_tensor_view, _exec_tensor_view),
        _spec("partition_view", "memory", 1, _effects_value("view"), _exec_partition_view),
        _spec("index_space", "memory", 1, _effects_value("view"), _exec_index_space),
        _spec("load_view", "memory", 4, _effects_load_view, _exec_load_view),
        _spec("store_view", "memory", 4, _effects_store_view, _exec_store_view),
        *(_spec(name, "math", 3, _effects_value("value"), _exec_unary) for name in _FLOAT_UNARY),
        *(_spec(name, "math", 1, _effects_value("value"), _exec_unary) for name in _KEEP_UNARY),
        _spec("max", "math", 1, _effects_value("lhs", "rhs"), _exec_minmax),
        _spec("min", "math", 1, _effects_value("lhs", "rhs"), _exec_minmax),
        _spec("select", "math", 1, _effects_value("cond", "true", "false"), _exec_select),
        _spec("if", "control", 1, _effects_value("cond", barrier=True), _exec_cond),
        _spec("else", "control", 0, _effects_value(barrier=True), _exec_marker),
        _spec("endif", "control", 0, _effects_value(barrier=True), _exec_marker),
        _spec("for", "control", 1, _effects_for, _exec_for),
        _spec("endfor", "control", 0, _effects_value(barrier=True), _exec_marker),
        _spec("while", "control", 1, _effects_value("cond", barrier=True), _exec_cond),
        _spec("endwhile", "control", 0, _effects_value(barrier=True), _exec_marker),
        _spec("kernel", "meta", 0, _effects_kernel, _exec_kernel),
        _spec("return", "meta", 0, _effects_value("value", barrier=True), _exec_return),
        _spec("assign", "meta", 1, _effects_value("value"), _exec_assign),
        _spec("call", "meta", 2, _effects_value("args"), _exec_call),
    ]
)


def opcodes_by_kind(kind: str) -> list[str]:
    """Opcode names belonging to one kind, sorted."""
    return sorted(name for name, spec in OPCODES.items() if spec.kind == kind)


def latency_of(op: Op) -> int:
    """Cost-model latency for one op."""
    return spec_for(op).latency


def touched_buffers(program: Iterable[Op]) -> list[str]:
    """Every buffer name a program loads from or stores to, in first-use order."""
    seen: dict[str, None] = {}
    for op in program:
        effects = effects_of(op)
        for name in (*effects.mem_reads, *effects.mem_writes):
            seen.setdefault(name, None)
    return list(seen)
