"""Check the interpreter against numpy on any tile IR program.

    python npcheck.py examples/gemm.lineir
    python npcheck.py examples/gemm_tile_64x64_fixed.tileir --elems 40000
    python npcheck.py examples/reduction.lineir --rtol 1e-9

This is an INDEPENDENT oracle. The interpreter computes every value with the
hand-written Tile class in tile_interp/values.py: its own broadcasting, matmul,
reduction, gather and scatter. This tool re-executes the same program with numpy
doing all of that instead, then asserts the two agree buffer for buffer. A
disagreement means the hand-written numeric code diverges from numpy, which is
exactly the code most likely to be subtly wrong.

numpy is used ONLY here. The interpreter itself stays dependency-free; this file
is a separate test harness, not part of the package.

The numpy executor mirrors the interpreter's control-flow structure (it reuses
match_blocks, which only pairs brackets and touches no values) but shares none of
its arithmetic. Numbers are held in float64 / int64 so the comparison tests
structure, indexing and matmul order rather than float32 rounding.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import numpy as np

from tile_interp.interpreter import Interpreter, match_blocks
from tile_interp.ir import Op, Program, split_list
from tile_interp.lineir import parse_lineir_file
from tile_interp.memory import Memory
from tile_interp.semantics import memory_target, split_args
from tile_interp.values import Tile

try:
    from tile_interp.cuda_tile import translate_cuda_tile_file
except Exception:  # pragma: no cover - translator is optional
    translate_cuda_tile_file = None

try:
    from reference.kernels import example, example_names
except Exception:  # pragma: no cover - registry is optional
    example = None
    example_names = lambda: []


# --------------------------------------------------------------- numpy dtypes

def np_dtype(label: str) -> np.dtype:
    """Map an IR dtype label onto the numpy type that matches the interpreter's math."""
    if label in ("f32", "f64"):
        return np.dtype(np.float64)
    if label in ("i32", "i64"):
        return np.dtype(np.int64)
    if label == "bool":
        return np.dtype(np.bool_)
    return np.dtype(np.float64)


def tile_to_np(tile: Tile) -> np.ndarray:
    return np.array(tile.data, dtype=np_dtype(tile.dtype)).reshape(tile.shape)


# ------------------------------------------------------ safe expression eval

_ALLOWED = (
    ast.Expression, ast.Name, ast.Load, ast.Constant, ast.UnaryOp, ast.USub,
    ast.UAdd, ast.Not, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.FloorDiv, ast.Mod, ast.Pow, ast.BoolOp, ast.And, ast.Or, ast.Compare,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.Subscript,
    ast.Index, ast.Slice, ast.Tuple, ast.List,
)
_LITERALS = {"true": True, "false": False, "True": True, "False": False}


def np_eval(text: str, env: dict[str, object]) -> object:
    """Evaluate an attribute expression against a numpy environment."""
    tree = ast.parse(text.strip(), mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ValueError(f"disallowed expression node {type(node).__name__} in {text!r}")
    return _eval(tree.body, env)


def _eval(node: ast.AST, env: dict[str, object]) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in _LITERALS:
            return _LITERALS[node.id]
        if node.id not in env:
            raise NameError(f"name {node.id!r} is not defined")
        return env[node.id]
    if isinstance(node, ast.UnaryOp):
        operand = _eval(node.operand, env)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        return np.logical_not(operand)
    if isinstance(node, ast.BinOp):
        left, right = _eval(node.left, env), _eval(node.right, env)
        return _BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.BoolOp):
        values = [_eval(v, env) for v in node.values]
        out = values[0]
        for nxt in values[1:]:
            out = np.logical_and(out, nxt) if isinstance(node.op, ast.And) else np.logical_or(out, nxt)
        return out
    if isinstance(node, ast.Compare):
        left = _eval(node.left, env)
        result = None
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval(comparator, env)
            step = _CMPS[type(op)](left, right)
            result = step if result is None else np.logical_and(result, step)
            left = right
        return result
    if isinstance(node, ast.Subscript):
        return _eval(node.value, env)[_eval_slice(node.slice, env)]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_eval(elt, env) for elt in node.elts]
    raise ValueError(f"unhandled expression node {type(node).__name__}")


def _eval_slice(node: ast.AST, env: dict[str, object]) -> object:
    if isinstance(node, ast.Slice):
        lo = _eval(node.lower, env) if node.lower else None
        hi = _eval(node.upper, env) if node.upper else None
        st = _eval(node.step, env) if node.step else None
        return slice(lo, hi, st)
    return _eval(node, env)


_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: np.asarray(a, dtype=np.float64) / np.asarray(b, dtype=np.float64),
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}
_CMPS = {
    ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b,
    ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b,
}


# --------------------------------------------------------------- numpy memory

class NpMemory:
    """A flat numpy buffer store with the same gather / scatter contract as memory.py."""

    def __init__(self) -> None:
        self.buffers: dict[str, np.ndarray] = {}

    def declare(self, name: str, array: np.ndarray) -> None:
        self.buffers[name] = np.array(array)

    def load(self, name: str, index: np.ndarray | None,
             mask: np.ndarray | None, other: object) -> np.ndarray:
        buf = self.buffers[name]
        if index is None:
            return np.array(buf)
        flat = buf.reshape(-1)
        idx = np.asarray(index).astype(np.int64)
        if mask is not None:
            # masked-off lanes take `other` and must never index memory: the
            # interpreter checks the mask before touching the buffer, so a
            # masked-off lane may legitimately hold an out-of-range index.
            m = np.broadcast_to(np.asarray(mask), idx.shape)
            safe = np.where(m, idx, 0)
            gathered = flat[safe.reshape(-1)].reshape(idx.shape).astype(buf.dtype)
            fill = np.broadcast_to(np.asarray(other), idx.shape).astype(buf.dtype)
            return np.where(m, gathered, fill)
        return flat[idx.reshape(-1)].reshape(idx.shape).astype(buf.dtype)

    def store(self, name: str, value: np.ndarray,
              index: np.ndarray | None, mask: np.ndarray | None) -> None:
        buf = self.buffers[name]
        if index is None:
            buf[...] = np.broadcast_to(np.asarray(value), buf.shape).astype(buf.dtype)
            return
        flat = buf.reshape(-1)
        idx = np.asarray(index).astype(np.int64).reshape(-1)
        vals = np.broadcast_to(np.asarray(value), np.asarray(index).shape).reshape(-1)
        if mask is not None:
            m = np.broadcast_to(np.asarray(mask), np.asarray(index).shape).reshape(-1)
            for position, keep in enumerate(m):
                if keep:
                    flat[idx[position]] = vals[position]
        else:
            flat[idx] = vals.astype(buf.dtype)
        buf[...] = flat.reshape(buf.shape)

    def snapshot(self) -> dict[str, np.ndarray]:
        return {name: np.array(buf) for name, buf in self.buffers.items()}


# ------------------------------------------------------------ numpy executor

def _shape_from_attr(op: Op, env: dict[str, object]) -> tuple[int, ...]:
    text = op.require("shape")
    if "x" in text and not text.strip().startswith("["):
        return tuple(int(part) for part in text.strip().split("x") if part)
    value = np_eval(text, env)
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return (int(value),)


def _reduce_axis(op: Op, env: dict[str, object]) -> int | None:
    text = op.get("axis")
    if text is None or text.strip() in ("", "all", "none"):
        return None
    return int(np_eval(text, env))


def _index_from_ptr(op: Op, buffer: str, index_text: str | None,
                    env: dict[str, object]) -> np.ndarray | None:
    if index_text is None:
        return None
    scope = dict(env)
    scope[buffer] = 0  # same trick as the interpreter: base pointer contributes 0
    return np.asarray(np_eval(index_text, scope)).astype(np.int64)


def _literal_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in split_list(text))


def _evaluated_ints(executor, text: str) -> tuple[int, ...]:
    return tuple(int(np_eval(part, executor.env)) for part in split_list(text))


def _view_index(pv: dict, block: tuple[int, ...]) -> np.ndarray:
    """Flat buffer indices for one partition-view tile, computed with numpy."""
    tile, dim_map = pv["tile"], pv["dim_map"]
    strides, shape = pv["tensor"]["strides"], pv["tensor"]["shape"]
    grids = np.indices(tile)  # grids[axis] is the coordinate along tile axis
    tensor_coord = [np.zeros(tile, dtype=np.int64) for _ in shape]
    for axis in range(len(tile)):
        tensor_coord[dim_map[axis]] = block[axis] * tile[axis] + grids[axis]
    flat = np.zeros(tile, dtype=np.int64)
    for t in range(len(shape)):
        flat = flat + tensor_coord[t] * strides[t]
    return flat


_REDUCERS = {
    "sum": np.sum, "prod": np.prod, "max": np.max, "min": np.min,
}

_NP_UNARY = {
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "sin": np.sin, "cos": np.cos,
    "tanh": np.tanh, "sigmoid": lambda a: 1.0 / (1.0 + np.exp(-a)),
    "rsqrt": lambda a: 1.0 / np.sqrt(a), "recip": lambda a: 1.0 / a,
    "abs": np.abs, "neg": lambda a: -a, "floor": np.floor, "ceil": np.ceil,
    "sign": np.sign, "relu": lambda a: np.maximum(a, 0.0),
}


class NumpyExecutor:
    """Runs a Program with numpy, mirroring the interpreter's control flow."""

    def __init__(self, program: Program, memory: NpMemory,
                 program_ids: tuple[int, ...],
                 env: dict[str, object] | None = None) -> None:
        self.program = program
        self.memory = memory
        self.pids = program_ids
        self.env: dict[str, object] = dict(env) if env else {}

    def run(self) -> NpMemory:
        blocks = match_blocks(self.program)
        ops = self.program.ops
        for_state: dict[int, tuple[list[int], int]] = {}
        position = 0
        steps = 0
        while position < len(ops):
            steps += 1
            if steps > 1_000_000:
                raise RuntimeError("numpy executor exceeded the step budget")
            op = ops[position]
            opcode = op.opcode
            if opcode == "if":
                if self._truthy(np_eval(op.require("cond"), self.env)):
                    position += 1
                else:
                    alt = blocks.alt.get(position)
                    position = (alt + 1) if alt is not None else blocks.close[position]
            elif opcode == "else":
                position = blocks.alt_end[position]
            elif opcode == "endif":
                position += 1
            elif opcode == "while":
                if self._truthy(np_eval(op.require("cond"), self.env)):
                    position += 1
                else:
                    position = blocks.close[position] + 1
            elif opcode == "for":
                state = for_state.get(position)
                if state is None:
                    values, cursor = self._for_values(op), 0
                else:
                    values, cursor = state[0], state[1] + 1
                if cursor >= len(values):
                    for_state.pop(position, None)
                    position = blocks.close[position] + 1
                else:
                    for_state[position] = (values, cursor)
                    self.env[op.require("target").strip()] = values[cursor]
                    position += 1
            elif opcode in ("endfor", "endwhile"):
                position = blocks.open[position]
            elif opcode == "return":
                break
            else:
                self._exec(op)
                position += 1
        return self.memory

    def _exec(self, op: Op) -> None:
        unary = _NP_UNARY.get(op.opcode)
        if unary is not None:
            self._bind(op, unary(np.asarray(self._val(op, "value"), np.float64)))
            return
        handler = getattr(self, f"_op_{op.opcode}", None)
        if handler is None:
            raise ValueError(f"numpy executor has no rule for opcode {op.opcode!r}")
        result = handler(op)
        if result is not None:
            self._bind(op, result)

    def _bind(self, op: Op, value: object) -> None:
        text = op.get("out")
        if not text:
            return
        names = split_list(text.strip().strip("()[]"))  # a tuple target unpacks
        if len(names) == 1:
            self.env[names[0]] = value
        else:
            for name, item in zip(names, value):
                self.env[name] = item

    def _val(self, op: Op, key: str, default: object = None) -> object:
        text = op.get(key)
        if text is None or not text.strip():
            return default
        return np_eval(text, self.env)

    # -- opcode rules -----------------------------------------------------

    def _op_kernel(self, op: Op) -> None:
        return None

    def _op_assign(self, op: Op) -> object:
        return self._val(op, "value")

    def _op_program_id(self, op: Op) -> object:
        axis = int(self._val(op, "axis", 0))
        return int(self.pids[axis])

    def _op_arange(self, op: Op) -> object:
        start = int(self._val(op, "start", 0))
        stop = int(self._val(op, "stop"))
        step = int(self._val(op, "step", 1))
        return np.arange(start, stop, step, dtype=np.int64)

    def _op_fill(self, op: Op) -> object:
        parts = split_args(op.get("args"))  # respects brackets in "[64,64]"
        shape_value = np_eval(parts[0], self.env) if parts else ()
        if isinstance(shape_value, (list, tuple)):
            shape = tuple(int(v) for v in shape_value)
        else:
            shape = (int(shape_value),)
        value = self._val(op, "value", None)
        if value is None and len(parts) > 1:
            value = np_eval(parts[1], self.env)
        return np.full(shape, 0.0 if value is None else value, dtype=np.float64)

    def _op_iota(self, op: Op) -> object:
        return None

    def _op_return(self, op: Op) -> object:
        return None

    def _op_call(self, op: Op) -> object:
        callee = (op.get("callee") or "").strip().rsplit(".", 1)[-1]
        args = [np_eval(a, self.env) for a in split_list(op.get("args"))]
        if callee == "sum":
            return np.sum(np.asarray(args[0]))
        if callee == "abs":
            return np.abs(np.asarray(args[0]))
        if callee in ("transpose", "trans"):
            return np.transpose(np.asarray(args[0]))
        if callee in ("reshape", "view"):
            return np.reshape(np.asarray(args[0]), tuple(int(v) for v in args[1]))
        if callee in ("int", "float", "bool"):
            cast = {"int": np.int64, "float": np.float64, "bool": np.bool_}[callee]
            return np.asarray(args[0]).astype(cast)
        raise ValueError(f"numpy executor has no rule for call callee {callee!r}")

    def _binary(self, op: Op, fn) -> object:
        return fn(self._val(op, "lhs"), self._val(op, "rhs"))

    def _op_add(self, op): return self._binary(op, lambda a, b: a + b)
    def _op_sub(self, op): return self._binary(op, lambda a, b: a - b)
    def _op_mul(self, op): return self._binary(op, lambda a, b: a * b)
    def _op_div(self, op):
        return np.asarray(self._val(op, "lhs"), np.float64) / np.asarray(self._val(op, "rhs"), np.float64)
    def _op_floordiv(self, op): return self._binary(op, lambda a, b: a // b)
    def _op_mod(self, op): return self._binary(op, lambda a, b: a % b)
    def _op_pow(self, op): return self._binary(op, lambda a, b: a ** b)
    def _op_lt(self, op): return self._binary(op, lambda a, b: a < b)
    def _op_le(self, op): return self._binary(op, lambda a, b: a <= b)
    def _op_gt(self, op): return self._binary(op, lambda a, b: a > b)
    def _op_ge(self, op): return self._binary(op, lambda a, b: a >= b)
    def _op_eq(self, op): return self._binary(op, lambda a, b: a == b)
    def _op_ne(self, op): return self._binary(op, lambda a, b: a != b)
    def _op_max(self, op): return np.maximum(self._val(op, "lhs"), self._val(op, "rhs"))
    def _op_min(self, op): return np.minimum(self._val(op, "lhs"), self._val(op, "rhs"))

    def _op_and(self, op): return self._logic(op, np.logical_and)
    def _op_or(self, op): return self._logic(op, np.logical_or)

    def _logic(self, op: Op, fn) -> object:
        args = split_list(op.get("args"))
        if args:
            values = [np_eval(a, self.env) for a in args]
        else:
            values = [self._val(op, "lhs"), self._val(op, "rhs")]
        out = values[0]
        for nxt in values[1:]:
            out = fn(out, nxt)
        return out

    # exp, sqrt, and the other elementwise unary ops share one numpy dispatch;
    # see _NP_UNARY and _exec.

    def _op_dot(self, op): return np.matmul(self._val(op, "lhs"), self._val(op, "rhs"))

    def _op_mma(self, op) -> object:
        product = np.matmul(self._val(op, "lhs"), self._val(op, "rhs"))
        acc = self._val(op, "acc")
        return product if acc is None else acc + product

    def _op_select(self, op) -> object:
        return np.where(self._val(op, "cond"), self._val(op, "true"), self._val(op, "false"))

    def _op_reshape(self, op) -> object:
        return np.reshape(np.asarray(self._val(op, "value")), _shape_from_attr(op, self.env))

    def _op_broadcast(self, op) -> object:
        return np.broadcast_to(np.asarray(self._val(op, "value")), _shape_from_attr(op, self.env))

    def _op_transpose(self, op) -> object:
        return np.transpose(np.asarray(self._val(op, "value")))

    def _op_reduce(self, op) -> object:
        how = (op.get("op") or "sum").strip()
        fn = _REDUCERS[how]
        axis = _reduce_axis(op, self.env)
        keep = bool(self._val(op, "keepdims", False))
        return fn(np.asarray(self._val(op, "value")), axis=axis, keepdims=keep)

    def _op_load(self, op) -> object:
        name, index_text = memory_target(op)
        index = _index_from_ptr(op, name, index_text, self.env)
        mask = self._val(op, "mask")
        other = self._val(op, "other", 0.0)
        return self.memory.load(name, index, mask, other)

    def _op_store(self, op) -> None:
        name, index_text = memory_target(op)
        index = _index_from_ptr(op, name, index_text, self.env)
        mask = self._val(op, "mask")
        self.memory.store(name, np.asarray(self._val(op, "value")), index, mask)
        return None

    # -- structured tensor/partition views --------------------------------

    def _op_tensor_view(self, op) -> object:
        return {
            "buffer": op.require("buf"),
            "shape": _evaluated_ints(self, op.require("shape")),
            "strides": _evaluated_ints(self, op.require("strides")),
        }

    def _op_partition_view(self, op) -> object:
        return {
            "tensor": self._val(op, "view"),
            "tile": _literal_ints(op.require("tile")),
            "dim_map": _literal_ints(op.require("dim_map")),
        }

    def _op_index_space(self, op) -> object:
        pv = self._val(op, "view")
        shape, tile, dim_map = pv["tensor"]["shape"], pv["tile"], pv["dim_map"]
        return [int(-(-shape[dim_map[a]] // tile[a])) for a in range(len(tile))]

    def _op_load_view(self, op) -> object:
        pv = self._val(op, "view")
        index = _view_index(pv, _evaluated_ints(self, op.require("index")))
        return self.memory.load(pv["tensor"]["buffer"], index, None, 0.0)

    def _op_store_view(self, op) -> None:
        pv = self._val(op, "view")
        index = _view_index(pv, _evaluated_ints(self, op.require("index")))
        self.memory.store(pv["tensor"]["buffer"], np.asarray(self._val(op, "value")), index, None)
        return None

    # -- helpers ----------------------------------------------------------

    def _for_values(self, op: Op) -> list[int]:
        text = (op.get("iter") or "").strip()
        tree = ast.parse(text, mode="eval").body
        if isinstance(tree, ast.Call) and getattr(tree.func, "id", "") == "range":
            bounds = [int(np_eval(ast.unparse(a), self.env)) for a in tree.args]
            return list(range(*bounds))
        value = np_eval(text, self.env)
        return [int(v) for v in np.asarray(value).reshape(-1)]

    @staticmethod
    def _truthy(value: object) -> bool:
        array = np.asarray(value)
        return bool(array) if array.size == 1 else bool(array.all())


# ------------------------------------------------------------------- driver

def load_program(path: Path) -> Program:
    if path.suffix == ".tileir":
        if translate_cuda_tile_file is None:
            raise SystemExit("cuda_tile translator is unavailable")
        return translate_cuda_tile_file(path)
    return parse_lineir_file(path)


def registry_seed(stem: str, required: set[str]) -> dict[str, Tile] | None:
    """The registered seeds for this stem, but only if they cover its buffers.

    A .tileir file can share a stem with a line-format example while naming its
    buffers differently, so a stem match alone is not enough.
    """
    if example is None or stem not in set(example_names()):
        return None
    seed = example(stem).seed()
    return seed if required <= set(seed) else None


def synthesize_seed(program: Program, elems: int, dim: int = 128) -> dict[str, Tile]:
    """Deterministic inputs for a program with no registered reference.

    Buffers (from either memory model) get 1-D data; scalar parameters default to
    dim; view kernels size buffers to dim*dim so a tile fits.
    """
    inputs: set[str] = set()
    outputs: set[str] = set()
    for op in program.ops:
        if op.opcode == "load":
            inputs.add(memory_target(op)[0])
        elif op.opcode == "store":
            outputs.add(memory_target(op)[0])
        elif op.opcode == "load_view" and op.get("buf"):
            inputs.add(op.get("buf"))
        elif op.opcode == "store_view" and op.get("buf"):
            outputs.add(op.get("buf"))
    buffers = {name for name in inputs | outputs if name}
    uses_views = any(op.opcode == "tensor_view" for op in program.ops)
    size = max(elems, dim * dim) if uses_views else elems
    seed: dict[str, Tile] = {}
    for name in sorted(buffers):
        data = [float(i % 7) for i in range(size)] if name in inputs else [0.0] * size
        seed[name] = Tile.from_flat(data, (size,), "f32")
    for param in program.params():
        if param not in buffers and param not in seed:
            seed[param] = Tile.scalar(dim, "i32")
    return seed


def run_checks(path: Path, elems: int, pids: tuple[int, ...],
               rtol: float, atol: float) -> int:
    program = load_program(path)
    required = {memory_target(op)[0] for op in program.ops if op.opcode in ("load", "store")}
    seed = registry_seed(path.stem, required)
    origin = "reference registry"
    if seed is None:
        seed = synthesize_seed(program, elems)
        origin = f"synthesized ({elems} elems/buffer)"

    # interpreter side
    interp_mem = Memory()
    for name, tile in seed.items():
        interp_mem.declare(name, tile)
    result = Interpreter(program, memory=interp_mem, grid=pids or (1,),
                         program_ids=pids).run(**seed)
    interp_out = result.memory.snapshot()

    # numpy side, same seed. The interpreter binds every seed into the env AND
    # declares Tiles as buffers, so mirror both: scalars reach ops like
    # `mul rhs=alpha` through the env, tiles are loaded through memory.
    np_mem = NpMemory()
    np_env: dict[str, object] = {}
    for name, tile in seed.items():
        array = tile_to_np(tile)
        np_mem.declare(name, array)
        np_env[name] = array.reshape(()) if array.size == 1 else array
    np_out = NumpyExecutor(program, np_mem, pids, env=np_env).run()

    print(f"program : {path.name}  ({len(program)} ops)")
    print(f"inputs  : {origin}")
    print(f"tolerance rtol={rtol:g} atol={atol:g}")
    print()

    names = sorted(interp_out)
    failures = 0
    for name in names:
        got = tile_to_np(interp_out[name])
        want = np_out.buffers.get(name)
        if want is None:
            continue
        want = want.reshape(got.shape) if want.shape != got.shape else want
        ok = np.allclose(got, want, rtol=rtol, atol=atol)
        diff = float(np.max(np.abs(got.astype(np.float64) - want.astype(np.float64)))) if got.size else 0.0
        flag = "MATCH" if ok else "DIFFER"
        print(f"  [{flag:6}] {name:10} shape {str(got.shape):10} max abs diff = {diff:.3e}")
        if not ok:
            failures += 1

    print()
    if failures:
        print(f"FAIL  {failures} buffer(s) disagree with numpy")
        return 1
    print(f"PASS  all {len(names)} buffers match numpy")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Check the interpreter against numpy.")
    parser.add_argument("program", help="a .lineir or .tileir file")
    parser.add_argument("--elems", type=int, default=20000,
                        help="buffer size when inputs are synthesized")
    parser.add_argument("--pid", default="0,0,0", help="program ids, comma separated")
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args(argv[1:])

    path = Path(args.program)
    if not path.exists():
        print(f"no such file: {path}", file=sys.stderr)
        return 2
    pids = tuple(int(p) for p in args.pid.split(",") if p.strip())
    try:
        return run_checks(path, args.elems, pids, args.rtol, args.atol)
    except Exception as exc:  # surface the failure with its type
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
