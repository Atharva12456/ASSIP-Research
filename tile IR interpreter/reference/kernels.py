from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from tile_interp.values import Tile

Inputs = dict[str, Tile]
Outputs = dict[str, Tile]
Kernel = Callable[[Inputs], Outputs]

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _like(inputs: Inputs, name: str, values: list[float]) -> Tile:
    """Wrap a flat Python list in a tile shaped and typed like an existing buffer."""
    template = inputs[name]
    return Tile.from_flat(values, template.shape, template.dtype)


def load_compute_store(inputs: Inputs) -> Outputs:
    """C = A + B, then E = C * D, elementwise over flat buffers."""
    a = list(inputs["A"].data)
    b = list(inputs["B"].data)
    d = list(inputs["D"].data)

    c: list[float] = []
    for i in range(len(a)):
        c.append(a[i] + b[i])

    e: list[float] = []
    for i in range(len(c)):
        e.append(c[i] * d[i])

    return {"C": _like(inputs, "C", c), "E": _like(inputs, "E", e)}


def vector_add(inputs: Inputs) -> Outputs:
    """out[i] = x[i] + y[i] for every lane of the block."""
    x = list(inputs["x_ptr"].data)
    y = list(inputs["y_ptr"].data)
    out = list(inputs["out_ptr"].data)

    for i in range(len(out)):
        out[i] = x[i] + y[i]

    return {"out_ptr": _like(inputs, "out_ptr", out)}


def scale_add(inputs: Inputs) -> Outputs:
    """out[i] = alpha * x[i] + y[i], with an integer x and a float y."""
    x = list(inputs["x_ptr"].data)
    y = list(inputs["y_ptr"].data)
    out = list(inputs["out_ptr"].data)
    alpha = inputs["alpha"].item()

    for i in range(len(out)):
        out[i] = x[i] * alpha + y[i]

    return {"out_ptr": _like(inputs, "out_ptr", out)}


def masked_load(inputs: Inputs) -> Outputs:
    """Gather 8 lanes from two 5-element buffers under a bounds mask, add, store under a
    wider mask so the out-of-bounds 'other' values reach memory."""
    src = list(inputs["src"].data)
    alt = list(inputs["alt"].data)
    dst = list(inputs["dst"].data)
    n = inputs["n"].item()

    inbounds: list[bool] = []
    for i in range(len(dst)):
        inbounds.append(i < n)

    left: list[float] = []
    right: list[float] = []
    for i in range(len(dst)):
        if inbounds[i]:
            left.append(src[i])
            right.append(alt[i])
        else:
            left.append(-1.0)
            right.append(0.5)

    total: list[float] = []
    for i in range(len(dst)):
        total.append(left[i] + right[i])

    keep: list[bool] = []
    for i in range(len(dst)):
        keep.append(i < 7)

    for i in range(len(dst)):
        if keep[i]:
            dst[i] = total[i]

    return {"dst": _like(inputs, "dst", dst)}


def reduce_sum(inputs: Inputs) -> Outputs:
    """dst[0] = sqrt(sum(src) + sum(aux))."""
    src = list(inputs["src"].data)
    aux = list(inputs["aux"].data)

    total = 0.0
    for value in src:
        total = total + value

    extra = 0.0
    for value in aux:
        extra = extra + value

    root = math.sqrt(total + extra)
    return {"dst": _like(inputs, "dst", [root])}


def matmul_tile(inputs: Inputs) -> Outputs:
    """C = A @ B for small 2-D tiles, by the textbook triple loop."""
    a = inputs["A"]
    b = inputs["B"]
    rows, inner = a.shape
    _, cols = b.shape

    out = [0.0] * (rows * cols)
    for i in range(rows):
        for j in range(cols):
            total = 0.0
            for k in range(inner):
                total = total + a.data[i * inner + k] * b.data[k * cols + j]
            out[i * cols + j] = total

    return {"C": _like(inputs, "C", out)}


def control_flow(inputs: Inputs) -> Outputs:
    """acc accumulates i*2 below the limit and i*10 above it; dst = src + acc."""
    src = list(inputs["src"].data)
    limit = inputs["limit"].item()

    acc = 0.0
    for i in range(4):
        if i < limit:
            term = i * 2
        else:
            term = i * 10
        acc = acc + term

    dst: list[float] = []
    for value in src:
        dst.append(value + acc)

    return {"dst": _like(inputs, "dst", dst)}


def long_chain(inputs: Inputs) -> Outputs:
    """A wide reduction tree over eight buffers, ending in a select and a store."""
    a0 = list(inputs["A0"].data)
    a1 = list(inputs["A1"].data)
    a2 = list(inputs["A2"].data)
    a3 = list(inputs["A3"].data)
    a4 = list(inputs["A4"].data)
    a5 = list(inputs["A5"].data)
    a6 = list(inputs["A6"].data)
    a7 = list(inputs["A7"].data)
    width = len(a0)

    s0: list[float] = []
    s1: list[float] = []
    s2: list[float] = []
    s3: list[float] = []
    for i in range(width):
        s0.append(a0[i] + a1[i])
        s1.append(a2[i] + a3[i])
        s2.append(a4[i] + a5[i])
        s3.append(a6[i] + a7[i])

    p0: list[float] = []
    p1: list[float] = []
    for i in range(width):
        p0.append(s0[i] * s1[i])
        p1.append(s2[i] * s3[i])

    q: list[float] = []
    for i in range(width):
        q.append(p0[i] + p1[i])

    r0: list[float] = []
    r1: list[float] = []
    for i in range(width):
        r0.append(s0[i] - s1[i])
        r1.append(s2[i] - s3[i])

    r2: list[float] = []
    for i in range(width):
        r2.append(r0[i] * r1[i])

    t: list[float] = []
    for i in range(width):
        t.append(q[i] + r2[i])

    m: list[float] = []
    for i in range(width):
        m.append(max(t[i], 0.0))

    g: list[float] = []
    for i in range(width):
        g.append(math.sqrt(m[i]))

    mask: list[bool] = []
    for i in range(width):
        mask.append(i < 2)

    u: list[float] = []
    for i in range(width):
        if mask[i]:
            u.append(g[i])
        else:
            u.append(t[i])

    w: list[float] = []
    for i in range(width):
        w.append(u[i] + q[i])

    return {
        "T0": _like(inputs, "T0", p0),
        "T1": _like(inputs, "T1", p1),
        "OUT": _like(inputs, "OUT", w),
    }


def _inputs_load_compute_store() -> Inputs:
    return {
        "A": Tile.from_flat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], (8,), "f32"),
        "B": Tile.from_flat([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], (8,), "f32"),
        "D": Tile.from_flat([2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0], (8,), "f32"),
        "C": Tile.zeros((8,), "f32"),
        "E": Tile.zeros((8,), "f32"),
    }


def _inputs_vector_add() -> Inputs:
    return {
        "x_ptr": Tile.from_flat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], (8,), "f32"),
        "y_ptr": Tile.from_flat([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], (8,), "f32"),
        "out_ptr": Tile.zeros((8,), "f32"),
    }


def _inputs_scale_add() -> Inputs:
    return {
        "x_ptr": Tile.from_flat([1, 2, 3, 4, 5, 6, 7, 8], (8,), "i32"),
        "y_ptr": Tile.from_flat([0.5, 0.25, 0.125, 1.0, 2.0, 4.0, 8.0, 16.0], (8,), "f32"),
        "out_ptr": Tile.zeros((8,), "f32"),
        "alpha": Tile.scalar(2.5, "f64"),
    }


def _inputs_masked_load() -> Inputs:
    return {
        "src": Tile.from_flat([1.0, 2.0, 3.0, 4.0, 5.0], (5,), "f32"),
        "alt": Tile.from_flat([10.0, 20.0, 30.0, 40.0, 50.0], (5,), "f32"),
        "dst": Tile.full((8,), 9.0, "f32"),
        "n": Tile.scalar(5, "i32"),
    }


def _inputs_reduce_sum() -> Inputs:
    return {
        "src": Tile.from_flat([float(i + 1) for i in range(16)], (16,), "f32"),
        "aux": Tile.full((16,), 0.5, "f32"),
        "dst": Tile.zeros((1,), "f32"),
    }


def _inputs_matmul_tile() -> Inputs:
    return {
        "A": Tile.from_flat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), "f32"),
        "B": Tile.from_flat([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], (3, 2), "f32"),
        "C": Tile.zeros((2, 2), "f32"),
    }


def _inputs_control_flow() -> Inputs:
    return {
        "src": Tile.from_flat([1.0, 2.0, 3.0, 4.0], (4,), "f32"),
        "dst": Tile.zeros((4,), "f32"),
        "limit": Tile.scalar(2, "i32"),
    }


def _inputs_long_chain() -> Inputs:
    lanes = [
        [5.0, 1.0, 2.0, 4.0],
        [3.0, 3.0, 2.0, 1.0],
        [-1.0, 2.0, 0.0, 3.0],
        [1.0, 1.0, 3.0, 1.0],
        [-0.5, 2.5, 0.5, 1.0],
        [0.5, 1.5, 3.5, 1.0],
        [3.0, 1.0, 2.0, 5.0],
        [2.0, 3.0, 1.0, 1.0],
    ]
    inputs: Inputs = {}
    for slot, values in enumerate(lanes):
        inputs[f"A{slot}"] = Tile.from_flat(values, (4,), "f32")
    inputs["T0"] = Tile.zeros((4,), "f32")
    inputs["T1"] = Tile.zeros((4,), "f32")
    inputs["OUT"] = Tile.zeros((4,), "f32")
    return inputs


def gemm(inputs: Inputs) -> Outputs:
    """D = alpha * (A @ B) + beta * C, by the textbook triple loop."""
    a = inputs["A"].to_nested()
    b = inputs["B"].to_nested()
    c = inputs["C"].to_nested()
    alpha = inputs["alpha"].item()
    beta = inputs["beta"].item()

    rows = len(a)
    inner = len(a[0])
    cols = len(b[0])

    out: list[float] = []
    for i in range(rows):
        for j in range(cols):
            acc = 0.0
            for k in range(inner):
                acc = acc + a[i][k] * b[k][j]
            out.append(alpha * acc + beta * c[i][j])

    return {"D": _like(inputs, "D", out)}


def transpose(inputs: Inputs) -> Outputs:
    """T = A with its axes reversed, then the Gram matrix G = T @ A."""
    a = inputs["A"].to_nested()
    rows = len(a)
    cols = len(a[0])

    t: list[float] = []
    for j in range(cols):
        for i in range(rows):
            t.append(a[i][j])

    g: list[float] = []
    for i in range(cols):
        for j in range(cols):
            acc = 0.0
            for k in range(rows):
                acc = acc + a[k][i] * a[k][j]
            g.append(acc)

    return {"T": _like(inputs, "T", t), "G": _like(inputs, "G", g)}


def reduction(inputs: Inputs) -> Outputs:
    """Row sums, their total, and every row shifted by its own maximum."""
    a = inputs["A"].to_nested()

    rows: list[float] = []
    for row in a:
        acc = 0.0
        for value in row:
            acc = acc + value
        rows.append(acc)

    total = 0.0
    for value in rows:
        total = total + value

    shifted: list[float] = []
    for row in a:
        top = row[0]
        for value in row:
            if value > top:
                top = value
        for value in row:
            shifted.append(value - top)

    return {
        "RowSum": _like(inputs, "RowSum", rows),
        "Total": _like(inputs, "Total", [total]),
        "Norm": _like(inputs, "Norm", shifted),
    }


def _inputs_gemm() -> Inputs:
    return {
        "A": Tile.from_flat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), "f32"),
        "B": Tile.from_flat([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], (3, 2), "f32"),
        "C": Tile.from_flat([1.0, 1.0, 2.0, 2.0], (2, 2), "f32"),
        "D": Tile.zeros((2, 2), "f32"),
        "alpha": Tile.scalar(2.0, "f32"),
        "beta": Tile.scalar(0.5, "f32"),
    }


def _inputs_transpose() -> Inputs:
    return {
        "A": Tile.from_flat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), "f32"),
        "T": Tile.zeros((3, 2), "f32"),
        "G": Tile.zeros((3, 3), "f32"),
    }


def _inputs_reduction() -> Inputs:
    return {
        "A": Tile.from_flat(
            [1.0, 2.0, 3.0, 4.0, 0.5, 0.25, 0.125, 0.125, -1.0, -2.0, 3.0, 0.0],
            (3, 4),
            "f32",
        ),
        "RowSum": Tile.zeros((3,), "f32"),
        "Total": Tile.zeros((1,), "f32"),
        "Norm": Tile.zeros((3, 4), "f32"),
    }


@dataclass(slots=True)
class Example:
    """One example program: its reference kernel, its seed buffers, and its outputs."""

    name: str
    reference: Kernel
    inputs: Callable[[], Inputs]
    outputs: list[str]
    summary: str

    @property
    def path(self) -> Path:
        """Location of the .lineir file for this example."""
        return EXAMPLES_DIR / f"{self.name}.lineir"

    def seed(self) -> Inputs:
        """A fresh set of input tiles."""
        return self.inputs()

    def expected(self) -> Outputs:
        """Reference outputs for the standard seed inputs."""
        return self.reference(self.inputs())


EXAMPLES: dict[str, Example] = {
    example.name: example
    for example in (
        Example(
            "load_compute_store",
            load_compute_store,
            _inputs_load_compute_store,
            ["C", "E"],
            "load A, load B, C = A + B, store C, load D, E = C * D, store E",
        ),
        Example(
            "vector_add",
            vector_add,
            _inputs_vector_add,
            ["out_ptr"],
            "elementwise add over arange offsets",
        ),
        Example(
            "scale_add",
            scale_add,
            _inputs_scale_add,
            ["out_ptr"],
            "fused multiply-add across i32, f32 and f64 operands",
        ),
        Example(
            "masked_load",
            masked_load,
            _inputs_masked_load,
            ["dst"],
            "two masked loads with an 'other' value, then a masked store",
        ),
        Example(
            "reduce_sum",
            reduce_sum,
            _inputs_reduce_sum,
            ["dst"],
            "two independent reductions to scalars, combined and rooted",
        ),
        Example(
            "matmul_tile",
            matmul_tile,
            _inputs_matmul_tile,
            ["C"],
            "2x3 by 3x2 tile dot product",
        ),
        Example(
            "control_flow",
            control_flow,
            _inputs_control_flow,
            ["dst"],
            "a for loop wrapping an if/else, scheduled as barriers",
        ),
        Example(
            "long_chain",
            long_chain,
            _inputs_long_chain,
            ["T0", "T1", "OUT"],
            "30 ops over a wide reduction tree, for an interesting speedup",
        ),
        Example(
            "gemm",
            gemm,
            _inputs_gemm,
            ["D"],
            "D = alpha * (A @ B) + beta * C, with three independent loads",
        ),
        Example(
            "transpose",
            transpose,
            _inputs_transpose,
            ["T", "G"],
            "axis reversal feeding a dot product, for the Gram matrix",
        ),
        Example(
            "reduction",
            reduction,
            _inputs_reduction,
            ["RowSum", "Total", "Norm"],
            "reduce along an axis, reduce to a scalar, and a keepdims broadcast",
        ),
    )
}

KERNELS: dict[str, Kernel] = {name: item.reference for name, item in EXAMPLES.items()}


def example(name: str) -> Example:
    """Look up one example by name."""
    try:
        return EXAMPLES[name]
    except KeyError:
        known = ", ".join(EXAMPLES)
        raise KeyError(f"unknown example {name!r}; known: {known}") from None


def example_names() -> list[str]:
    """Every example name, in declaration order."""
    return list(EXAMPLES)
