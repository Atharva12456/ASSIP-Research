from __future__ import annotations

import ast


BINOP_TO_IR = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.Div: "div",
    ast.FloorDiv: "floordiv",
    ast.Mod: "mod",
    ast.Pow: "pow",
    ast.MatMult: "dot",
}

COMPARE_TO_IR = {
    ast.Lt: "lt",
    ast.LtE: "le",
    ast.Gt: "gt",
    ast.GtE: "ge",
    ast.Eq: "eq",
    ast.NotEq: "ne",
}

BOOL_TO_IR = {
    ast.And: "and",
    ast.Or: "or",
}

CALL_TO_IR = {
    "tl.load": "load",
    "load": "load",
    "load_tile": "load",
    "cutile.load": "load",
    "cuTile.load": "load",
    "tl.store": "store",
    "store": "store",
    "store_tile": "store",
    "cutile.store": "store",
    "cuTile.store": "store",
    "tl.arange": "arange",
    "arange": "arange",
    "tile_range": "arange",
    "range_tile": "arange",
    "tl.program_id": "program_id",
    "program_id": "program_id",
    "pid": "program_id",
    "tl.zeros": "fill",
    "zeros": "fill",
    "tl.full": "fill",
    "full": "fill",
    "tl.dot": "dot",
    "dot": "dot",
    "tl.exp": "exp",
    "exp": "exp",
    "tl.sqrt": "sqrt",
    "sqrt": "sqrt",
    "tl.maximum": "max",
    "maximum": "max",
    "max": "max",
    "tl.minimum": "min",
    "minimum": "min",
    "min": "min",
    "tl.where": "select",
    "where": "select",
    "select": "select",
    "add": "add",
    "tile_add": "add",
    "sub": "sub",
    "tile_sub": "sub",
    "mul": "mul",
    "tile_mul": "mul",
    "div": "div",
    "tile_div": "div",
}

SUPPORTED_OPS = {
    "kernel",
    "return",
    "assign",
    "call",
    "if",
    "endif",
    "for",
    "endfor",
    "while",
    "endwhile",
    *BINOP_TO_IR.values(),
    *COMPARE_TO_IR.values(),
    *BOOL_TO_IR.values(),
    *CALL_TO_IR.values(),
}
