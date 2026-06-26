from __future__ import annotations

import json
import re

from .ir import IROp, IRProgram

_BARE_VALUE = re.compile(r"^[A-Za-z0-9_.$:/<>+\-]+$")


def to_line_format(program: IRProgram) -> str:
    lines = [
        "# unified-tile-ir line-format v0",
        f"# source_lang={program.source_lang}",
        f"# source={_quote(program.source_name)}",
        f"# max_ops={program.max_ops}",
    ]
    lines.extend(_format_op(op) for op in program.ops)
    return "\n".join(lines)


def _format_op(op: IROp) -> str:
    attrs = " ".join(f"{key}={_quote(value)}" for key, value in op.attrs.items())
    if attrs:
        return f"{op.index:04d} | {op.opcode:<12} | {attrs}"
    return f"{op.index:04d} | {op.opcode:<12} |"


def _quote(value: str) -> str:
    if value == "" or not _BARE_VALUE.match(value):
        return json.dumps(value)
    return value
