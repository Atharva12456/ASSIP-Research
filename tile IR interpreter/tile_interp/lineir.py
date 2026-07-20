from __future__ import annotations

import json
import re
from pathlib import Path

from .ir import IRError, Op, Program

HEADER = "# unified-tile-ir line-format v0"

_BARE_VALUE = re.compile(r"^[A-Za-z0-9_.$:/<>+\-]+$")
_DECODER = json.JSONDecoder()


def parse_lineir(text: str, source_name: str = "<memory>") -> Program:
    """Parse line-format IR text into a Program, honouring the '#' headers."""
    source_lang = "unknown"
    name = source_name
    max_ops = 220
    ops: list[Op] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            key, sep, value = line[1:].strip().partition("=")
            if not sep:
                continue
            key = key.strip()
            value = _unquote(value.strip())
            if key == "source_lang":
                source_lang = value
            elif key == "source":
                name = value
            elif key == "max_ops":
                try:
                    max_ops = int(value)
                except ValueError:
                    raise IRError(f"line {lineno}: max_ops is not an integer: {value!r}") from None
            continue
        op = _parse_op(line, lineno)
        if ops and op.index <= ops[-1].index:
            raise IRError(
                f"line {lineno}: op index {op.index:04d} does not follow {ops[-1].index:04d}; "
                f"indices must be unique and increasing"
            )
        ops.append(op)
    return Program(source_lang, name, ops, max_ops)


def parse_lineir_file(path: str | Path) -> Program:
    """Read and parse a .lineir file."""
    resolved = Path(path)
    return parse_lineir(resolved.read_text(encoding="utf-8"), str(resolved))


def to_lineir(program: Program) -> str:
    """Render a Program back to the exact sibling line format."""
    lines = [
        HEADER,
        f"# source_lang={program.source_lang}",
        f"# source={_quote(program.source_name)}",
        f"# max_ops={program.max_ops}",
    ]
    lines.extend(_format_op(op) for op in program.ops)
    return "\n".join(lines)


def _format_op(op: Op) -> str:
    attrs = " ".join(f"{key}={_quote(value)}" for key, value in op.attrs.items())
    if attrs:
        return f"{op.index:04d} | {op.opcode:<12} | {attrs}"
    return f"{op.index:04d} | {op.opcode:<12} |"


def _quote(value: str) -> str:
    if value == "" or not _BARE_VALUE.match(value):
        return json.dumps(value)
    return value


def _unquote(value: str) -> str:
    if value.startswith('"'):
        decoded, end = _decode_string(value, 0)
        if end != len(value):
            raise IRError(f"trailing text after quoted value: {value!r}")
        return decoded
    return value


def _parse_op(line: str, lineno: int) -> Op:
    parts = line.split("|", 2)
    if len(parts) < 2:
        raise IRError(f"line {lineno}: expected 'index | opcode | attrs', got {line!r}")
    index_text = parts[0].strip()
    if not index_text.isdigit():
        raise IRError(f"line {lineno}: op index is not a number: {index_text!r}")
    opcode = parts[1].strip()
    if not opcode:
        raise IRError(f"line {lineno}: missing opcode")
    attr_text = parts[2].strip() if len(parts) > 2 else ""
    return Op(int(index_text), opcode, _parse_attrs(attr_text, lineno))


def _parse_attrs(text: str, lineno: int) -> dict[str, str]:
    attrs: dict[str, str] = {}
    pos = 0
    size = len(text)
    while pos < size:
        while pos < size and text[pos].isspace():
            pos += 1
        if pos >= size:
            break
        eq = text.find("=", pos)
        if eq < 0:
            raise IRError(f"line {lineno}: attribute without '=': {text[pos:]!r}")
        key = text[pos:eq]
        if not key or any(char.isspace() for char in key):
            raise IRError(f"line {lineno}: malformed attribute key: {key!r}")
        pos = eq + 1
        if pos < size and text[pos] == '"':
            value, pos = _decode_string(text, pos)
        else:
            start = pos
            while pos < size and not text[pos].isspace():
                pos += 1
            value = text[start:pos]
        attrs[key] = value
    return attrs


def _decode_string(text: str, pos: int) -> tuple[str, int]:
    try:
        value, end = _DECODER.raw_decode(text, pos)
    except json.JSONDecodeError as exc:
        raise IRError(f"bad quoted value at offset {pos}: {exc}") from exc
    if not isinstance(value, str):
        raise IRError(f"quoted value is not a string at offset {pos}")
    return value, end
