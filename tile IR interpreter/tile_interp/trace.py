from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

from .expr import parse_names
from .ir import Op
from .values import Tile

if TYPE_CHECKING:
    from .interpreter import ExecContext
    from .scheduler import DependencyGraph

MAX_WIDTH = 118
MAX_CAPTURED = 256
ELLIPSIS = "..."

_FIELDS: tuple[str, ...] = (
    "seq",
    "op_index",
    "opcode",
    "level",
    "inputs",
    "output",
    "result",
    "mem_effect",
    "note",
)

_SYMBOLS: dict[str, str] = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "floordiv": "//",
    "mod": "%",
    "pow": "**",
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
    "eq": "==",
    "ne": "!=",
    "and": "and",
    "or": "or",
    "dot": "@",
}

_CONTROL = frozenset({"if", "else", "endif", "for", "endfor", "while", "endwhile"})

_OUTPUT_KEYS: tuple[str, ...] = ("out", "target")
_POINTER_KEYS: tuple[str, ...] = ("buf", "ptr")
_SKIP_KEYS: tuple[str, ...] = ("name", "params", "inplace")

_KEY_ORDER: dict[str, tuple[str, ...]] = {
    "store": ("value", "ptr", "buf", "mask"),
    "select": ("cond", "true", "false"),
    "load": ("ptr", "buf", "mask", "other"),
}

_MAX_PREDS = 6

_COLUMNS: tuple[tuple[str, str, int], ...] = (
    ("seq", "seq", 5),
    ("op_index", "op", 5),
    ("opcode", "opcode", 11),
    ("output", "out", 12),
    ("result", "result", 21),
    ("inputs", "inputs", 0),
    ("mem_effect", "memory", 22),
    ("note", "note", 20),
)


@dataclass(slots=True)
class TraceEvent:
    """One executed op: what it read, what it produced, and where it touched memory."""

    seq: int
    op_index: int
    opcode: str
    level: int | None = None
    inputs: dict[str, str] = field(default_factory=dict)
    output: str | None = None
    result: str | None = None
    mem_effect: str | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, object]:
        """JSON-ready mapping with every field present and a stable key order."""
        return {name: getattr(self, name) for name in _FIELDS}

    def detail(self) -> str:
        """Short human rendering such as 'c = a + b' or 'a <- A[64]'."""
        names = list(self.inputs)
        symbol = _SYMBOLS.get(self.opcode)
        if symbol and self.output and len(names) == 2:
            return f"{self.output} = {names[0]} {symbol} {names[1]}"
        if self.opcode in _CONTROL:
            operands = ", ".join(names)
            if self.output:
                if not operands:
                    return f"{self.opcode} {self.output}"
                return f"{self.opcode} {self.output} in {operands}"
            return f"{self.opcode} {operands}".strip()
        if self.opcode == "assign" and self.output and len(names) == 1:
            return f"{self.output} = {names[0]}"
        source = _mem_target(self.mem_effect, "read")
        if source is not None and self.output:
            return f"{self.output} <- {source}"
        target = _mem_target(self.mem_effect, "write")
        if target is not None:
            operand = names[0] if names else (self.result or "?")
            return f"{target} <- {operand}"
        if self.output:
            return f"{self.output} = {self.opcode}({', '.join(names)})"
        if names:
            return f"{self.opcode}({', '.join(names)})"
        return self.note or self.opcode


class Trace:
    """An ordered log of executed ops, renderable as a table, JSON, or a schedule."""

    __slots__ = ("events",)

    def __init__(self, events: list[TraceEvent] | None = None) -> None:
        self.events: list[TraceEvent] = list(events) if events else []

    def add(self, event: TraceEvent) -> None:
        """Append an event."""
        self.events.append(event)

    def __iter__(self) -> Iterator[TraceEvent]:
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)

    def op_indices(self) -> list[int]:
        """Op index of every event, in execution order."""
        return [event.op_index for event in self.events]

    def to_text(self) -> str:
        """Fixed-width table, one row per executed op, fitting a 120-column terminal."""
        if not self.events:
            return "trace: no events"
        rows = [
            {
                "seq": str(event.seq),
                "op_index": f"{event.op_index:04d}",
                "opcode": event.opcode,
                "output": event.output or "-",
                "result": event.result or "-",
                "inputs": _format_inputs(event.inputs),
                "mem_effect": event.mem_effect or "-",
                "note": event.note or "-",
            }
            for event in self.events
        ]
        columns = [entry for entry in _COLUMNS if entry[0] != "note" or any(e.note for e in self.events)]
        widths = _layout(columns, rows)
        lines = [" | ".join(title.ljust(widths[key]) for key, title, _ in columns).rstrip()]
        lines.append("-+-".join("-" * widths[key] for key, _, _ in columns))
        for row in rows:
            cells = [_fit(row[key], widths[key]).ljust(widths[key]) for key, _, _ in columns]
            lines.append(" | ".join(cells).rstrip())
        return "\n".join(lines)

    def to_json(self) -> str:
        """A JSON array of event objects, one per executed op, for programmatic diffing."""
        return json.dumps([event.to_dict() for event in self.events], indent=2)

    def to_schedule_text(self, graph: DependencyGraph) -> str:
        """Wavefront view: ops grouped by dependency level, with 'waits on' annotations."""
        if not self.events:
            return "schedule: no events"
        groups: dict[int, dict[int, list[TraceEvent]]] = {}
        for event in self.events:
            batch = groups.setdefault(_level_of(event, graph), {})
            batch.setdefault(event.op_index, []).append(event)
        lines: list[str] = []
        for level in sorted(groups):
            batch = groups[level]
            count = len(batch)
            label = "1 op" if count == 1 else f"{count} ops in parallel"
            lines.append(f"step {level:<2} ({label})")
            for index in sorted(batch):
                repeats = batch[index]
                lines.append(_schedule_line(repeats[0], graph, len(repeats)))
        lines.append("")
        lines.extend(_schedule_footer(graph))
        return "\n".join(lines)

    def to_dot(self, graph: DependencyGraph) -> str:
        """Graphviz digraph of the dependency DAG, rank-aligned so wavefronts line up."""
        seen: dict[int, TraceEvent] = {}
        for event in self.events:
            seen.setdefault(event.op_index, event)
        levels: dict[int, list[int]] = {}
        for index, node in graph.nodes.items():
            levels.setdefault(int(getattr(node, "level", 0)), []).append(index)
        lines = [
            "digraph tile_ir {",
            "  rankdir=TB;",
            '  graph [fontname="Consolas", nodesep=0.35, ranksep=0.55];',
            '  node [shape=box, style=rounded, fontname="Consolas", fontsize=10];',
            '  edge [color="#666666", arrowsize=0.7];',
        ]
        for index in sorted(graph.nodes):
            node = graph.nodes[index]
            lines.append(f"  n{index} [label=\"{_dot_label(index, node, seen.get(index))}\"{_dot_style(node, index in seen)}];")
        for level in sorted(levels):
            members = " ".join(f"n{index};" for index in sorted(levels[level]))
            lines.append(f"  {{ rank=same; {members} }}")
        for index in sorted(graph.nodes):
            for pred in sorted(graph.nodes[index].preds):
                lines.append(f"  n{pred} -> n{index};")
        lines.append("}")
        return "\n".join(lines)


class TraceRecorder:
    """Builds a Trace as the interpreter executes; optionally keeps full values."""

    __slots__ = ("_trace", "capture_values", "values")

    def __init__(self, *, capture_values: bool = False) -> None:
        self.capture_values = capture_values
        self.values: dict[int, object] = {}
        self._trace = Trace()

    @property
    def trace(self) -> Trace:
        """The trace accumulated so far."""
        return self._trace

    def reset(self) -> None:
        """Drop every recorded event and captured value."""
        self._trace = Trace()
        self.values.clear()

    def record(self, op: Op, ctx: ExecContext | None, result: object, level: int | None = None) -> None:
        """Log one executed op, formatting tiles as 'shape:dtype@digest'."""
        env = getattr(ctx, "env", None) or {}
        reads, writes, mem_reads, mem_writes = _effects_of(op)
        inputs: dict[str, str] = {}
        operands: list[object] = []
        for label in _operand_labels(op, reads):
            if label not in reads:
                inputs.setdefault(label, label)
            elif label in env:
                value = env[label]
                inputs[label] = _describe(value)
                operands.append(value)
            else:
                inputs.setdefault(label, "?")
        note = None
        seq = len(self._trace.events)
        if self.capture_values:
            note = self._capture(seq, result)
        event = TraceEvent(
            seq=seq,
            op_index=op.index,
            opcode=op.opcode,
            level=level,
            inputs=inputs,
            output=writes[0] if writes else None,
            result=None if result is None else _describe(result),
            mem_effect=_mem_effect(mem_reads, mem_writes, result, operands),
            note=note,
        )
        self._trace.add(event)

    def value_at(self, seq: int) -> object:
        """The captured value for one event, or None when nothing was captured."""
        return self.values.get(seq)

    def values_for(self, op_index: int) -> list[object]:
        """Every captured value produced by one op index, in execution order."""
        return [
            self.values[event.seq]
            for event in self._trace.events
            if event.op_index == op_index and event.seq in self.values
        ]

    def _capture(self, seq: int, result: object) -> str | None:
        if result is None:
            return None
        if not isinstance(result, Tile):
            self.values[seq] = result
            return None
        if result.size <= MAX_CAPTURED:
            self.values[seq] = result.to_nested()
            return None
        self.values[seq] = list(result.data[:MAX_CAPTURED])
        return f"captured {MAX_CAPTURED} of {result.size} values"


def _effects_of(op: Op) -> tuple[list[str], list[str], list[str], list[str]]:
    """Reads, writes, memory reads and memory writes, from semantics when available."""
    try:
        from .ir import IRError
        from .semantics import UnsupportedOpcode, effects_of
    except ImportError:
        return _guessed_effects(op)
    try:
        effects = effects_of(op)
    except (UnsupportedOpcode, IRError):
        return _guessed_effects(op)
    return (
        list(effects.reads),
        list(effects.writes),
        list(effects.mem_reads),
        list(effects.mem_writes),
    )


def _operand_keys(op: Op) -> list[str]:
    """Attribute keys that carry operands, in the order a reader expects them."""
    keys = [key for key in op.attrs if key not in _OUTPUT_KEYS and key not in _SKIP_KEYS]
    priority = _KEY_ORDER.get(op.opcode)
    if priority:
        keys.sort(key=lambda key: priority.index(key) if key in priority else len(priority))
    return keys


def _operand_labels(op: Op, reads: list[str]) -> list[str]:
    """Env names plus literal operand texts, ordered so 'c = a + b' reads naturally."""
    labels: list[str] = []
    for key in _operand_keys(op):
        text = op.attrs.get(key) or ""
        if not text.strip():
            continue
        names = parse_names(text)
        found = [name for name in names if name in reads and name not in labels]
        if found:
            labels.extend(found)
        elif not names and text.strip() not in labels:
            labels.append(text.strip())
    labels.extend(name for name in reads if name not in labels)
    return labels


def _guessed_effects(op: Op) -> tuple[list[str], list[str], list[str], list[str]]:
    """Attribute-shaped guess used when semantics does not know the opcode."""
    writes = [op.attrs[key] for key in _OUTPUT_KEYS if op.attrs.get(key)]
    reads: dict[str, None] = {}
    buffers: list[str] = []
    for key, value in op.attrs.items():
        if key in _OUTPUT_KEYS or key in _SKIP_KEYS:
            continue
        names = parse_names(value)
        if key in _POINTER_KEYS:
            buffers.append(names[0] if names else value.strip())
            names = names[1:]
        for name in names:
            reads.setdefault(name, None)
    if op.opcode == "load":
        return list(reads), writes, buffers, []
    if op.opcode == "store":
        return list(reads), writes, [], buffers
    return list(reads), writes, [], []


def _describe(value: object) -> str:
    """Format a runtime value for a trace cell."""
    if isinstance(value, Tile):
        return value.describe()
    if isinstance(value, (list, tuple)):
        return _fit(repr(value), 32)
    if isinstance(value, str):
        return _fit(repr(value), 24)
    return _fit(repr(value), 24)


def _mem_effect(
    mem_reads: list[str],
    mem_writes: list[str],
    result: object,
    operands: list[object],
) -> str | None:
    parts = [f"read {name}{_extent(result)}" for name in mem_reads]
    stored = next((value for value in operands if isinstance(value, Tile)), None)
    parts.extend(f"write {name}{_extent(stored)}" for name in mem_writes)
    return "; ".join(parts) if parts else None


def _extent(value: object) -> str:
    if isinstance(value, Tile) and value.ndim:
        return f"[{value.size}]"
    return ""


def _mem_target(mem_effect: str | None, verb: str) -> str | None:
    if not mem_effect:
        return None
    prefix = f"{verb} "
    for part in mem_effect.split("; "):
        if part.startswith(prefix):
            return part[len(prefix) :]
    return None


def _format_inputs(inputs: dict[str, str]) -> str:
    if not inputs:
        return "-"
    return ", ".join(
        name if name == value else f"{name}={value}" for name, value in inputs.items()
    )


def _fit(text: str, width: int) -> str:
    if width <= 0 or len(text) <= width:
        return text
    if width <= len(ELLIPSIS):
        return text[:width]
    return text[: width - len(ELLIPSIS)] + ELLIPSIS


def _layout(
    columns: list[tuple[str, str, int]],
    rows: list[dict[str, str]],
) -> dict[str, int]:
    natural = {
        key: max([len(title)] + [len(row[key]) for row in rows]) for key, title, _ in columns
    }
    widths = {key: min(natural[key], cap) for key, _, cap in columns if cap}
    fixed = sum(widths.values()) + 3 * (len(columns) - 1)
    remaining = MAX_WIDTH - fixed
    widths["inputs"] = max(12, min(natural["inputs"], remaining))
    return widths


def _level_of(event: TraceEvent, graph: DependencyGraph) -> int:
    node = graph.nodes.get(event.op_index)
    if node is not None:
        return int(getattr(node, "level", 0))
    return event.level if event.level is not None else 0


def _schedule_line(event: TraceEvent, graph: DependencyGraph, repeats: int = 1) -> str:
    node = graph.nodes.get(event.op_index)
    preds = sorted(node.preds) if node is not None else []
    detail = event.detail()
    if detail == event.opcode:
        detail = ""
    if repeats > 1:
        detail = f"{detail}  (x{repeats})".strip()
    body = f"{'':<10}{event.op_index:04d}  {event.opcode:<10} {detail}".rstrip()
    if not preds:
        return _fit(body, MAX_WIDTH)
    waits = f"[waits on {_pred_list(preds)}]"
    column = max(len(body) + 2, 62)
    return _fit(f"{body:<{column - 1}} {waits}", MAX_WIDTH)


def _pred_list(preds: list[int]) -> str:
    shown = ", ".join(f"{index:04d}" for index in preds[:_MAX_PREDS])
    if len(preds) > _MAX_PREDS:
        return f"{shown}, +{len(preds) - _MAX_PREDS} more"
    return shown


def _schedule_footer(graph: DependencyGraph) -> list[str]:
    makespan = int(graph.makespan())
    sequential = sum(
        max(0, int(node.finish) - int(node.start)) for node in graph.nodes.values()
    )
    if sequential <= 0:
        sequential = len(graph.nodes)
    speedup = sequential / makespan if makespan > 0 else 1.0
    lines = [
        f"makespan {makespan} cycles   sequential {sequential} cycles   speedup {speedup:.2f}x",
    ]
    path = list(graph.critical_path())
    if path:
        lines.append(_fit("critical path  " + " -> ".join(f"{i:04d}" for i in path), MAX_WIDTH))
    return lines


def _dot_label(index: int, node: object, event: TraceEvent | None) -> str:
    op = getattr(node, "op", None)
    opcode = getattr(op, "opcode", event.opcode if event else "?")
    parts = [f"{index:04d} {opcode}"]
    if event is not None:
        detail = event.detail()
        if detail and detail != opcode:
            parts.append(_fit(detail, 34))
    return "\\n".join(_dot_escape(part) for part in parts)


def _dot_style(node: object, executed: bool) -> str:
    if not executed:
        return ', color="#999999", fontcolor="#666666"'
    if getattr(getattr(node, "effects", None), "is_barrier", False):
        return ', style="rounded,filled", fillcolor="#fde8e8"'
    return ', style="rounded,filled", fillcolor="#e8f0fe"'


def _dot_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')
