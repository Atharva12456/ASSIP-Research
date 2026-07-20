from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from .expr import ExprError
from .interpreter import ExecResult, Interpreter, InterpreterError
from .ir import IRError, Program
from .lineir import parse_lineir, to_lineir
from .memory import MemoryError_
from .scheduler import DependencyGraph, Schedule, schedule
from .semantics import UnsupportedOpcode, spec_for
from .values import ShapeError, Tile

CHECK_NAMES: tuple[str, ...] = (
    "reference-match",
    "memory-match",
    "schedule-sound",
    "permutation-sound",
    "trace-complete",
    "roundtrip",
)

_CONTROL_OPCODES = frozenset({"if", "else", "endif", "for", "endfor", "while", "endwhile"})

_EXEC_ERRORS: tuple[type[Exception], ...] = (
    InterpreterError,
    UnsupportedOpcode,
    IRError,
    MemoryError_,
    ShapeError,
    ExprError,
)


class VerificationError(Exception):
    """Raised when the harness itself cannot run, as distinct from a failing check."""


@dataclass(slots=True)
class Check:
    """One named equivalence assertion and its outcome."""

    name: str
    passed: bool
    detail: str

    def to_text(self) -> str:
        """Single report row."""
        status = "PASS" if self.passed else "FAIL"
        return f"  [{status}] {self.name:<18} {self.detail}"


@dataclass(slots=True)
class VerificationReport:
    """Every check run for one kernel, plus the overall verdict."""

    kernel: str
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True only when every check ran and every check passed."""
        return bool(self.checks) and all(check.passed for check in self.checks)

    def check(self, name: str) -> Check:
        """Look up one check by name."""
        for item in self.checks:
            if item.name == name:
                return item
        raise KeyError(f"no check named {name!r}")

    def failures(self) -> list[Check]:
        """Every failing check, in report order."""
        return [item for item in self.checks if not item.passed]

    def to_text(self) -> str:
        """Human-readable report, one row per check."""
        lines = [f"verification: {self.kernel}"]
        lines.extend(item.to_text() for item in self.checks)
        good = sum(1 for item in self.checks if item.passed)
        verdict = "EQUIVALENT" if self.passed else "NOT EQUIVALENT"
        lines.append(f"  {good} of {len(self.checks)} checks passed -> {verdict}")
        return "\n".join(lines)


def verify_kernel(
    program: Program,
    reference: Callable[[dict[str, Tile]], dict[str, Tile]],
    inputs: dict[str, Tile],
    *,
    outputs: list[str],
    rtol: float = 1e-9,
    atol: float = 1e-12,
    permutations: int = 16,
) -> VerificationReport:
    """Assert the interpreter reproduces a reference kernel, under every execution order."""
    if not program.ops:
        raise VerificationError("cannot verify an empty program")
    seeds = _fresh(inputs)
    try:
        expected = reference(_fresh(seeds))
    except Exception as exc:
        raise VerificationError(f"the reference kernel itself raised {_describe(exc)}") from exc
    if not isinstance(expected, dict):
        raise VerificationError(
            f"the reference must return dict[str, Tile], got {type(expected).__name__}"
        )
    interpreter = Interpreter(program)
    try:
        baseline = interpreter.run(**_fresh(seeds))
    except _EXEC_ERRORS as exc:
        return _aborted(program, f"the sequential run raised {_describe(exc)}")
    try:
        graph = DependencyGraph.build(program)
        plan = schedule(program)
    except _EXEC_ERRORS as exc:
        graph, plan, blocked = None, None, f"the dependency graph raised {_describe(exc)}"
    else:
        blocked = None
    control = _has_control(program)
    checks = [
        _check_reference(baseline, expected, seeds, outputs, rtol, atol),
        _check_memory(baseline, expected, seeds, rtol, atol),
    ]
    if graph is None or plan is None:
        detail = blocked or "the dependency graph could not be built"
        checks.append(Check(CHECK_NAMES[2], False, detail))
        checks.append(Check(CHECK_NAMES[3], False, detail))
    else:
        checks.append(
            _check_schedule(interpreter, graph, plan, baseline, seeds, control, rtol, atol)
        )
        checks.append(
            _check_permutations(
                interpreter, graph, baseline, seeds, control, permutations, rtol, atol
            )
        )
    checks.append(_check_trace(program, baseline, control))
    checks.append(_check_roundtrip(program))
    return VerificationReport(program.kernel_name(), checks)


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _aborted(program: Program, detail: str) -> VerificationReport:
    """Report shape preserved when the program cannot execute at all: every run check fails."""
    checks = [Check(name, False, detail) for name in CHECK_NAMES[:5]]
    checks.append(_check_roundtrip(program))
    return VerificationReport(program.kernel_name(), checks)


def _fresh(tiles: dict[str, Tile]) -> dict[str, Tile]:
    return {name: tile.copy() for name, tile in tiles.items()}


def _has_control(program: Program) -> bool:
    for op in program.ops:
        try:
            if spec_for(op).kind == "control":
                return True
        except UnsupportedOpcode:
            if op.opcode in _CONTROL_OPCODES:
                return True
    return False


def _tile_mismatch(name: str, got: object, want: object, rtol: float, atol: float) -> str | None:
    """Describe the first disagreement between two tiles, or None when they agree."""
    if not isinstance(want, Tile):
        return f"{name}: reference produced {type(want).__name__}, not a Tile"
    if not isinstance(got, Tile):
        return f"{name}: interpreter produced {type(got).__name__}, not a Tile"
    if got.shape != want.shape:
        return f"{name}: shape {got.shape} != reference {want.shape}"
    if got.dtype != want.dtype:
        return f"{name}: dtype {got.dtype!r} != reference {want.dtype!r}"
    if got.allclose(want, rtol, atol):
        return None
    for position in range(got.size):
        left = Tile((), got.dtype, [got.data[position]])
        right = Tile((), want.dtype, [want.data[position]])
        if not left.allclose(right, rtol, atol):
            return (
                f"{name}: element {position} is {got.data[position]!r}, "
                f"reference says {want.data[position]!r}"
            )
    return f"{name}: values differ but no single element could be isolated"


def _same_value(got: object, want: object, rtol: float, atol: float) -> bool:
    if isinstance(got, Tile) or isinstance(want, Tile):
        if not isinstance(got, Tile) or not isinstance(want, Tile):
            return False
        return got.shape == want.shape and got.allclose(want, rtol, atol)
    if isinstance(got, float) or isinstance(want, float):
        try:
            return Tile.scalar(float(got)).allclose(Tile.scalar(float(want)), rtol, atol)
        except (TypeError, ValueError):
            return False
    return got == want


def _snapshot_mismatch(
    got: dict[str, Tile],
    want: dict[str, Tile],
    rtol: float,
    atol: float,
) -> str | None:
    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    if missing:
        return f"interpreter memory is missing {', '.join(missing)}"
    if extra:
        return f"interpreter memory has unexpected buffers {', '.join(extra)}"
    for name in sorted(want):
        problem = _tile_mismatch(name, got[name], want[name], rtol, atol)
        if problem is not None:
            return problem
    return None


def _check_reference(
    baseline: ExecResult,
    expected: dict[str, Tile],
    seeds: dict[str, Tile],
    outputs: list[str],
    rtol: float,
    atol: float,
) -> Check:
    name = CHECK_NAMES[0]
    if not outputs:
        return Check(name, False, "no output names were requested; the check would be vacuous")
    if not expected:
        return Check(name, False, "the reference produced no outputs")
    absent = [key for key in outputs if key not in expected]
    if absent:
        return Check(name, False, f"the reference never produced {', '.join(absent)}")
    unwritten = [key for key in outputs if not baseline.memory.has(key)]
    if unwritten:
        return Check(
            name, False, f"the interpreter never produced buffer(s) {', '.join(unwritten)}"
        )
    for key in outputs:
        problem = _tile_mismatch(key, baseline.memory.buffer(key).tile, expected[key], rtol, atol)
        if problem is not None:
            return Check(name, False, problem)
    changed = [
        key
        for key in outputs
        if key not in seeds or not expected[key].allclose(seeds[key], rtol, atol)
    ]
    if not changed:
        return Check(
            name,
            False,
            "every requested output still equals its seed value, so the check proves nothing",
        )
    shapes = ", ".join(f"{key}{expected[key].shape}:{expected[key].dtype}" for key in outputs)
    return Check(
        name,
        True,
        f"{len(outputs)} output(s) match the reference [{shapes}]; "
        f"{len(changed)} differ from the seed buffers",
    )


def _check_memory(
    baseline: ExecResult,
    expected: dict[str, Tile],
    seeds: dict[str, Tile],
    rtol: float,
    atol: float,
) -> Check:
    name = CHECK_NAMES[1]
    wanted = _fresh(seeds)
    wanted.update(expected)
    got = baseline.memory.snapshot()
    if not got:
        return Check(name, False, "the interpreter finished with no buffers at all")
    problem = _snapshot_mismatch(got, wanted, rtol, atol)
    if problem is not None:
        return Check(name, False, problem)
    return Check(
        name,
        True,
        f"all {len(got)} buffers match: {len(expected)} written by the kernel, "
        f"{len(got) - len(expected)} left at their seed values",
    )


def _check_schedule(
    interpreter: Interpreter,
    graph: DependencyGraph,
    plan: Schedule,
    baseline: ExecResult,
    seeds: dict[str, Tile],
    control: bool,
    rtol: float,
    atol: float,
) -> Check:
    name = CHECK_NAMES[2]
    flat = [index for step in plan.levels for index in step]
    if not graph.is_valid_order(flat):
        return Check(name, False, "the wavefront schedule is not a dependency-respecting order")
    if control:
        problem = _barrier_containment(graph)
        if problem is not None:
            return Check(name, False, problem)
        barriers = sum(1 for node in graph.nodes.values() if node.effects.is_barrier)
        return Check(
            name,
            True,
            f"n/a for execution (control flow present): the wavefront order is legal and all "
            f"{barriers} barriers fully separate the ops around them",
        )
    try:
        result = interpreter.run_in_order(flat, **_fresh(seeds))
    except _EXEC_ERRORS as exc:
        return Check(name, False, f"the wavefront order raised {_describe(exc)}")
    problem = _snapshot_mismatch(
        result.memory.snapshot(), baseline.memory.snapshot(), rtol, atol
    )
    if problem is not None:
        return Check(name, False, f"scheduled run diverged: {problem}")
    if not _same_value(result.returned, baseline.returned, rtol, atol):
        return Check(name, False, "scheduled run returned a different value")
    tag = " (identical to index order)" if flat == sorted(graph.nodes) else ""
    return Check(
        name,
        True,
        f"{len(plan.levels)} wavefronts, widest step {plan.widest_step}, "
        f"speedup {plan.speedup:.2f}x, results identical{tag}",
    )


def _barrier_containment(graph: DependencyGraph) -> str | None:
    """Every op before a barrier must be its ancestor, and every op after its descendant."""
    ancestors = _ancestors(graph)
    for index in sorted(graph.nodes):
        if not graph.nodes[index].effects.is_barrier:
            continue
        for other in sorted(graph.nodes):
            if other < index and other not in ancestors[index]:
                return f"op {other:04d} can float past the barrier at op {index:04d}"
            if other > index and index not in ancestors[other]:
                return f"op {other:04d} can float above the barrier at op {index:04d}"
    return None


def _ancestors(graph: DependencyGraph) -> dict[int, set[int]]:
    found: dict[int, set[int]] = {}
    for index in sorted(graph.nodes):
        reach: set[int] = set()
        for pred in graph.nodes[index].preds:
            reach.add(pred)
            reach |= found.get(pred, set())
        found[index] = reach
    return found


def _check_permutations(
    interpreter: Interpreter,
    graph: DependencyGraph,
    baseline: ExecResult,
    seeds: dict[str, Tile],
    control: bool,
    permutations: int,
    rtol: float,
    atol: float,
) -> Check:
    name = CHECK_NAMES[3]
    if permutations <= 0:
        return Check(name, False, "no permutations were requested; the check would be vacuous")
    try:
        orders = graph.topological_orders(permutations)
    except _EXEC_ERRORS as exc:
        return Check(name, False, f"sampling orders raised {_describe(exc)}")
    if not orders:
        return Check(name, False, "the scheduler produced no dependency-respecting orders")
    distinct = {tuple(order) for order in orders}
    if len(distinct) != len(orders):
        return Check(name, False, f"only {len(distinct)} of {len(orders)} sampled orders are distinct")
    index_order = sorted(graph.nodes)
    varied = [order for order in orders if list(order) != index_order]
    if not varied:
        return Check(
            name,
            False,
            "every sampled order is the plain index order, so no reordering was actually tested",
        )
    for order in orders:
        if not graph.is_valid_order(order):
            return Check(name, False, f"sampled order {_short(order)} violates a dependency")
    if control:
        return Check(
            name,
            True,
            f"n/a for execution (control flow present): {len(orders)} distinct orders sampled, "
            f"{len(varied)} differ from index order, all dependency-respecting",
        )
    wanted = baseline.memory.snapshot()
    for order in orders:
        try:
            result = interpreter.run_in_order(order, **_fresh(seeds))
        except _EXEC_ERRORS as exc:
            return Check(name, False, f"order {_short(order)} raised {_describe(exc)}")
        problem = _snapshot_mismatch(result.memory.snapshot(), wanted, rtol, atol)
        if problem is not None:
            return Check(name, False, f"order {_short(order)} diverged: {problem}")
        if not _same_value(result.returned, baseline.returned, rtol, atol):
            return Check(name, False, f"order {_short(order)} returned a different value")
    return Check(
        name,
        True,
        f"{len(orders)} distinct orders executed ({len(varied)} differ from index order, "
        f"{permutations} requested), all matching the sequential run",
    )


def _short(order: Sequence[int]) -> str:
    body = ",".join(f"{index:04d}" for index in list(order)[:6])
    return f"[{body}{',...' if len(order) > 6 else ''}]"


def _check_trace(program: Program, baseline: ExecResult, control: bool) -> Check:
    name = CHECK_NAMES[4]
    traced = baseline.trace.op_indices()
    if not traced:
        return Check(name, False, "the trace is empty")
    if traced != baseline.order:
        return Check(
            name,
            False,
            f"the trace does not follow the execution order "
            f"({len(traced)} events vs {len(baseline.order)} executed ops)",
        )
    known = [op.index for op in program.ops]
    unknown = sorted(set(traced) - set(known))
    if unknown:
        return Check(name, False, f"the trace mentions unknown op indices {unknown}")
    counts: dict[int, int] = {}
    for index in traced:
        counts[index] = counts.get(index, 0) + 1
    covered = len(counts)
    if not control:
        missing = [index for index in known if index not in counts]
        repeated = sorted(index for index, count in counts.items() if count != 1)
        if missing:
            return Check(name, False, f"ops never traced: {_short(missing)}")
        if repeated:
            return Check(name, False, f"ops traced more than once: {_short(repeated)}")
        return Check(name, True, f"all {len(known)} ops appear in the trace exactly once")
    return Check(
        name,
        True,
        f"{len(traced)} events cover {covered} of {len(known)} ops "
        f"(control flow re-executes the loop body), trace order matches execution order",
    )


def _check_roundtrip(program: Program) -> Check:
    name = CHECK_NAMES[5]
    text = to_lineir(program)
    again = parse_lineir(text, program.source_name)
    if len(again) != len(program):
        return Check(name, False, f"round trip produced {len(again)} ops, expected {len(program)}")
    for original, rebuilt in zip(program.ops, again.ops):
        if original.index != rebuilt.index or original.opcode != rebuilt.opcode:
            return Check(
                name,
                False,
                f"op {original.index:04d} came back as {rebuilt.index:04d} "
                f"({rebuilt.opcode!r} vs {original.opcode!r})",
            )
        if list(original.attrs.items()) != list(rebuilt.attrs.items()):
            return Check(
                name,
                False,
                f"op {original.index:04d} attrs changed: "
                f"{original.attrs} -> {rebuilt.attrs}",
            )
    if again != program:
        return Check(
            name,
            False,
            f"headers changed: {again.source_lang}/{again.source_name}/{again.max_ops} "
            f"vs {program.source_lang}/{program.source_name}/{program.max_ops}",
        )
    if to_lineir(again) != text:
        return Check(name, False, "re-rendering the parsed program produced different text")
    return Check(
        name,
        True,
        f"{len(program)} ops survive to_lineir -> parse_lineir unchanged, "
        f"and the text is stable",
    )
