from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

from . import __version__
from .expr import ExprError
from .interpreter import ExecResult, Interpreter, InterpreterError
from .ir import IRError, Op, Program
from .lineir import parse_lineir, parse_lineir_file
from .memory import MemoryError_
from .scheduler import DependencyGraph, schedule
from .semantics import UnsupportedOpcode, effects_of, opcodes_by_kind, spec_for, touched_buffers
from .trace import Trace, TraceRecorder
from .values import ShapeError, Tile

PROG = "tile-interp"
EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2

PREVIEW = 8
KIND_ORDER = ("memory", "arith", "compare", "logic", "math", "control", "meta")

_USAGE_ERRORS = (IRError, OSError)
_RUNTIME_ERRORS = (ExprError, InterpreterError, MemoryError_, ShapeError, UnsupportedOpcode)

_CASE_REGISTRIES = ("CASES", "KERNEL_CASES", "EXAMPLES", "REGISTRY", "KERNELS")
_CASE_LOOKUPS = ("case_for", "get_case", "lookup", "case", "for_kernel")
_REFERENCE_KEYS = ("reference", "kernel", "fn", "func", "callable")
_INPUT_KEYS = ("inputs", "bindings", "args")
_OUTPUT_KEYS = ("outputs", "output_names", "buffers")


class CliError(Exception):
    """Raised for bad command-line input; reported with exit code 2."""


def build_parser() -> argparse.ArgumentParser:
    """Assemble the argument parser for every subcommand."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Execute, trace, schedule and verify unified tile IR line-format programs.",
    )
    parser.add_argument("--version", action="version", version=f"tile-interp {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    run = subparsers.add_parser("run", help="Execute a program and print the final buffer state.")
    _add_source(run)
    _add_execution_options(run)
    run.set_defaults(handler=cmd_run)

    trace = subparsers.add_parser("trace", help="Execute a program and dump its trace.")
    _add_source(trace)
    _add_execution_options(trace)
    trace.add_argument(
        "--format",
        choices=["text", "json", "schedule", "dot"],
        default="text",
        help="Trace rendering. Default: text.",
    )
    trace.add_argument("--out", help="Write the trace to this file instead of stdout.")
    trace.add_argument(
        "--capture-values",
        action="store_true",
        help="Record full intermediate values alongside the trace.",
    )
    trace.set_defaults(handler=cmd_trace)

    sched = subparsers.add_parser(
        "schedule", help="Print the dependency wavefronts and the speedup summary."
    )
    _add_source(sched)
    sched.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Cap the number of ops issued per step. Default: unlimited.",
    )
    sched.add_argument(
        "--format",
        choices=["text", "dot"],
        default="text",
        help="Schedule rendering. Default: text.",
    )
    sched.add_argument("--out", help="Write the schedule to this file instead of stdout.")
    sched.set_defaults(handler=cmd_schedule)

    verify = subparsers.add_parser(
        "verify", help="Run the equivalence harness; with no file, every example."
    )
    verify.add_argument("source", nargs="?", help="Line-format IR file. Default: all examples.")
    verify.add_argument("--rtol", type=float, default=1e-9, help="Relative tolerance.")
    verify.add_argument("--atol", type=float, default=1e-12, help="Absolute tolerance.")
    verify.add_argument(
        "--permutations",
        type=int,
        default=16,
        help="Dependency-respecting orders to sample. Default: 16.",
    )
    verify.set_defaults(handler=cmd_verify)

    ops = subparsers.add_parser("ops", help="List supported opcodes with kind and latency.")
    ops.set_defaults(handler=cmd_ops)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch one command line. 0 success, 1 failure, 2 usage or parse error."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] | None = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return EXIT_USAGE
    try:
        return handler(args)
    except (CliError, *_USAGE_ERRORS) as exc:
        print(f"{PROG}: error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except _RUNTIME_ERRORS as exc:
        print(f"{PROG}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FAIL


def cmd_run(args: argparse.Namespace) -> int:
    """Execute a program and report its final buffers and return value."""
    program = load_program(args.source)
    result = execute(program, args)
    print(_header(program, args))
    print()
    print(_buffer_report(result))
    if result.returned is not None:
        print()
        print(f"returned  {_describe(result.returned)}")
    return EXIT_OK


def cmd_trace(args: argparse.Namespace) -> int:
    """Execute a program and render its trace as text, JSON, a schedule, or dot."""
    program = load_program(args.source)
    recorder = TraceRecorder(capture_values=args.capture_values)
    result = execute(program, args, recorder=recorder)
    trace = result.trace
    if args.format == "json":
        output = _trace_json(trace, recorder)
    elif args.format == "text":
        output = trace.to_text()
        if args.capture_values and recorder.values:
            output = f"{output}\n\n{_captured_report(trace, recorder)}"
    else:
        graph = DependencyGraph.build(program)
        output = trace.to_dot(graph) if args.format == "dot" else trace.to_schedule_text(graph)
    _emit(output, args.out)
    return EXIT_OK


def cmd_schedule(args: argparse.Namespace) -> int:
    """Build the dependency DAG and print its wavefronts or its dot graph."""
    program = load_program(args.source)
    if args.max_parallel is not None and args.max_parallel < 1:
        raise CliError(f"--max-parallel must be >= 1, got {args.max_parallel}")
    if args.format == "dot":
        _emit(DependencyGraph.build(program).to_dot(), args.out)
        return EXIT_OK
    plan = schedule(program, max_parallel=args.max_parallel)
    head = f"kernel {program.kernel_name()}  ({len(program)} ops from {program.source_name})"
    _emit(f"{head}\n\n{plan.to_text()}", args.out)
    return EXIT_OK


def cmd_verify(args: argparse.Namespace) -> int:
    """Run the equivalence harness over one file or over every bundled example."""
    verify = _import_verify()
    kernels = _import_kernels()
    paths = [Path(args.source)] if args.source else example_paths()
    failures = 0
    for position, path in enumerate(paths):
        if position:
            print()
        program = load_program(str(path))
        report = _verify_one(verify, kernels, program, path, args)
        print(report.to_text())
        if not report.passed:
            failures += 1
    print()
    total = len(paths)
    noun = "kernel" if total == 1 else "kernels"
    if failures:
        print(f"FAIL  {failures} of {total} {noun} failed verification")
        return EXIT_FAIL
    print(f"PASS  {total} {noun} verified")
    return EXIT_OK


def cmd_ops(args: argparse.Namespace) -> int:
    """Print every supported opcode with its kind and its cost-model latency."""
    rows = [
        (name, kind, str(spec_for(Op(0, name, {})).latency))
        for kind in KIND_ORDER
        for name in opcodes_by_kind(kind)
    ]
    header = ("opcode", "kind", "latency")
    widths = [max(len(row[column]) for row in [header, *rows]) for column in range(3)]
    print(f"{header[0].ljust(widths[0])}  {header[1].ljust(widths[1])}  {header[2]}")
    print(f"{'-' * widths[0]}  {'-' * widths[1]}  {'-' * widths[2]}")
    for name, kind, latency in rows:
        print(f"{name.ljust(widths[0])}  {kind.ljust(widths[1])}  {latency}")
    print()
    print(f"{len(rows)} opcodes")
    return EXIT_OK


def load_program(path_text: str | None) -> Program:
    """Parse a line-format file, or stdin when the path is '-' or omitted."""
    if path_text is None or path_text == "-":
        return parse_lineir(sys.stdin.read(), "<stdin>")
    path = Path(path_text)
    if not path.is_file():
        raise CliError(f"no such file: {path}")
    return parse_lineir_file(path)


def parse_input(spec: str) -> tuple[str, object]:
    """Turn one --input NAME=JSON or NAME:dtype=JSON argument into a binding."""
    name, sep, payload = spec.partition("=")
    if not sep:
        raise CliError(f"--input expects NAME=JSON, got {spec!r}")
    dtype: str | None = None
    if ":" in name:
        name, _, dtype = name.partition(":")
        dtype = dtype.strip() or None
    name = name.strip()
    if not name.isidentifier():
        raise CliError(f"--input name {name!r} is not an identifier")
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise CliError(f"--input {name}: not valid JSON: {exc}") from exc
    return name, _to_value(name, decoded, dtype)


def parse_ints(text: str, option: str) -> tuple[int, ...]:
    """Parse a comma-separated integer tuple such as --grid 4,4."""
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        raise CliError(f"{option} expects at least one integer, got {text!r}")
    try:
        return tuple(int(part) for part in parts)
    except ValueError:
        raise CliError(f"{option} expects comma-separated integers, got {text!r}") from None


def bindings_from(args: argparse.Namespace, program: Program) -> dict[str, object]:
    """Seed buffers when asked or when nothing else is bound, then every --input flag."""
    specs = getattr(args, "input", None) or []
    bindings: dict[str, object] = {}
    if getattr(args, "seed_inputs", False):
        bindings.update(seed_inputs(program, args.source))
    elif not specs:
        bindings.update(seed_inputs(program, args.source, optional=True))
    for spec in specs:
        name, value = parse_input(spec)
        bindings[name] = value
    return bindings


def seed_inputs(
    program: Program,
    source: str | None,
    *,
    optional: bool = False,
) -> dict[str, object]:
    """The reference seed buffers registered for this kernel."""
    path = Path(source) if source else Path(program.source_name)
    try:
        case = _resolve_case(_import_kernels(), program, path)
    except CliError:
        if optional:
            return {}
        raise
    if case is None:
        if optional:
            return {}
        raise CliError(f"no reference seed inputs registered for {program.kernel_name()!r}")
    return dict(_normalize_case(case, program, path)[1])


def execute(
    program: Program,
    args: argparse.Namespace,
    *,
    recorder: TraceRecorder | None = None,
) -> ExecResult:
    """Run a program with the grid, ids and inputs the command line supplied."""
    grid = parse_ints(args.grid, "--grid")
    pids = parse_ints(args.pid, "--pid")
    interpreter = Interpreter(program, recorder=recorder, grid=grid, program_ids=pids)
    try:
        return interpreter.run(**bindings_from(args, program))
    except MemoryError_ as exc:
        raise MemoryError_(f"{exc}\n{_buffer_hint(program)}") from exc


def example_paths() -> list[Path]:
    """Every bundled example, sorted, for a bare 'verify' invocation."""
    directory = project_root() / "examples"
    paths = sorted(directory.glob("*.lineir"))
    if not paths:
        raise CliError(f"no .lineir examples found in {directory}")
    return paths


def project_root() -> Path:
    """The directory holding the tile_interp package."""
    return Path(__file__).resolve().parent.parent


def _add_source(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("source", nargs="?", help="Line-format IR file. Reads stdin when omitted.")


def _add_execution_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        action="append",
        metavar="NAME=JSON",
        help="Bind an input buffer or scalar. Repeatable. NAME:dtype=JSON forces a tile.",
    )
    parser.add_argument("--grid", default="1", help="Launch grid, comma separated. Default: 1.")
    parser.add_argument("--pid", default="0", help="Program ids, comma separated. Default: 0.")
    parser.add_argument(
        "--seed-inputs",
        action="store_true",
        help="Start from the reference seed buffers for this kernel; --input overrides them.",
    )


def _to_value(name: str, decoded: object, dtype: str | None) -> object:
    if isinstance(decoded, dict):
        return _tile_from_spec(name, decoded, dtype)
    if isinstance(decoded, list):
        return Tile.from_nested(decoded, dtype)
    if dtype:
        return Tile.scalar(decoded, dtype)
    return decoded


def _tile_from_spec(name: str, spec: dict[str, object], dtype: str | None) -> Tile:
    resolved = spec.get("dtype", dtype)
    if resolved is not None and not isinstance(resolved, str):
        raise CliError(f"--input {name}: dtype must be a string")
    shape = spec.get("shape")
    if "data" in spec:
        data = spec["data"]
        if shape is None:
            return Tile.from_nested(data, resolved)
        return Tile.from_flat(_flatten(data), shape, resolved)
    if shape is None:
        raise CliError(f"--input {name}: object form needs 'data' or 'shape'")
    return Tile.full(shape, spec.get("value", 0), resolved or "f32")


def _flatten(data: object) -> list[object]:
    if not isinstance(data, list):
        return [data]
    flat: list[object] = []
    for item in data:
        flat.extend(_flatten(item))
    return flat


def _header(program: Program, args: argparse.Namespace) -> str:
    grid = parse_ints(args.grid, "--grid")
    pids = parse_ints(args.pid, "--pid")
    return (
        f"kernel {program.kernel_name()}  ({len(program)} ops, {program.source_lang})\n"
        f"source {program.source_name}\n"
        f"grid {grid}  pid {pids}"
    )


def _buffer_report(result: ExecResult) -> str:
    names = result.memory.names()
    if not names:
        return "buffers: none"
    tiles = [(name, result.memory.buffer(name).tile) for name in names]
    width = max(len(name) for name, _ in tiles)
    shape_width = max(len(tile.describe()) for _, tile in tiles)
    lines = ["buffers"]
    for name, tile in tiles:
        lines.append(f"  {name.ljust(width)}  {tile.describe().ljust(shape_width)}  {_preview(tile)}")
    return "\n".join(lines)


def _preview(tile: Tile) -> str:
    head = ", ".join(repr(value) for value in tile.data[:PREVIEW])
    if tile.size > PREVIEW:
        head = f"{head}, ..."
    return f"[{head}]"


def _describe(value: object) -> str:
    if isinstance(value, Tile):
        return f"{value.describe()}  {_preview(value)}"
    return repr(value)


def _trace_json(trace: Trace, recorder: TraceRecorder) -> str:
    events = json.loads(trace.to_json())
    if not recorder.capture_values:
        return json.dumps(events, indent=2)
    values = {str(seq): recorder.values[seq] for seq in sorted(recorder.values)}
    return json.dumps({"events": events, "values": values}, indent=2)


def _captured_report(trace: Trace, recorder: TraceRecorder) -> str:
    lookup = {event.seq: event for event in trace.events}
    lines = ["captured values"]
    for seq in sorted(recorder.values):
        event = lookup.get(seq)
        label = f"{event.op_index:04d} {event.opcode}" if event else f"seq {seq}"
        lines.append(f"  {label:<18} {recorder.values[seq]!r}")
    return "\n".join(lines)


def _buffer_hint(program: Program) -> str:
    names: list[str] = []
    for op in program.ops:
        try:
            found = touched_buffers([op])
        except (UnsupportedOpcode, IRError):
            continue
        names.extend(name for name in found if name not in names)
    if not names:
        return "hint: bind inputs with --input NAME=JSON"
    joined = ", ".join(names)
    return f"hint: this program touches {joined}; bind them with --input NAME=JSON"


def _emit(text: str, out: str | None) -> None:
    if out:
        Path(out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def _import_verify() -> object:
    try:
        from . import verify
    except ImportError as exc:
        raise CliError(f"the verification harness is unavailable: {exc}") from exc
    if not hasattr(verify, "verify_kernel"):
        raise CliError("tile_interp.verify does not export verify_kernel")
    return verify


def _import_kernels() -> object:
    root = str(project_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from reference import kernels
    except ImportError as exc:
        raise CliError(f"the reference kernels are unavailable: {exc}") from exc
    return kernels


def _verify_one(
    verify: object,
    kernels: object,
    program: Program,
    path: Path,
    args: argparse.Namespace,
) -> object:
    case = _resolve_case(kernels, program, path)
    if case is None:
        detail = f"no reference kernel registered for {program.kernel_name()!r} ({path.name})"
        return _failed_report(verify, program, "reference-case", detail)
    reference, inputs, outputs = _normalize_case(case, program, path)
    try:
        return verify.verify_kernel(
            program,
            reference,
            inputs,
            outputs=outputs,
            rtol=args.rtol,
            atol=args.atol,
            permutations=args.permutations,
        )
    except _harness_errors(verify) as exc:
        return _failed_report(verify, program, "harness", f"{path.name}: {exc}")


def _harness_errors(verify: object) -> tuple[type[BaseException], ...]:
    error = getattr(verify, "VerificationError", None)
    if isinstance(error, type) and issubclass(error, Exception):
        return (error,)
    return ()


def _resolve_case(kernels: object, program: Program, path: Path) -> object | None:
    keys = [program.kernel_name(), path.stem, path.name, str(path)]
    for attribute in _CASE_REGISTRIES:
        registry = getattr(kernels, attribute, None)
        if isinstance(registry, dict):
            for key in keys:
                if key in registry and _usable(registry[key]):
                    return registry[key]
    for attribute in _CASE_LOOKUPS:
        finder = getattr(kernels, attribute, None)
        if not callable(finder):
            continue
        for key in keys:
            try:
                found = finder(key)
            except (KeyError, LookupError, ValueError):
                continue
            if found is not None and _usable(found):
                return found
    return None


def _usable(case: object) -> bool:
    """Whether a registry entry carries inputs, not just a bare reference callable."""
    if isinstance(case, (tuple, list)):
        return len(case) == 3
    return _case_field(case, _INPUT_KEYS) is not None


def _normalize_case(
    case: object, program: Program, path: Path
) -> tuple[Callable[[dict[str, Tile]], dict[str, Tile]], dict[str, Tile], list[str]]:
    if isinstance(case, (tuple, list)) and len(case) == 3:
        reference, inputs, outputs = case
    else:
        reference = _case_field(case, _REFERENCE_KEYS)
        inputs = _case_field(case, _INPUT_KEYS)
        outputs = _case_field(case, _OUTPUT_KEYS)
    if not callable(reference):
        raise CliError(f"{path.name}: reference case exposes no callable reference kernel")
    if callable(inputs):
        inputs = inputs()
    if not isinstance(inputs, dict):
        raise CliError(f"{path.name}: reference case exposes no input mapping")
    if not outputs:
        outputs = _output_buffers(program)
    return reference, dict(inputs), list(outputs)


def _case_field(case: object, keys: tuple[str, ...]) -> object:
    if isinstance(case, dict):
        for key in keys:
            if key in case:
                return case[key]
        return None
    for key in keys:
        value = getattr(case, key, None)
        if value is not None:
            return value
    return None


def _output_buffers(program: Program) -> list[str]:
    names: dict[str, None] = {}
    for op in program.ops:
        try:
            effects = effects_of(op)
        except (UnsupportedOpcode, IRError):
            continue
        for name in effects.mem_writes:
            names.setdefault(name, None)
    return list(names)


def _failed_report(verify: object, program: Program, name: str, detail: str) -> object:
    check = verify.Check(name=name, passed=False, detail=detail)
    return verify.VerificationReport(kernel=program.kernel_name(), checks=[check])
