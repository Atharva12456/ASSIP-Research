"""Run a program end to end and show every stage: IR, trace, output, schedule.

    python demo.py                                  # the gemm example
    python demo.py reduction                        # any example by name
    python demo.py examples/gemm.lineir             # a line-format file
    python demo.py examples/gemm_tile_64x64_fixed.tileir   # CUDA Tile IR (translated)
    python demo.py mykernel.tileir --elems 40000    # your own code
    python demo.py --list                           # the built-in examples

Point it at your own .tileir or .lineir file to see the tile IR it becomes and
the values it produces. When the program is a built-in example it is also checked
against the plain-Python reference; for your own code, inputs are synthesized and
the reference check is skipped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from reference.kernels import EXAMPLES, example, example_names
from tile_interp.interpreter import Interpreter, InterpreterError
from tile_interp.ir import IRError
from tile_interp.lineir import parse_lineir_file, to_lineir
from tile_interp.memory import Memory, MemoryError_
from tile_interp.scheduler import DependencyGraph, schedule
from tile_interp.semantics import UnsupportedOpcode, memory_target
from tile_interp.trace import TraceRecorder
from tile_interp.values import Tile
from tile_interp.verify import verify_kernel

try:
    from tile_interp.cuda_tile import CuTileError
except Exception:  # pragma: no cover
    class CuTileError(Exception):
        pass

WIDTH = 78


def rule(title: str) -> None:
    print()
    print(f"{title} ".ljust(WIDTH, "-"))
    print()


def synthesize_seed(program, elems: int) -> dict[str, Tile]:
    """Deterministic 1-D inputs for every buffer a load or store names."""
    inputs, outputs = set(), set()
    for op in program.ops:
        if op.opcode == "load":
            inputs.add(memory_target(op)[0])
        elif op.opcode == "store":
            outputs.add(memory_target(op)[0])
    seed: dict[str, Tile] = {}
    for name in sorted(inputs | outputs):
        data = [float(i % 7) for i in range(elems)] if name in inputs else [0.0] * elems
        seed[name] = Tile.from_flat(data, (elems,), "f32")
    return seed


def load_any(path: Path):
    """Load a .lineir or .tileir file, returning (program, source_text_or_None)."""
    if path.suffix == ".tileir":
        from tile_interp.cuda_tile import translate_cuda_tile_file

        return translate_cuda_tile_file(path), path.read_text()
    return parse_lineir_file(path), None


def show(program, *, title: str, summary: str, source: str | None,
         seed: dict[str, Tile], outputs: list[str], reference=None,
         grid=(1,), pids=(0,)) -> int:
    print("=" * WIDTH)
    print(f"  {title}  --  {summary}")
    print("=" * WIDTH)

    if source is not None:
        rule("0. SOURCE (CUDA Tile IR)")
        print(source.rstrip())

    rule("1. THE TILE IR" + (" (translated)" if source is not None else ""))
    print(to_lineir(program))

    rule("2. INPUT BUFFERS")
    for name, tile in seed.items():
        preview = tile.to_nested() if tile.size <= 24 else f"<{tile.size} values>"
        print(f"  {name:10} {str(tile.shape):10} {tile.dtype:4} = {preview}")

    memory = Memory()
    for name, tile in seed.items():
        memory.declare(name, tile)
    recorder = TraceRecorder()
    result = Interpreter(program, memory=memory, recorder=recorder,
                         grid=grid, program_ids=pids).run(**seed)

    rule("3. EXECUTION TRACE")
    print(result.trace.to_text())

    rule("4. OUTPUT BUFFERS")
    for name in outputs:
        tile = result.memory.buffer(name).tile
        preview = tile.to_nested() if tile.size <= 24 else f"<{tile.size} values>"
        print(f"  {name:10} {str(tile.shape):10} {tile.dtype:4} = {preview}")

    rule("5. DEPENDENCY SCHEDULE")
    print(result.trace.to_schedule_text(DependencyGraph.build(program)))

    plan = schedule(program)
    print()
    print(f"{len(program)} ops in {len(plan.levels)} steps, speedup {plan.speedup:.2f}x")

    if reference is not None:
        rule("6. AGAINST THE PLAIN-PYTHON REFERENCE")
        report = verify_kernel(program, reference, seed, outputs=outputs)
        print(report.to_text())
        return 0 if report.passed else 1

    print()
    print("(no reference kernel registered for this program; reference check skipped)")
    return 0


def run_example(name: str) -> int:
    ex = example(name)
    return show(
        parse_lineir_file(ex.path),
        title=name, summary=ex.summary, source=None,
        seed=ex.seed(), outputs=ex.outputs, reference=ex.reference,
    )


def _required_buffers(program) -> set[str]:
    return {memory_target(op)[0] for op in program.ops if op.opcode in ("load", "store")}


def run_file(path: Path, elems: int, pids: tuple[int, ...]) -> int:
    program, source = load_any(path)
    required = _required_buffers(program)
    # Use the registered reference only when its buffers actually match this
    # program. A .tileir file can share a stem with a line-format example while
    # naming its buffers differently, so a stem match alone is not enough.
    if path.stem in EXAMPLES:
        ex = EXAMPLES[path.stem]
        seed = ex.seed()
        if required <= set(seed):
            return show(program, title=path.name, summary=ex.summary, source=source,
                        seed=seed, outputs=ex.outputs, reference=ex.reference)
    seed = synthesize_seed(program, elems)
    outputs = sorted({memory_target(op)[0] for op in program.ops if op.opcode == "store"})
    grid = pids or (1,)
    return show(program, title=path.name, summary="your program",
                source=source, seed=seed, outputs=outputs,
                grid=grid, pids=pids)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run a tile IR program end to end.")
    parser.add_argument("program", nargs="?", default="gemm",
                        help="an example name, or a .lineir / .tileir file")
    parser.add_argument("--elems", type=int, default=4096,
                        help="buffer size when inputs are synthesized")
    parser.add_argument("--pid", default="0,0,0", help="program ids, comma separated")
    parser.add_argument("--list", action="store_true", help="list the built-in examples")
    args = parser.parse_args(argv[1:])

    if args.list:
        print("examples:", ", ".join(example_names()))
        return 0

    pids = tuple(int(p) for p in args.pid.split(",") if p.strip())
    path = Path(args.program)
    looks_like_file = (
        path.suffix in (".lineir", ".tileir")
        or "/" in args.program
        or "\\" in args.program
    )
    try:
        if path.exists():
            return run_file(path, args.elems, pids)
        if looks_like_file:
            print(f"no such file: {path}", file=sys.stderr)
            print("  create the file first, then paste your CUDA Tile IR or line-format IR "
                  "into it.", file=sys.stderr)
            print("  a working template is at examples/vector_add.tileir", file=sys.stderr)
            return 2
        return run_example(args.program)
    except (CuTileError, IRError, MemoryError_, InterpreterError, UnsupportedOpcode) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except KeyError as exc:
        print(exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
