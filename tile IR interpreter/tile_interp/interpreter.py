from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

from .expr import eval_attr
from .ir import Op, Program
from .memory import Memory
from .semantics import Effects, bind_loop_value, effects_of, spec_for
from .values import Tile

if TYPE_CHECKING:
    from .trace import Trace, TraceRecorder


class InterpreterError(Exception):
    """Raised for unbalanced control flow, runaway loops, and bad execution orders."""


_OPENERS = {"if": "endif", "for": "endfor", "while": "endwhile"}
_CLOSERS = {"endif": "if", "endfor": "for", "endwhile": "while"}


@dataclass(slots=True)
class BlockMap:
    """Bracket pairings for the flat control-flow markers of a program."""

    close: dict[int, int] = field(default_factory=dict)
    open: dict[int, int] = field(default_factory=dict)
    alt: dict[int, int] = field(default_factory=dict)
    alt_end: dict[int, int] = field(default_factory=dict)


def match_blocks(program: Program) -> BlockMap:
    """Pair every control marker by position, raising InterpreterError when unbalanced."""
    blocks = BlockMap()
    stack: list[int] = []
    for position, op in enumerate(program.ops):
        opcode = op.opcode
        if opcode in _OPENERS:
            stack.append(position)
        elif opcode == "else":
            if not stack or program.ops[stack[-1]].opcode != "if":
                raise InterpreterError(f"op {op.index:04d}: 'else' without a matching 'if'")
            head = stack[-1]
            if head in blocks.alt:
                raise InterpreterError(
                    f"op {op.index:04d}: second 'else' for the 'if' at op "
                    f"{program.ops[head].index:04d}"
                )
            blocks.alt[head] = position
        elif opcode in _CLOSERS:
            wanted = _CLOSERS[opcode]
            if not stack:
                raise InterpreterError(
                    f"op {op.index:04d}: '{opcode}' without a matching '{wanted}'"
                )
            head = stack.pop()
            head_op = program.ops[head]
            if head_op.opcode != wanted:
                raise InterpreterError(
                    f"op {op.index:04d}: '{opcode}' closes '{head_op.opcode}' opened at op "
                    f"{head_op.index:04d}"
                )
            blocks.close[head] = position
            blocks.open[position] = head
            if head in blocks.alt:
                blocks.alt_end[blocks.alt[head]] = position
    if stack:
        head_op = program.ops[stack[-1]]
        raise InterpreterError(f"op {head_op.index:04d}: '{head_op.opcode}' is never closed")
    return blocks


class ExecContext:
    """The mutable state one execution threads through every op."""

    __slots__ = ("env", "memory", "program", "grid", "program_ids", "returned", "halted")

    def __init__(
        self,
        program: Program,
        memory: Memory,
        *,
        grid: tuple[int, ...] = (1,),
        program_ids: tuple[int, ...] = (0,),
        env: dict[str, object] | None = None,
    ) -> None:
        self.program = program
        self.memory = memory
        self.env: dict[str, object] = {} if env is None else env
        self.grid = tuple(grid)
        self.program_ids = tuple(program_ids)
        self.returned: object = None
        self.halted = False

    def value(self, text: str | None, default: object = None) -> object:
        """Evaluate an attribute value against the environment, or return default."""
        if text is None or not text.strip():
            return default
        return eval_attr(text, self.env)

    def bind(self, name: str, value: object) -> None:
        """Bind one environment name."""
        self.env[name] = value

    def __repr__(self) -> str:
        return f"ExecContext(env={sorted(self.env)}, memory={self.memory!r})"


@dataclass(slots=True)
class ExecResult:
    """Everything one execution produced: final state, trace, return value, order."""

    env: dict[str, object]
    memory: Memory
    trace: "Trace"
    returned: object
    order: list[int]

    def output(self, name: str) -> Tile:
        """The final contents of a named buffer."""
        return self.memory.buffer(name).tile


class Interpreter:
    """Executes a Program either in index order or in a supplied dependency-safe order."""

    __slots__ = (
        "program",
        "_memory",
        "_recorder",
        "grid",
        "program_ids",
        "max_iterations",
        "max_steps",
    )

    def __init__(
        self,
        program: Program,
        *,
        memory: Memory | None = None,
        recorder: "TraceRecorder | None" = None,
        grid: tuple[int, ...] = (1,),
        program_ids: tuple[int, ...] = (0,),
        max_iterations: int = 10000,
        max_steps: int = 1_000_000,
    ) -> None:
        self.program = program
        self._memory = memory
        self._recorder = recorder
        self.grid = tuple(grid)
        self.program_ids = tuple(program_ids)
        self.max_iterations = int(max_iterations)
        self.max_steps = int(max_steps)

    def run(self, **bindings: object) -> ExecResult:
        """Execute every op in index order with full structured control flow."""
        ctx = self._context(bindings)
        recorder = self._new_recorder()
        blocks = match_blocks(self.program)
        ops = self.program.ops
        order: list[int] = []
        for_state: dict[int, tuple[list[object], int]] = {}
        while_counts: dict[int, int] = {}
        position = 0
        steps = 0
        while position < len(ops):
            steps += 1
            if steps > self.max_steps:
                raise InterpreterError(
                    f"execution exceeded {self.max_steps} steps; the program does not terminate"
                )
            op = ops[position]
            opcode = op.opcode
            spec = spec_for(op)
            if opcode == "if":
                taken = bool(spec.execute(ctx, op))
                recorder.record(op, ctx, taken)
                order.append(op.index)
                if taken:
                    position += 1
                else:
                    alt = blocks.alt.get(position)
                    position = (alt + 1) if alt is not None else blocks.close[position]
            elif opcode == "else":
                recorder.record(op, ctx, None)
                order.append(op.index)
                position = blocks.alt_end[position]
            elif opcode == "while":
                count = while_counts.get(position, 0) + 1
                if count > self.max_iterations:
                    raise InterpreterError(
                        f"op {op.index:04d}: while loop exceeded {self.max_iterations} iterations"
                    )
                keep = bool(spec.execute(ctx, op))
                recorder.record(op, ctx, keep)
                order.append(op.index)
                if keep:
                    while_counts[position] = count
                    position += 1
                else:
                    while_counts.pop(position, None)
                    position = blocks.close[position] + 1
            elif opcode == "for":
                state = for_state.get(position)
                if state is None:
                    values = list(spec.execute(ctx, op))  # type: ignore[arg-type]
                    cursor = 0
                else:
                    values, cursor = state
                    cursor += 1
                if cursor >= len(values):
                    for_state.pop(position, None)
                    recorder.record(op, ctx, None)
                    order.append(op.index)
                    position = blocks.close[position] + 1
                else:
                    if cursor >= self.max_iterations:
                        raise InterpreterError(
                            f"op {op.index:04d}: for loop exceeded "
                            f"{self.max_iterations} iterations"
                        )
                    for_state[position] = (values, cursor)
                    bind_loop_value(ctx, op, values[cursor])
                    recorder.record(op, ctx, values[cursor])
                    order.append(op.index)
                    position += 1
            elif opcode in ("endfor", "endwhile"):
                recorder.record(op, ctx, None)
                order.append(op.index)
                position = blocks.open[position]
            elif opcode == "endif":
                recorder.record(op, ctx, None)
                order.append(op.index)
                position += 1
            elif opcode == "return":
                ctx.returned = spec.execute(ctx, op)
                ctx.halted = True
                recorder.record(op, ctx, ctx.returned)
                order.append(op.index)
                break
            else:
                result = spec.execute(ctx, op)
                recorder.record(op, ctx, result)
                order.append(op.index)
                position += 1
        return ExecResult(ctx.env, ctx.memory, recorder.trace, ctx.returned, order)

    def run_in_order(self, order: Sequence[int], **bindings: object) -> ExecResult:
        """Execute a permutation of the op indices; rejects programs with control flow."""
        positions = {op.index: index for index, op in enumerate(self.program.ops)}
        for op in self.program.ops:
            if spec_for(op).kind == "control":
                raise InterpreterError(
                    f"run_in_order cannot execute control flow: op {op.index:04d} is "
                    f"'{op.opcode}'"
                )
        wanted = list(order)
        if sorted(wanted) != sorted(positions):
            raise InterpreterError(
                f"run_in_order needs a permutation of all {len(positions)} op indices, "
                f"got {len(wanted)} entries"
            )
        ctx = self._context(bindings)
        recorder = self._new_recorder()
        executed: list[int] = []
        for index in wanted:
            op = self.program.ops[positions[index]]
            spec = spec_for(op)
            result = spec.execute(ctx, op)
            if op.opcode == "return":
                ctx.returned = result
                ctx.halted = True
                recorder.record(op, ctx, result)
                executed.append(op.index)
                break
            recorder.record(op, ctx, result)
            executed.append(op.index)
        return ExecResult(ctx.env, ctx.memory, recorder.trace, ctx.returned, executed)

    def effects(self, index: int) -> Effects:
        """Effects of one op, straight from the semantics table."""
        return effects_of(self.program.op(index))

    def _context(self, bindings: dict[str, object]) -> ExecContext:
        memory = self._memory.copy() if self._memory is not None else Memory()
        env: dict[str, object] = {}
        for name, value in bindings.items():
            env[name] = value
            if isinstance(value, Tile) and not memory.has(name):
                memory.declare(name, value)
        return ExecContext(
            self.program,
            memory,
            grid=self.grid,
            program_ids=self.program_ids,
            env=env,
        )

    def _new_recorder(self) -> "TraceRecorder":
        if self._recorder is not None:
            self._recorder.reset()
            return self._recorder
        from .trace import TraceRecorder

        return TraceRecorder()


def execute(program: Program, **bindings: object) -> ExecResult:
    """Convenience wrapper: build an Interpreter and run it once."""
    return Interpreter(program).run(**bindings)


def control_ops(program: Program) -> list[Op]:
    """Every control-flow op in a program, in index order."""
    return [op for op in program.ops if spec_for(op).kind == "control"]
