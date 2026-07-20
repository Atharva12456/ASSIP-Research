from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from .ir import IRError, Op, Program
from .semantics import Effects, UnsupportedOpcode, effects_of, spec_for

_LCG_SEED = 0x5EED1234
_LCG_MULT = 1103515245
_LCG_ADD = 12345
_LCG_MASK = 0x7FFFFFFF


@dataclass(slots=True)
class Node:
    """One op in the dependency DAG, with its edges and its scheduled timing."""

    index: int
    op: Op
    effects: Effects
    preds: set[int] = field(default_factory=set)
    succs: set[int] = field(default_factory=set)
    level: int = 0
    start: int = 0
    finish: int = 0
    latency: int = 1

    @property
    def opcode(self) -> str:
        return self.op.opcode


class DependencyGraph:
    """RAW/WAR/WAW dependency DAG over a flat op list, with ASAP wavefronts."""

    __slots__ = ("nodes", "program")

    def __init__(self, program: Program, nodes: dict[int, Node]) -> None:
        self.program = program
        self.nodes = nodes

    @staticmethod
    def build(program: Program) -> DependencyGraph:
        """Derive every dependency edge from semantics.effects_of and order barriers."""
        nodes: dict[int, Node] = {}
        for op in program.ops:
            if op.index in nodes:
                raise IRError(
                    f"duplicate op index {op.index:04d}: the schedule would drop an op"
                )
            nodes[op.index] = Node(
                index=op.index,
                op=op,
                effects=_effects_for(op),
                latency=_latency_for(op),
            )

        env_writer: dict[str, int] = {}
        env_readers: dict[str, list[int]] = {}
        mem_writer: dict[str, int] = {}
        mem_readers: dict[str, list[int]] = {}
        region: list[int] = []
        last_barrier: int | None = None

        graph = DependencyGraph(program, nodes)
        for index in sorted(nodes):
            node = nodes[index]
            effects = node.effects
            if effects.is_barrier:
                for earlier in region:
                    graph._link(earlier, index)
                if last_barrier is not None:
                    graph._link(last_barrier, index)
                last_barrier = index
                region = []
            elif last_barrier is not None:
                graph._link(last_barrier, index)

            graph._apply_flow(index, effects.reads, effects.writes, env_writer, env_readers)
            graph._apply_flow(
                index, effects.mem_reads, effects.mem_writes, mem_writer, mem_readers
            )
            if not effects.is_barrier:
                region.append(index)

        graph._assign_times()
        return graph

    def levels(self) -> list[list[int]]:
        """ASAP wavefronts: level(n) = 0 with no preds, else 1 + max(level(pred))."""
        buckets: dict[int, list[int]] = {}
        for index in sorted(self.nodes):
            buckets.setdefault(self.nodes[index].level, []).append(index)
        if not buckets:
            return []
        return [sorted(buckets.get(step, [])) for step in range(max(buckets) + 1)]

    def wavefronts(self, max_parallel: int | None = None) -> list[list[int]]:
        """Issue steps honoring a per-step width cap; overflow spills by longest path."""
        if max_parallel is None:
            return self.levels()
        if max_parallel < 1:
            raise ValueError(f"max_parallel must be >= 1, got {max_parallel}")
        heights = self.heights()
        remaining = {index: len(node.preds) for index, node in self.nodes.items()}
        ready = sorted(index for index, count in remaining.items() if count == 0)
        issued: set[int] = set()
        steps: list[list[int]] = []
        while ready:
            ranked = sorted(ready, key=lambda i: (-heights[i], i))
            chosen = sorted(ranked[:max_parallel])
            steps.append(chosen)
            issued.update(chosen)
            held = [index for index in ready if index not in issued]
            for index in chosen:
                for succ in sorted(self.nodes[index].succs):
                    remaining[succ] -= 1
                    if remaining[succ] == 0:
                        held.append(succ)
            ready = sorted(set(held))
        if len(issued) != len(self.nodes):
            raise ValueError("dependency graph contains a cycle")
        return steps

    def heights(self) -> dict[int, int]:
        """Latency-weighted longest path from each node to a sink, inclusive."""
        heights: dict[int, int] = {}
        for index in sorted(self.nodes, reverse=True):
            node = self.nodes[index]
            best = 0
            for succ in node.succs:
                best = max(best, heights.get(succ, 0))
            heights[index] = node.latency + best
        return heights

    def critical_path(self) -> list[int]:
        """The latency-weighted longest chain of dependent ops, source to sink.

        Its weight is the lower bound no schedule can beat; makespan() is what the
        step-synchronised wavefront schedule actually costs.
        """
        if not self.nodes:
            return []
        heights = self.heights()
        roots = [index for index, node in self.nodes.items() if not node.preds]
        current = min(roots, key=lambda i: (-heights[i], i))
        path = [current]
        while True:
            succs = self.nodes[current].succs
            if not succs:
                return path
            current = min(succs, key=lambda i: (-heights[i], i))
            path.append(current)

    def makespan(self) -> int:
        """Finish time of the wavefront schedule: each step costs its slowest op.

        A wavefront machine issues a whole step and syncs before the next, so this
        is the sum of per-step maxima and is an upper bound on the critical path.
        """
        if not self.nodes:
            return 0
        return max(node.finish for node in self.nodes.values())

    def sequential_cost(self) -> int:
        """Total latency if every op ran one after another."""
        return sum(node.latency for node in self.nodes.values())

    def is_valid_order(self, order: Sequence[int]) -> bool:
        """True when order is a permutation of all ops with every pred placed earlier."""
        seen: set[int] = set()
        position: dict[int, int] = {}
        for slot, index in enumerate(order):
            if index not in self.nodes or index in seen:
                return False
            seen.add(index)
            position[index] = slot
        if len(seen) != len(self.nodes):
            return False
        for index, node in self.nodes.items():
            for pred in node.preds:
                if position[pred] >= position[index]:
                    return False
        return True

    def topological_orders(self, limit: int = 50) -> list[list[int]]:
        """Distinct dependency-respecting orders, generated deterministically."""
        if limit <= 0 or not self.nodes:
            return [] if limit <= 0 else [[]]
        found: dict[tuple[int, ...], None] = {}
        base = sorted(self.nodes)
        found[tuple(base)] = None
        rng = _LCG(_LCG_SEED)
        attempts = 0
        budget = max(64, limit * 24)
        while len(found) < limit and attempts < budget:
            attempts += 1
            found.setdefault(tuple(self._kahn(rng)), None)
        if len(found) < limit:
            for candidate in self._systematic_swaps(base):
                found.setdefault(tuple(candidate), None)
                if len(found) >= limit:
                    break
        return [list(order) for order in list(found)[:limit]]

    def to_dot(self) -> str:
        """Graphviz digraph with one rank per wavefront and one edge per dependency."""
        name = _escape(self.program.kernel_name())
        lines = [
            f'digraph "{name}" {{',
            "  rankdir=TB;",
            '  graph [fontname="Consolas"];',
            '  node [shape=box, style=rounded, fontname="Consolas", fontsize=10];',
            '  edge [fontname="Consolas", fontsize=8];',
        ]
        for index in sorted(self.nodes):
            node = self.nodes[index]
            lines.append(f'  n{index:04d} [label="{_escape(_label(node))}"];')
        for step, members in enumerate(self.levels()):
            if len(members) > 1:
                joined = " ".join(f"n{index:04d};" for index in members)
                lines.append(f"  {{ rank=same; {joined} }}  // step {step}")
        for index in sorted(self.nodes):
            for succ in sorted(self.nodes[index].succs):
                lines.append(f"  n{index:04d} -> n{succ:04d};")
        lines.append("}")
        return "\n".join(lines)

    def _link(self, pred: int, succ: int) -> None:
        if pred == succ or pred not in self.nodes or succ not in self.nodes:
            return
        self.nodes[succ].preds.add(pred)
        self.nodes[pred].succs.add(succ)

    def _apply_flow(
        self,
        index: int,
        reads: Iterable[str],
        writes: Iterable[str],
        writer: dict[str, int],
        readers: dict[str, list[int]],
    ) -> None:
        for name in reads:
            producer = writer.get(name)
            if producer is not None:
                self._link(producer, index)
            readers.setdefault(name, []).append(index)
        for name in writes:
            producer = writer.get(name)
            if producer is not None:
                self._link(producer, index)
            for reader in readers.get(name, ()):
                self._link(reader, index)
            writer[name] = index
            readers[name] = []

    def _assign_times(self) -> None:
        for index in sorted(self.nodes):
            node = self.nodes[index]
            node.level = 0 if not node.preds else 1 + max(
                self.nodes[pred].level for pred in node.preds
            )
        starts: dict[int, int] = {}
        clock = 0
        for members in self.levels():
            width = max(self.nodes[index].latency for index in members)
            for index in members:
                starts[index] = clock
            clock += width
        for index, node in self.nodes.items():
            node.start = starts.get(index, 0)
            node.finish = node.start + node.latency

    def _kahn(self, rng: _LCG) -> list[int]:
        remaining = {index: len(node.preds) for index, node in self.nodes.items()}
        ready = sorted(index for index, count in remaining.items() if count == 0)
        order: list[int] = []
        while ready:
            pick = rng.below(len(ready))
            index = ready.pop(pick)
            order.append(index)
            for succ in sorted(self.nodes[index].succs):
                remaining[succ] -= 1
                if remaining[succ] == 0:
                    ready.append(succ)
            ready.sort()
        return order

    def _systematic_swaps(self, base: Sequence[int]) -> list[list[int]]:
        """Every single adjacent swap of a provably independent pair, in order."""
        results: list[list[int]] = []
        for slot in range(len(base) - 1):
            left, right = base[slot], base[slot + 1]
            if left in self.nodes[right].preds or right in self.nodes[left].preds:
                continue
            candidate = list(base)
            candidate[slot], candidate[slot + 1] = right, left
            if self.is_valid_order(candidate):
                results.append(candidate)
        return results


@dataclass(slots=True)
class Schedule:
    """The wavefront issue plan for a program, plus its cost-model numbers."""

    levels: list[list[int]]
    makespan: int
    critical_path: list[int]
    sequential_cost: int
    speedup: float
    max_parallel: int | None = None
    deferred: list[int] = field(default_factory=list)
    widest_step: int = 0

    def to_text(self) -> str:
        """Human-readable summary of the wavefronts and the cost model."""
        lines = []
        for step, members in enumerate(self.levels):
            count = len(members)
            noun = "op" if count == 1 else "ops"
            tag = f"({count} {noun}, parallel)" if count > 1 else f"({count} {noun})"
            body = " | ".join(f"{index:04d}" for index in members)
            lines.append(f"step {step:<3} {tag:<22} {body}")
        lines.append("")
        lines.append(f"sequential cost : {self.sequential_cost}")
        lines.append(f"makespan        : {self.makespan}")
        lines.append(f"speedup         : {self.speedup:.2f}x")
        lines.append(f"widest step     : {self.widest_step}")
        if self.max_parallel is not None:
            lines.append(f"max_parallel    : {self.max_parallel}")
        if self.deferred:
            deferred = ", ".join(f"{index:04d}" for index in self.deferred)
            lines.append(f"deferred by cap : {deferred}")
        path = " -> ".join(f"{index:04d}" for index in self.critical_path)
        lines.append(f"critical path   : {path}")
        return "\n".join(lines)


def schedule(program: Program, *, max_parallel: int | None = None) -> Schedule:
    """Build the dependency DAG and return its ASAP wavefront schedule."""
    graph = DependencyGraph.build(program)
    levels = graph.wavefronts(max_parallel)
    sequential_cost = graph.sequential_cost()
    makespan = _lockstep_makespan(graph, levels)
    speedup = (sequential_cost / makespan) if makespan else 1.0
    deferred = [
        index
        for step, members in enumerate(levels)
        for index in members
        if graph.nodes[index].level != step
    ]
    return Schedule(
        levels=levels,
        makespan=makespan,
        critical_path=graph.critical_path(),
        sequential_cost=sequential_cost,
        speedup=speedup,
        max_parallel=max_parallel,
        deferred=sorted(deferred),
        widest_step=max((len(members) for members in levels), default=0),
    )


class _LCG:
    """Seeded linear congruential generator; the project bans real randomness."""

    __slots__ = ("state",)

    def __init__(self, seed: int) -> None:
        self.state = seed & _LCG_MASK

    def next(self) -> int:
        self.state = (_LCG_MULT * self.state + _LCG_ADD) & _LCG_MASK
        return self.state

    def below(self, bound: int) -> int:
        """Draw from [0, bound) off the high bits; the low bits of an LCG barely move."""
        if bound <= 1:
            return 0
        return (self.next() * bound) >> 31


def _lockstep_makespan(graph: DependencyGraph, levels: Sequence[Sequence[int]]) -> int:
    total = 0
    for members in levels:
        total += max((graph.nodes[index].latency for index in members), default=0)
    return total


def _effects_for(op: Op) -> Effects:
    """Effects for an op, degrading an unknown or malformed op to a full barrier."""
    try:
        return effects_of(op)
    except (UnsupportedOpcode, IRError):
        return Effects(reads=[], writes=[], mem_reads=[], mem_writes=[], is_barrier=True)


def _latency_for(op: Op) -> int:
    try:
        return max(0, int(spec_for(op).latency))
    except UnsupportedOpcode:
        return 1


def _label(node: Node) -> str:
    attrs = " ".join(f"{key}={value}" for key, value in node.op.attrs.items())
    if len(attrs) > 48:
        attrs = attrs[:45] + "..."
    head = f"{node.index:04d} {node.opcode}  L{node.level} t{node.start}"
    return f"{head}\n{attrs}" if attrs else head


def _escape(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return escaped.replace("\r\n", "\\n").replace("\n", "\\n")
