from __future__ import annotations

import json
import unittest
from collections import Counter

from tile_interp.interpreter import Interpreter
from tile_interp.ir import Op
from tile_interp.lineir import parse_lineir
from tile_interp.scheduler import DependencyGraph
from tile_interp.trace import MAX_WIDTH, Trace, TraceEvent, TraceRecorder
from tile_interp.values import Tile

HEADLINE = """# unified-tile-ir line-format v0
# source_lang=triton
# source=examples/headline.py
0000 | kernel | name=headline params="A,B,C,D"
0001 | load   | out=a buf=A
0002 | load   | out=b buf=B
0003 | add    | out=c lhs=a rhs=b
0004 | load   | out=d buf=D
0005 | mul    | out=e lhs=c rhs=d
0006 | store  | buf=C value=e"""

CONTROL = """0000 | assign | out=t value=0
0001 | for    | target=i iter=range(3)
0002 | add    | out=t lhs=t rhs=i
0003 | endfor |"""


def inputs() -> dict[str, object]:
    """Buffers for the headline program."""
    return {
        "A": Tile.from_flat([1.0, 2.0], (2,), "f32"),
        "B": Tile.from_flat([3.0, 4.0], (2,), "f32"),
        "C": Tile.zeros(2, "f32"),
        "D": Tile.from_flat([2.0, 2.0], (2,), "f32"),
    }


def run_headline():
    """Execute the headline program and return (result, graph)."""
    program = parse_lineir(HEADLINE)
    result = Interpreter(program).run(**inputs())
    return result, DependencyGraph.build(program)


class EventCoverageTests(unittest.TestCase):
    """Every executed op produces exactly one event, in execution order."""

    def test_one_event_per_op(self) -> None:
        result, _ = run_headline()
        self.assertEqual(len(result.trace), 7)
        self.assertEqual(result.trace.op_indices(), list(range(7)))

    def test_events_match_the_execution_order(self) -> None:
        result, _ = run_headline()
        self.assertEqual(result.trace.op_indices(), result.order)

    def test_sequence_numbers_are_dense(self) -> None:
        result, _ = run_headline()
        self.assertEqual([event.seq for event in result.trace], list(range(7)))

    def test_every_op_appears_exactly_once(self) -> None:
        result, _ = run_headline()
        counts = Counter(result.trace.op_indices())
        self.assertEqual(sorted(counts), list(range(7)))
        self.assertTrue(all(count == 1 for count in counts.values()))

    def test_control_flow_repeats_loop_bodies(self) -> None:
        result = Interpreter(parse_lineir(CONTROL)).run()
        counts = Counter(result.trace.op_indices())
        self.assertEqual(counts[2], 3)
        self.assertEqual(counts[1], 4)
        self.assertEqual(result.trace.op_indices(), result.order)

    def test_run_in_order_traces_the_given_order(self) -> None:
        program = parse_lineir(HEADLINE)
        order = [0, 4, 1, 2, 3, 5, 6]
        result = Interpreter(program).run_in_order(order, **inputs())
        self.assertEqual(result.trace.op_indices(), order)


class EventContentTests(unittest.TestCase):
    """Field-level contents of a recorded event."""

    def setUp(self) -> None:
        self.result, self.graph = run_headline()
        self.events = {event.op_index: event for event in self.result.trace}

    def test_opcode_is_recorded(self) -> None:
        self.assertEqual(self.events[3].opcode, "add")
        self.assertEqual(self.events[6].opcode, "store")

    def test_output_name(self) -> None:
        self.assertEqual(self.events[1].output, "a")
        self.assertEqual(self.events[3].output, "c")
        self.assertIsNone(self.events[6].output)

    def test_result_uses_the_tile_describe_format(self) -> None:
        self.assertEqual(self.events[3].result, self.result.env["c"].describe())
        self.assertTrue(self.events[3].result.startswith("2:f32@"))

    def test_inputs_carry_operand_labels(self) -> None:
        self.assertEqual(sorted(self.events[3].inputs), ["a", "b"])
        self.assertEqual(self.events[3].inputs["a"], self.result.env["a"].describe())

    def test_memory_effects(self) -> None:
        self.assertEqual(self.events[1].mem_effect, "read A[2]")
        self.assertEqual(self.events[6].mem_effect, "write C[2]")
        self.assertIsNone(self.events[3].mem_effect)

    def test_detail_rendering(self) -> None:
        self.assertEqual(self.events[3].detail(), "c = a + b")
        self.assertEqual(self.events[1].detail(), "a <- A[2]")
        self.assertEqual(self.events[6].detail(), "C[2] <- e")

    def test_to_dict_has_every_field(self) -> None:
        payload = self.events[3].to_dict()
        self.assertEqual(
            list(payload),
            [
                "seq",
                "op_index",
                "opcode",
                "level",
                "inputs",
                "output",
                "result",
                "mem_effect",
                "note",
            ],
        )

    def test_missing_env_name_renders_as_a_question_mark(self) -> None:
        recorder = TraceRecorder()
        recorder.record(Op(0, "add", {"out": "c", "lhs": "a", "rhs": "b"}), None, None)
        self.assertEqual(recorder.trace.events[0].inputs, {"a": "?", "b": "?"})

    def test_unknown_opcode_still_records(self) -> None:
        recorder = TraceRecorder()
        recorder.record(Op(0, "frobnicate", {"out": "q", "value": "a"}), None, None, 7)
        event = recorder.trace.events[0]
        self.assertEqual(event.opcode, "frobnicate")
        self.assertEqual(event.output, "q")
        self.assertEqual(event.level, 7)


class RenderTests(unittest.TestCase):
    """All four render formats must produce useful, non-empty output."""

    def setUp(self) -> None:
        self.result, self.graph = run_headline()
        self.trace = self.result.trace

    def test_to_text_is_a_table(self) -> None:
        text = self.trace.to_text()
        self.assertTrue(text)
        lines = text.splitlines()
        self.assertEqual(len(lines), len(self.trace) + 2)
        self.assertIn("opcode", lines[0])
        self.assertTrue(set(lines[1]) <= {"-", "+"})

    def test_to_text_names_every_opcode(self) -> None:
        text = self.trace.to_text()
        for opcode in ("kernel", "load", "add", "mul", "store"):
            self.assertIn(opcode, text)

    def test_to_text_fits_the_terminal(self) -> None:
        for line in self.trace.to_text().splitlines():
            self.assertLessEqual(len(line), MAX_WIDTH + 2)

    def test_to_text_is_ascii(self) -> None:
        self.trace.to_text().encode("ascii")

    def test_empty_trace_renders_a_placeholder(self) -> None:
        self.assertEqual(Trace().to_text(), "trace: no events")
        self.assertEqual(Trace().to_schedule_text(self.graph), "schedule: no events")

    def test_to_json_parses_back(self) -> None:
        payload = json.loads(self.trace.to_json())
        self.assertIsInstance(payload, list)
        self.assertEqual(len(payload), len(self.trace))
        self.assertEqual([entry["op_index"] for entry in payload], self.trace.op_indices())
        self.assertEqual(payload[3]["opcode"], "add")
        self.assertEqual(payload[3]["output"], "c")

    def test_to_json_is_stable(self) -> None:
        self.assertEqual(self.trace.to_json(), self.trace.to_json())

    def test_to_json_of_an_empty_trace(self) -> None:
        self.assertEqual(json.loads(Trace().to_json()), [])

    def test_to_schedule_text_shows_wavefronts(self) -> None:
        text = self.trace.to_schedule_text(self.graph)
        self.assertTrue(text)
        self.assertIn("step 0", text)
        self.assertIn("ops in parallel", text)
        self.assertIn("waits on", text)
        self.assertIn("makespan", text)
        self.assertIn("critical path", text)

    def test_to_schedule_text_groups_the_parallel_loads(self) -> None:
        text = self.trace.to_schedule_text(self.graph)
        step_lines = [line for line in text.splitlines() if line.startswith("step ")]
        self.assertEqual(len(step_lines), len(self.graph.levels()))
        self.assertIn("4 ops in parallel", text)
        for index in (1, 2, 4):
            self.assertEqual(self.graph.nodes[index].level, 0)

    def test_to_schedule_text_is_ascii(self) -> None:
        self.trace.to_schedule_text(self.graph).encode("ascii")

    def test_to_dot_is_well_formed(self) -> None:
        dot = self.trace.to_dot(self.graph)
        self.assertTrue(dot.startswith("digraph"))
        self.assertTrue(dot.rstrip().endswith("}"))
        for index in self.graph.nodes:
            self.assertIn(f"n{index} [label=", dot)
        expected = sum(len(node.preds) for node in self.graph.nodes.values())
        self.assertEqual(dot.count(" -> "), expected)

    def test_to_dot_quotes_are_balanced(self) -> None:
        for line in self.trace.to_dot(self.graph).splitlines():
            self.assertEqual(line.count('"') % 2, 0, line)

    def test_all_four_formats_are_non_empty(self) -> None:
        for text in (
            self.trace.to_text(),
            self.trace.to_json(),
            self.trace.to_schedule_text(self.graph),
            self.trace.to_dot(self.graph),
        ):
            self.assertTrue(text.strip())


class RecorderTests(unittest.TestCase):
    """TraceRecorder's own surface."""

    def test_starts_empty(self) -> None:
        recorder = TraceRecorder()
        self.assertEqual(len(recorder.trace), 0)

    def test_capture_values_off_by_default(self) -> None:
        recorder = TraceRecorder()
        recorder.record(Op(0, "assign", {"out": "a", "value": "1"}), None, Tile.scalar(1.0))
        self.assertEqual(recorder.values, {})

    def test_capture_values_records_nested_data(self) -> None:
        recorder = TraceRecorder(capture_values=True)
        tile = Tile.from_flat([1.0, 2.0], (2,), "f32")
        recorder.record(Op(0, "assign", {"out": "a", "value": "v"}), None, tile)
        self.assertEqual(recorder.value_at(0), [1.0, 2.0])
        self.assertEqual(recorder.values_for(0), [[1.0, 2.0]])

    def test_capture_values_truncates_large_tiles(self) -> None:
        recorder = TraceRecorder(capture_values=True)
        recorder.record(Op(0, "fill", {"out": "a"}), None, Tile.zeros(1024, "f32"))
        self.assertEqual(len(recorder.value_at(0)), 256)
        self.assertIn("captured", recorder.trace.events[0].note)

    def test_reset(self) -> None:
        recorder = TraceRecorder(capture_values=True)
        recorder.record(Op(0, "assign", {"out": "a"}), None, 1)
        recorder.reset()
        self.assertEqual(len(recorder.trace), 0)
        self.assertEqual(recorder.values, {})

    def test_supplied_recorder_is_used(self) -> None:
        recorder = TraceRecorder(capture_values=True)
        program = parse_lineir(HEADLINE)
        result = Interpreter(program, recorder=recorder).run(**inputs())
        self.assertIs(result.trace, recorder.trace)
        self.assertTrue(recorder.values)


class TraceContainerTests(unittest.TestCase):
    """Trace behaves like the ordered container it is."""

    def test_add_and_iterate(self) -> None:
        trace = Trace()
        trace.add(TraceEvent(seq=0, op_index=0, opcode="add"))
        trace.add(TraceEvent(seq=1, op_index=1, opcode="mul"))
        self.assertEqual(len(trace), 2)
        self.assertEqual([event.opcode for event in trace], ["add", "mul"])
        self.assertEqual(trace.op_indices(), [0, 1])

    def test_constructor_copies_the_list(self) -> None:
        events = [TraceEvent(seq=0, op_index=0, opcode="add")]
        trace = Trace(events)
        events.append(TraceEvent(seq=1, op_index=1, opcode="mul"))
        self.assertEqual(len(trace), 1)


if __name__ == "__main__":
    unittest.main()
