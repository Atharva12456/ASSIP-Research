from __future__ import annotations

import unittest

from tile_interp.interpreter import Interpreter
from tile_interp.lineir import parse_lineir
from tile_interp.scheduler import DependencyGraph, Schedule, schedule
from tile_interp.values import Tile

HEADLINE = """# unified-tile-ir line-format v0
# source_lang=triton
# source=examples/headline.py
0000 | kernel | name=headline params="A,B,C,D,E"
0001 | load   | out=a buf=A
0002 | load   | out=b buf=B
0003 | add    | out=c lhs=a rhs=b
0004 | store  | buf=C value=c
0005 | load   | out=d buf=D
0006 | mul    | out=e lhs=c rhs=d
0007 | store  | buf=E value=e"""

DEMO = """0000 | load  | out=a buf=A
0001 | load  | out=b buf=B
0002 | add   | out=c lhs=a rhs=b
0003 | store | buf=C value=c
0004 | load  | out=d buf=D"""


def graph_of(text: str) -> DependencyGraph:
    """Build a dependency graph from line-format text."""
    return DependencyGraph.build(parse_lineir(text))


class DemoWavefrontTests(unittest.TestCase):
    """load A and load B issue together, then compute C; the trailing load D hoists up."""

    def setUp(self) -> None:
        self.graph = graph_of(DEMO)

    def test_levels_are_exactly_the_documented_wavefronts(self) -> None:
        self.assertEqual(self.graph.levels(), [[0, 1, 4], [2], [3]])

    def test_both_loads_are_at_step_zero(self) -> None:
        self.assertEqual(self.graph.nodes[0].level, 0)
        self.assertEqual(self.graph.nodes[1].level, 0)

    def test_the_unrelated_trailing_load_joins_step_zero(self) -> None:
        self.assertEqual(self.graph.nodes[4].level, 0)
        self.assertEqual(self.graph.nodes[4].preds, set())

    def test_the_add_waits_for_exactly_the_two_loads(self) -> None:
        self.assertEqual(sorted(self.graph.nodes[2].preds), [0, 1])

    def test_the_store_waits_for_the_add(self) -> None:
        self.assertEqual(sorted(self.graph.nodes[3].preds), [2])

    def test_the_schedule_reports_a_real_speedup(self) -> None:
        plan = schedule(parse_lineir(DEMO))
        self.assertEqual(plan.levels, [[0, 1, 4], [2], [3]])
        self.assertEqual(plan.widest_step, 3)
        self.assertLess(plan.makespan, plan.sequential_cost)
        self.assertGreater(plan.speedup, 1.5)


class HazardEdgesAreLoadBearingTests(unittest.TestCase):
    """A missing WAR or WAW edge would let a reordering compute the wrong answer."""

    WAR = (
        "0000 | assign | out=a value=1\n"
        "0001 | add    | out=b lhs=a rhs=10\n"
        "0002 | assign | out=a value=99"
    )

    WAW = (
        "0000 | assign | out=v value=1.0\n"
        "0001 | assign | out=w value=2.0\n"
        "0002 | store  | buf=C value=v\n"
        "0003 | store  | buf=C value=w"
    )

    def test_war_edge_exists_and_the_illegal_order_really_differs(self) -> None:
        graph = graph_of(self.WAR)
        self.assertIn(1, graph.nodes[2].preds)
        self.assertFalse(graph.is_valid_order([0, 2, 1]))
        program = parse_lineir(self.WAR)
        good = Interpreter(program).run().env["b"]
        bad = Interpreter(program).run_in_order([0, 2, 1]).env["b"]
        self.assertEqual(good, 11)
        self.assertEqual(bad, 109)
        self.assertNotEqual(good, bad)

    def test_waw_edge_exists_and_the_illegal_order_really_differs(self) -> None:
        graph = graph_of(self.WAW)
        self.assertIn(2, graph.nodes[3].preds)
        self.assertFalse(graph.is_valid_order([0, 1, 3, 2]))
        program = parse_lineir(self.WAW)
        seeded = {"C": Tile.zeros(1, "f32")}
        good = Interpreter(program).run(**seeded).memory.buffer("C").tile.to_nested()
        bad = (
            Interpreter(program)
            .run_in_order([0, 1, 3, 2], **{"C": Tile.zeros(1, "f32")})
            .memory.buffer("C")
            .tile.to_nested()
        )
        self.assertEqual(good, [2.0])
        self.assertEqual(bad, [1.0])

    def test_every_sampled_order_reproduces_the_sequential_answer(self) -> None:
        for source in (self.WAR, self.WAW):
            with self.subTest(source=source.splitlines()[0]):
                program = parse_lineir(source)
                graph = DependencyGraph.build(program)
                base = Interpreter(program).run(C=Tile.zeros(1, "f32")).memory.snapshot()
                for order in graph.topological_orders(24):
                    moved = Interpreter(program).run_in_order(
                        order, C=Tile.zeros(1, "f32")
                    )
                    self.assertEqual(moved.memory.snapshot(), base, order)


class HeadlineWavefrontTests(unittest.TestCase):
    """The demonstration program: load A, load B and load D all issue together."""

    def setUp(self) -> None:
        self.graph = graph_of(HEADLINE)

    def test_level_assignment_is_exact(self) -> None:
        self.assertEqual(self.graph.levels(), [[0, 1, 2, 5], [3], [4, 6], [7]])

    def test_the_kernel_declaration_orders_nothing(self) -> None:
        self.assertFalse(self.graph.nodes[0].effects.is_barrier)
        self.assertEqual(self.graph.nodes[0].succs, set())
        self.assertEqual(self.graph.nodes[1].preds, set())

    def test_the_three_loads_share_one_level(self) -> None:
        levels = {index: node.level for index, node in self.graph.nodes.items()}
        self.assertEqual(levels[1], levels[2])
        self.assertEqual(levels[1], levels[5])

    def test_the_late_load_is_hoisted_above_the_add(self) -> None:
        self.assertLess(self.graph.nodes[5].level, self.graph.nodes[3].level)

    def test_store_c_runs_beside_the_multiply(self) -> None:
        self.assertEqual(self.graph.nodes[4].level, self.graph.nodes[6].level)

    def test_add_waits_on_both_loads(self) -> None:
        self.assertIn(1, self.graph.nodes[3].preds)
        self.assertIn(2, self.graph.nodes[3].preds)

    def test_multiply_waits_on_the_add_and_the_late_load(self) -> None:
        self.assertIn(3, self.graph.nodes[6].preds)
        self.assertIn(5, self.graph.nodes[6].preds)

    def test_loads_do_not_depend_on_each_other(self) -> None:
        self.assertNotIn(2, self.graph.nodes[1].preds)
        self.assertNotIn(1, self.graph.nodes[2].preds)
        self.assertNotIn(5, self.graph.nodes[1].preds)

    def test_edges_run_forward(self) -> None:
        for index, node in self.graph.nodes.items():
            for pred in node.preds:
                self.assertLess(pred, index)

    def test_edges_are_symmetric(self) -> None:
        for index, node in self.graph.nodes.items():
            for pred in node.preds:
                self.assertIn(index, self.graph.nodes[pred].succs)
            for succ in node.succs:
                self.assertIn(index, self.graph.nodes[succ].preds)

    def test_schedule_beats_the_sequential_cost(self) -> None:
        plan = schedule(parse_lineir(HEADLINE))
        self.assertIsInstance(plan, Schedule)
        self.assertEqual(plan.levels, [[0, 1, 2, 5], [3], [4, 6], [7]])
        self.assertLess(plan.makespan, plan.sequential_cost)
        self.assertGreater(plan.speedup, 1.0)
        self.assertEqual(plan.widest_step, 4)

    def test_critical_path_is_a_real_chain(self) -> None:
        path = self.graph.critical_path()
        self.assertGreater(len(path), 1)
        for earlier, later in zip(path, path[1:]):
            self.assertIn(later, self.graph.nodes[earlier].succs)

    def test_index_order_is_a_valid_order(self) -> None:
        self.assertTrue(self.graph.is_valid_order(sorted(self.graph.nodes)))

    def test_flattened_wavefronts_are_a_valid_order(self) -> None:
        flat = [index for level in self.graph.levels() for index in level]
        self.assertTrue(self.graph.is_valid_order(flat))


class DataDependenceTests(unittest.TestCase):
    """RAW, WAR and WAW on environment names, each isolated."""

    def test_raw_edge(self) -> None:
        graph = graph_of(
            "0000 | assign | out=a value=1\n0001 | add | out=b lhs=a rhs=1"
        )
        self.assertEqual(sorted(graph.nodes[1].preds), [0])
        self.assertEqual(graph.levels(), [[0], [1]])

    def test_war_edge_alone(self) -> None:
        graph = graph_of(
            "0000 | add    | out=b lhs=a rhs=1\n0001 | assign | out=a value=2"
        )
        self.assertIn(0, graph.nodes[1].preds)
        self.assertEqual(graph.levels(), [[0], [1]])
        self.assertFalse(graph.is_valid_order([1, 0]))

    def test_waw_edge_alone(self) -> None:
        graph = graph_of(
            "0000 | assign | out=a value=1\n0001 | assign | out=a value=2"
        )
        self.assertIn(0, graph.nodes[1].preds)
        self.assertFalse(graph.is_valid_order([1, 0]))

    def test_war_then_raw_chain_serialises(self) -> None:
        graph = graph_of(
            "0000 | load   | out=a buf=A\n"
            "0001 | add    | out=b lhs=a rhs=a\n"
            "0002 | assign | out=a value=5\n"
            "0003 | assign | out=a value=6"
        )
        self.assertEqual(graph.levels(), [[0], [1], [2], [3]])
        self.assertIn(1, graph.nodes[2].preds)
        self.assertIn(2, graph.nodes[3].preds)

    def test_in_place_update_does_not_self_link(self) -> None:
        graph = graph_of(
            "0000 | assign | out=t value=0\n"
            "0001 | add    | out=t lhs=t rhs=1 inplace=true"
        )
        self.assertNotIn(1, graph.nodes[1].preds)
        self.assertEqual(sorted(graph.nodes[1].preds), [0])

    def test_independent_ops_share_a_level(self) -> None:
        graph = graph_of(
            "0000 | assign | out=a value=1\n"
            "0001 | assign | out=b value=2\n"
            "0002 | assign | out=c value=3"
        )
        self.assertEqual(graph.levels(), [[0, 1, 2]])


class BufferDependenceTests(unittest.TestCase):
    """RAW, WAR and WAW on buffer names for load and store."""

    def test_store_then_load_is_raw(self) -> None:
        graph = graph_of("0000 | store | buf=A value=z\n0001 | load | out=v buf=A")
        self.assertIn(0, graph.nodes[1].preds)

    def test_load_then_store_is_war(self) -> None:
        graph = graph_of("0000 | load | out=a buf=A\n0001 | store | buf=A value=z")
        self.assertIn(0, graph.nodes[1].preds)
        self.assertFalse(graph.is_valid_order([1, 0]))

    def test_store_then_store_is_waw(self) -> None:
        graph = graph_of("0000 | store | buf=A value=z\n0001 | store | buf=A value=w")
        self.assertIn(0, graph.nodes[1].preds)
        self.assertFalse(graph.is_valid_order([1, 0]))

    def test_full_buffer_chain_serialises(self) -> None:
        graph = graph_of(
            "0000 | load  | out=a buf=A\n"
            "0001 | store | buf=A value=a\n"
            "0002 | store | buf=A value=a\n"
            "0003 | load  | out=c buf=A"
        )
        self.assertEqual(graph.levels(), [[0], [1], [2], [3]])

    def test_distinct_buffers_stay_parallel(self) -> None:
        graph = graph_of(
            "0000 | load | out=a buf=A\n"
            "0001 | load | out=b buf=B\n"
            "0002 | load | out=c buf=C"
        )
        self.assertEqual(graph.levels(), [[0, 1, 2]])

    def test_two_loads_of_one_buffer_stay_parallel(self) -> None:
        graph = graph_of("0000 | load | out=a buf=A\n0001 | load | out=b buf=A")
        self.assertEqual(graph.levels(), [[0, 1]])

    def test_pointer_form_names_the_buffer(self) -> None:
        graph = graph_of(
            '0000 | load  | out=a ptr="A + offsets"\n'
            '0001 | store | ptr="A + offsets" value=a'
        )
        self.assertEqual(graph.nodes[0].effects.mem_reads, ["A"])
        self.assertEqual(graph.nodes[1].effects.mem_writes, ["A"])
        self.assertIn(0, graph.nodes[1].preds)


class BarrierTests(unittest.TestCase):
    """Control-flow ops order everything around them."""

    def test_nothing_hoists_across_an_if(self) -> None:
        graph = graph_of(
            "0000 | assign | out=t value=0\n"
            "0001 | if     | cond=c\n"
            "0002 | assign | out=u value=1\n"
            "0003 | endif  |\n"
            "0004 | assign | out=w value=2"
        )
        self.assertEqual(graph.levels(), [[0], [1], [2], [3], [4]])

    def test_body_of_a_loop_still_finds_parallelism(self) -> None:
        graph = graph_of(
            "0000 | for   | target=i iter=range(2)\n"
            "0001 | load  | out=a buf=A\n"
            "0002 | load  | out=b buf=B\n"
            "0003 | add   | out=c lhs=a rhs=b\n"
            "0004 | endfor |"
        )
        self.assertEqual(graph.levels(), [[0], [1, 2], [3], [4]])

    def test_barrier_ops_are_flagged(self) -> None:
        graph = graph_of(
            "0000 | if       | cond=c\n"
            "0001 | else     |\n"
            "0002 | endif    |\n"
            "0003 | for      | target=i iter=range(1)\n"
            "0004 | endfor   |\n"
            "0005 | while    | cond=c\n"
            "0006 | endwhile |\n"
            "0007 | return   |"
        )
        for index in sorted(graph.nodes):
            with self.subTest(index=index):
                self.assertTrue(graph.nodes[index].effects.is_barrier)

    def test_arithmetic_is_not_a_barrier(self) -> None:
        graph = graph_of("0000 | add | out=c lhs=a rhs=b")
        self.assertFalse(graph.nodes[0].effects.is_barrier)

    def test_a_kernel_declaration_is_not_a_barrier(self) -> None:
        graph = graph_of(
            "0000 | kernel | name=first params=\n"
            "0001 | assign | out=a value=1\n"
            "0002 | kernel | name=second params=\n"
            "0003 | assign | out=b value=2"
        )
        self.assertFalse(graph.nodes[0].effects.is_barrier)
        self.assertFalse(graph.nodes[2].effects.is_barrier)
        self.assertEqual(graph.levels(), [[0, 1, 2, 3]])

    def test_a_real_barrier_still_separates_the_ops_around_it(self) -> None:
        graph = graph_of(
            "0000 | assign | out=a value=1\n"
            "0001 | return | value=a\n"
            "0002 | assign | out=b value=2"
        )
        self.assertTrue(graph.nodes[1].effects.is_barrier)
        self.assertEqual(graph.levels(), [[0], [1], [2]])


class OrderValidationTests(unittest.TestCase):
    """is_valid_order is the gate every sampled permutation must pass."""

    def setUp(self) -> None:
        self.graph = graph_of(HEADLINE)

    def test_accepts_index_order(self) -> None:
        self.assertTrue(self.graph.is_valid_order([0, 1, 2, 3, 4, 5, 6, 7]))

    def test_accepts_a_legal_reordering(self) -> None:
        self.assertTrue(self.graph.is_valid_order([0, 5, 2, 1, 3, 6, 4, 7]))

    def test_rejects_a_predecessor_placed_late(self) -> None:
        self.assertFalse(self.graph.is_valid_order([0, 1, 3, 2, 4, 5, 6, 7]))

    def test_rejects_a_reversed_order(self) -> None:
        self.assertFalse(self.graph.is_valid_order([7, 6, 5, 4, 3, 2, 1, 0]))

    def test_rejects_a_short_order(self) -> None:
        self.assertFalse(self.graph.is_valid_order([0, 1, 2, 3]))

    def test_rejects_duplicates(self) -> None:
        self.assertFalse(self.graph.is_valid_order([0, 1, 1, 2, 3, 4, 5, 6]))

    def test_rejects_unknown_indices(self) -> None:
        self.assertFalse(self.graph.is_valid_order([0, 1, 2, 3, 4, 5, 6, 99]))

    def test_rejects_an_empty_order(self) -> None:
        self.assertFalse(self.graph.is_valid_order([]))


class TopologicalOrderTests(unittest.TestCase):
    """Sampled orders must be distinct, valid, and deterministic."""

    def setUp(self) -> None:
        self.graph = graph_of(HEADLINE)

    def test_orders_are_distinct(self) -> None:
        orders = self.graph.topological_orders(16)
        self.assertGreater(len(orders), 1)
        unique = {tuple(order) for order in orders}
        self.assertEqual(len(unique), len(orders))

    def test_every_order_is_valid(self) -> None:
        for order in self.graph.topological_orders(16):
            with self.subTest(order=tuple(order)):
                self.assertTrue(self.graph.is_valid_order(order))

    def test_index_order_comes_first(self) -> None:
        self.assertEqual(self.graph.topological_orders(4)[0], sorted(self.graph.nodes))

    def test_generation_is_deterministic(self) -> None:
        again = graph_of(HEADLINE)
        self.assertEqual(self.graph.topological_orders(16), again.topological_orders(16))

    def test_respects_the_limit(self) -> None:
        self.assertLessEqual(len(self.graph.topological_orders(3)), 3)
        self.assertEqual(self.graph.topological_orders(0), [])

    def test_a_serial_chain_has_exactly_one_order(self) -> None:
        chain = graph_of(
            "0000 | assign | out=a value=1\n"
            "0001 | add    | out=b lhs=a rhs=1\n"
            "0002 | add    | out=c lhs=b rhs=1"
        )
        self.assertEqual(chain.topological_orders(20), [[0, 1, 2]])


class ScheduleTests(unittest.TestCase):
    """schedule() wraps the graph with the cost-model numbers."""

    def test_fields(self) -> None:
        plan = schedule(parse_lineir(HEADLINE))
        self.assertEqual(
            sorted(index for step in plan.levels for index in step), list(range(8))
        )
        self.assertGreater(plan.sequential_cost, 0)
        self.assertGreater(plan.makespan, 0)
        self.assertAlmostEqual(plan.speedup, plan.sequential_cost / plan.makespan)
        self.assertIsNone(plan.max_parallel)
        self.assertEqual(plan.deferred, [])

    def test_makespan_matches_the_graph(self) -> None:
        program = parse_lineir(HEADLINE)
        self.assertEqual(schedule(program).makespan, DependencyGraph.build(program).makespan())

    def test_max_parallel_caps_each_step(self) -> None:
        plan = schedule(parse_lineir(HEADLINE), max_parallel=2)
        self.assertTrue(all(len(step) <= 2 for step in plan.levels))
        self.assertEqual(plan.max_parallel, 2)
        self.assertTrue(plan.deferred)

    def test_capped_schedule_is_still_a_valid_order(self) -> None:
        program = parse_lineir(HEADLINE)
        graph = DependencyGraph.build(program)
        plan = schedule(program, max_parallel=2)
        flat = [index for step in plan.levels for index in step]
        self.assertTrue(graph.is_valid_order(flat))

    def test_max_parallel_one_is_fully_serial(self) -> None:
        plan = schedule(parse_lineir(HEADLINE), max_parallel=1)
        self.assertTrue(all(len(step) == 1 for step in plan.levels))
        self.assertEqual(plan.makespan, plan.sequential_cost)
        self.assertAlmostEqual(plan.speedup, 1.0)

    def test_zero_cap_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            schedule(parse_lineir(HEADLINE), max_parallel=0)

    def test_empty_program(self) -> None:
        plan = schedule(parse_lineir("# source_lang=triton"))
        self.assertEqual(plan.levels, [])
        self.assertEqual(plan.makespan, 0)
        self.assertEqual(plan.critical_path, [])

    def test_to_text_mentions_every_step(self) -> None:
        text = schedule(parse_lineir(HEADLINE)).to_text()
        self.assertIn("step 0", text)
        self.assertIn("step 3", text)
        self.assertIn("speedup", text)
        self.assertIn("critical path", text)


class RobustnessTests(unittest.TestCase):
    """Unknown opcodes and empty programs must not crash the scheduler."""

    def test_unknown_opcode_degrades_to_a_barrier(self) -> None:
        graph = graph_of(
            "0000 | assign     | out=a value=1\n"
            "0001 | frobnicate | out=q\n"
            "0002 | assign     | out=b value=2"
        )
        self.assertTrue(graph.nodes[1].effects.is_barrier)
        self.assertEqual(graph.levels(), [[0], [1], [2]])

    def test_empty_graph(self) -> None:
        graph = graph_of("")
        self.assertEqual(graph.levels(), [])
        self.assertEqual(graph.makespan(), 0)
        self.assertEqual(graph.critical_path(), [])
        self.assertEqual(graph.sequential_cost(), 0)

    def test_single_op_graph(self) -> None:
        graph = graph_of("0000 | assign | out=a value=1")
        self.assertEqual(graph.levels(), [[0]])
        self.assertEqual(graph.critical_path(), [0])


class DotTests(unittest.TestCase):
    """to_dot emits a well-formed graphviz digraph."""

    def setUp(self) -> None:
        self.dot = graph_of(HEADLINE).to_dot()

    def test_wrapper(self) -> None:
        self.assertTrue(self.dot.startswith("digraph"))
        self.assertTrue(self.dot.rstrip().endswith("}"))

    def test_one_node_per_op(self) -> None:
        for index in range(8):
            self.assertIn(f"n{index:04d} [label=", self.dot)

    def test_edge_count_matches_the_graph(self) -> None:
        graph = graph_of(HEADLINE)
        expected = sum(len(node.succs) for node in graph.nodes.values())
        self.assertEqual(self.dot.count(" -> "), expected)

    def test_rank_groups_for_parallel_steps(self) -> None:
        self.assertIn("rank=same", self.dot)

    def test_quotes_are_balanced(self) -> None:
        for line in self.dot.splitlines():
            self.assertEqual(line.count('"') % 2, 0, line)


if __name__ == "__main__":
    unittest.main()
