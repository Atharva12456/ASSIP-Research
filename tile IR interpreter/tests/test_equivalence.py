from __future__ import annotations

import unittest
from dataclasses import replace

from reference.kernels import EXAMPLES, Example, example, example_names
from tile_interp.interpreter import Interpreter
from tile_interp.ir import Op, Program
from tile_interp.lineir import parse_lineir, parse_lineir_file, to_lineir
from tile_interp.scheduler import DependencyGraph
from tile_interp.values import Tile
from tile_interp.verify import (
    CHECK_NAMES,
    Check,
    VerificationError,
    VerificationReport,
    verify_kernel,
)


def load(name: str) -> Program:
    """Parse one shipped example program."""
    return parse_lineir_file(example(name).path)


def report_for(name: str, **overrides: object) -> VerificationReport:
    """Verify one shipped example against its reference kernel."""
    item = example(name)
    return verify_kernel(
        load(name),
        item.reference,
        item.seed(),
        outputs=list(item.outputs),
        **overrides,  # type: ignore[arg-type]
    )


def mutate(program: Program, index: int, opcode: str) -> Program:
    """A copy of program with one op's opcode replaced."""
    ops = [
        Op(op.index, opcode if op.index == index else op.opcode, dict(op.attrs))
        for op in program.ops
    ]
    return Program(program.source_lang, program.source_name, ops, program.max_ops)


def retarget(program: Program, index: int, key: str, value: str) -> Program:
    """A copy of program with one attribute of one op replaced."""
    ops = []
    for op in program.ops:
        attrs = dict(op.attrs)
        if op.index == index:
            attrs[key] = value
        ops.append(Op(op.index, op.opcode, attrs))
    return Program(program.source_lang, program.source_name, ops, program.max_ops)


class ExampleInventoryTests(unittest.TestCase):
    """Every declared example must ship a parseable .lineir file."""

    def test_examples_exist(self) -> None:
        self.assertTrue(example_names())

    def test_every_example_file_is_present(self) -> None:
        for name in example_names():
            with self.subTest(name=name):
                self.assertTrue(example(name).path.is_file(), example(name).path)

    def test_every_example_parses(self) -> None:
        for name in example_names():
            with self.subTest(name=name):
                program = load(name)
                self.assertGreater(len(program), 0)

    def test_kernel_names_match_the_file_names(self) -> None:
        for name in example_names():
            with self.subTest(name=name):
                self.assertEqual(load(name).kernel_name(), name)

    def test_unknown_example_raises(self) -> None:
        with self.assertRaises(KeyError):
            example("no_such_kernel")


class ExampleVerificationTests(unittest.TestCase):
    """The headline deliverable: every example verifies against its reference kernel."""

    def test_every_example_verifies(self) -> None:
        for name in example_names():
            with self.subTest(name=name):
                report = report_for(name)
                self.assertTrue(
                    report.passed,
                    f"{name} failed:\n{report.to_text()}",
                )

    def test_every_check_runs_for_every_example(self) -> None:
        for name in example_names():
            with self.subTest(name=name):
                report = report_for(name)
                self.assertEqual([check.name for check in report.checks], list(CHECK_NAMES))

    def test_report_names_the_kernel(self) -> None:
        for name in example_names():
            with self.subTest(name=name):
                self.assertEqual(report_for(name).kernel, name)

    def test_report_text_is_renderable(self) -> None:
        text = report_for("load_compute_store").to_text()
        self.assertIn("load_compute_store", text)
        self.assertIn("EQUIVALENT", text)
        for name in CHECK_NAMES:
            self.assertIn(name, text)

    def test_no_failures_recorded(self) -> None:
        for name in example_names():
            with self.subTest(name=name):
                self.assertEqual(report_for(name).failures(), [])


class ReferenceOutputTests(unittest.TestCase):
    """The interpreter's buffers really do equal the pure-Python reference."""

    def test_outputs_match_element_by_element(self) -> None:
        for name in example_names():
            item = example(name)
            with self.subTest(name=name):
                result = Interpreter(load(name)).run(**item.seed())
                expected = item.expected()
                for key in item.outputs:
                    got = result.memory.buffer(key).tile
                    want = expected[key]
                    self.assertEqual(got.shape, want.shape, f"{name}/{key}")
                    self.assertTrue(got.allclose(want), f"{name}/{key}: {got!r} != {want!r}")

    def test_vector_add_by_hand(self) -> None:
        item = example("vector_add")
        result = Interpreter(load("vector_add")).run(**item.seed())
        self.assertEqual(
            result.memory.buffer("out_ptr").tile.to_nested(),
            [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
        )

    def test_matmul_by_hand(self) -> None:
        item = example("matmul_tile")
        result = Interpreter(load("matmul_tile")).run(**item.seed())
        self.assertEqual(
            result.memory.buffer("C").tile.to_nested(), [[58.0, 64.0], [139.0, 154.0]]
        )

    def test_inputs_are_not_mutated_by_verification(self) -> None:
        item = example("load_compute_store")
        seeds = item.seed()
        before = {key: tile.copy() for key, tile in seeds.items()}
        verify_kernel(
            load("load_compute_store"), item.reference, seeds, outputs=list(item.outputs)
        )
        for key, tile in before.items():
            self.assertEqual(seeds[key], tile, key)


class ScheduleEquivalenceTests(unittest.TestCase):
    """Reordering along the dependency DAG must not change any result."""

    def straight_line(self) -> list[str]:
        return [
            name
            for name in example_names()
            if not any(op.opcode in {"if", "for", "while"} for op in load(name).ops)
        ]

    def test_there_are_straight_line_examples(self) -> None:
        self.assertGreater(len(self.straight_line()), 1)

    def test_wavefront_order_reproduces_the_sequential_run(self) -> None:
        for name in self.straight_line():
            item = example(name)
            with self.subTest(name=name):
                program = load(name)
                graph = DependencyGraph.build(program)
                flat = [index for level in graph.levels() for index in level]
                self.assertTrue(graph.is_valid_order(flat))
                base = Interpreter(program).run(**item.seed()).memory.snapshot()
                moved = Interpreter(program).run_in_order(flat, **item.seed())
                self.assertEqual(moved.memory.snapshot(), base)

    def test_sampled_permutations_reproduce_the_sequential_run(self) -> None:
        for name in self.straight_line():
            item = example(name)
            with self.subTest(name=name):
                program = load(name)
                graph = DependencyGraph.build(program)
                base = Interpreter(program).run(**item.seed()).memory.snapshot()
                orders = graph.topological_orders(16)
                self.assertTrue(orders)
                for order in orders:
                    moved = Interpreter(program).run_in_order(order, **item.seed())
                    self.assertEqual(moved.memory.snapshot(), base, order)

    def test_the_long_chain_actually_has_parallelism(self) -> None:
        graph = DependencyGraph.build(load("long_chain"))
        widest = max(len(level) for level in graph.levels())
        self.assertGreater(widest, 1)
        self.assertLess(len(graph.levels()), len(graph.nodes))


class NegativeVerificationTests(unittest.TestCase):
    """Without these, a green suite proves nothing."""

    def test_mutating_an_opcode_is_reported(self) -> None:
        item = example("load_compute_store")
        program = load("load_compute_store")
        broken = mutate(program, 3, "sub")
        report = verify_kernel(
            broken, item.reference, item.seed(), outputs=list(item.outputs)
        )
        self.assertFalse(report.passed, report.to_text())
        self.assertFalse(report.check("reference-match").passed)
        self.assertIn("NOT EQUIVALENT", report.to_text())

    def test_the_failure_detail_names_the_buffer(self) -> None:
        item = example("load_compute_store")
        report = verify_kernel(
            mutate(load("load_compute_store"), 3, "sub"),
            item.reference,
            item.seed(),
            outputs=list(item.outputs),
        )
        self.assertIn("C", report.check("reference-match").detail)

    def test_mutating_every_arithmetic_op_is_caught(self) -> None:
        item = example("load_compute_store")
        program = load("load_compute_store")
        for index, replacement in ((3, "sub"), (3, "mul"), (6, "add"), (6, "div")):
            with self.subTest(index=index, opcode=replacement):
                report = verify_kernel(
                    mutate(program, index, replacement),
                    item.reference,
                    item.seed(),
                    outputs=list(item.outputs),
                )
                self.assertFalse(report.passed, report.to_text())

    def test_mutating_a_store_target_is_reported(self) -> None:
        item = example("load_compute_store")
        broken = retarget(load("load_compute_store"), 4, "value", "a")
        report = verify_kernel(
            broken, item.reference, item.seed(), outputs=list(item.outputs)
        )
        self.assertFalse(report.passed, report.to_text())

    def test_a_wrong_reference_is_reported(self) -> None:
        item = example("vector_add")

        def wrong(inputs: dict[str, Tile]) -> dict[str, Tile]:
            out = inputs["out_ptr"]
            return {"out_ptr": Tile.full(out.shape, 1.0, out.dtype)}

        report = verify_kernel(
            load("vector_add"), wrong, item.seed(), outputs=["out_ptr"]
        )
        self.assertFalse(report.passed, report.to_text())
        self.assertFalse(report.check("reference-match").passed)

    def test_mutation_in_the_control_flow_example_is_caught(self) -> None:
        item = example("control_flow")
        program = load("control_flow")
        adds = [op.index for op in program.ops if op.opcode == "add"]
        self.assertTrue(adds)
        report = verify_kernel(
            mutate(program, adds[-1], "sub"),
            item.reference,
            item.seed(),
            outputs=list(item.outputs),
        )
        self.assertFalse(report.passed, report.to_text())

    def test_a_kernel_that_writes_nothing_is_reported(self) -> None:
        item = example("vector_add")
        program = load("vector_add")
        stripped = Program(
            program.source_lang,
            program.source_name,
            [Op(op.index, op.opcode, dict(op.attrs)) for op in program.ops if op.opcode != "store"],
            program.max_ops,
        )
        report = verify_kernel(
            stripped, item.reference, item.seed(), outputs=["out_ptr"]
        )
        self.assertFalse(report.passed, report.to_text())

    def test_a_tiny_tolerance_change_is_still_caught(self) -> None:
        item = example("vector_add")

        def nudged(inputs: dict[str, Tile]) -> dict[str, Tile]:
            expected = item.reference(inputs)["out_ptr"]
            data = [value + 1e-3 for value in expected.data]
            return {"out_ptr": Tile.from_flat(data, expected.shape, expected.dtype)}

        report = verify_kernel(
            load("vector_add"), nudged, item.seed(), outputs=["out_ptr"]
        )
        self.assertFalse(report.passed, report.to_text())


class HarnessGuardTests(unittest.TestCase):
    """The harness refuses to report a vacuous pass."""

    def test_empty_program_raises(self) -> None:
        item = example("vector_add")
        empty = Program("triton", "empty.lineir", [], 220)
        with self.assertRaises(VerificationError):
            verify_kernel(empty, item.reference, item.seed(), outputs=["out_ptr"])

    def test_no_outputs_is_a_failing_check(self) -> None:
        item = example("vector_add")
        report = verify_kernel(load("vector_add"), item.reference, item.seed(), outputs=[])
        self.assertFalse(report.check("reference-match").passed)
        self.assertFalse(report.passed)

    def test_zero_permutations_is_a_failing_check(self) -> None:
        report = report_for("vector_add", permutations=0)
        self.assertFalse(report.check("permutation-sound").passed)
        self.assertFalse(report.passed)

    def test_a_non_dict_reference_raises(self) -> None:
        item = example("vector_add")
        with self.assertRaises(VerificationError):
            verify_kernel(
                load("vector_add"),
                lambda inputs: [1, 2, 3],  # type: ignore[arg-type,return-value]
                item.seed(),
                outputs=["out_ptr"],
            )

    def test_report_without_checks_does_not_pass(self) -> None:
        self.assertFalse(VerificationReport("k", []).passed)

    def test_check_lookup(self) -> None:
        report = report_for("vector_add")
        self.assertIsInstance(report.check("roundtrip"), Check)
        with self.assertRaises(KeyError):
            report.check("no-such-check")


class RoundTripCheckTests(unittest.TestCase):
    """The roundtrip check must be sensitive to the text format, not a rubber stamp."""

    def test_every_example_round_trips(self) -> None:
        for name in example_names():
            with self.subTest(name=name):
                program = load(name)
                self.assertEqual(parse_lineir(to_lineir(program), program.source_name), program)
                self.assertTrue(report_for(name).check("roundtrip").passed)

    def test_check_names_are_the_contract_names(self) -> None:
        self.assertEqual(
            list(CHECK_NAMES),
            [
                "reference-match",
                "memory-match",
                "schedule-sound",
                "permutation-sound",
                "trace-complete",
                "roundtrip",
            ],
        )


class ExampleDataclassTests(unittest.TestCase):
    """reference.kernels.Example is the table the CLI and the tests both read."""

    def test_seed_returns_fresh_tiles(self) -> None:
        item = example("vector_add")
        first = item.seed()
        first["x_ptr"].data[0] = 999.0
        self.assertNotEqual(item.seed()["x_ptr"].data[0], 999.0)

    def test_expected_matches_the_reference(self) -> None:
        item = example("vector_add")
        self.assertEqual(item.expected(), item.reference(item.seed()))

    def test_outputs_are_declared(self) -> None:
        for name, item in EXAMPLES.items():
            with self.subTest(name=name):
                self.assertIsInstance(item, Example)
                self.assertTrue(item.outputs)
                self.assertTrue(item.summary)

    def test_replace_keeps_the_dataclass_usable(self) -> None:
        item = replace(example("vector_add"), summary="edited")
        self.assertEqual(item.summary, "edited")
        self.assertEqual(item.name, "vector_add")


if __name__ == "__main__":
    unittest.main()
