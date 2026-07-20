from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from tile_interp import cli
from tile_interp.lineir import parse_lineir

EXAMPLES = cli.project_root() / "examples"
HEADLINE = EXAMPLES / "load_compute_store.lineir"

MALFORMED = "0000 | kernel | name=broken params=Q\n0001 | load | out=a buf=Q\n0002 | load | out=b\n"


def invoke(*argv: str) -> tuple[int, str, str]:
    """Run one command line, capturing its exit code, stdout and stderr."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class SubcommandSmokeTests(unittest.TestCase):
    """Every subcommand runs against every bundled example and exits zero."""

    def test_run_every_example(self) -> None:
        for path in cli.example_paths():
            with self.subTest(example=path.stem):
                code, out, _ = invoke("run", str(path))
                self.assertEqual(code, cli.EXIT_OK)
                self.assertIn("buffers", out)

    def test_trace_every_format(self) -> None:
        for fmt in ("text", "json", "schedule", "dot"):
            for path in cli.example_paths():
                with self.subTest(example=path.stem, format=fmt):
                    code, out, _ = invoke("trace", str(path), "--format", fmt)
                    self.assertEqual(code, cli.EXIT_OK)
                    self.assertTrue(out.strip())

    def test_trace_json_is_parseable(self) -> None:
        code, out, _ = invoke("trace", str(HEADLINE), "--format", "json")
        self.assertEqual(code, cli.EXIT_OK)
        events = json.loads(out)
        self.assertEqual([event["op_index"] for event in events], list(range(8)))

    def test_schedule_every_example(self) -> None:
        for path in cli.example_paths():
            with self.subTest(example=path.stem):
                code, out, _ = invoke("schedule", str(path))
                self.assertEqual(code, cli.EXIT_OK)
                self.assertIn("critical path", out)

    def test_ops_lists_every_opcode(self) -> None:
        from tile_interp.semantics import OPCODES

        code, out, _ = invoke("ops")
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn(f"{len(OPCODES)} opcodes", out)

    def test_verify_all_examples_passes(self) -> None:
        code, out, _ = invoke("verify")
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("PASS", out)


class HeadlineScheduleTests(unittest.TestCase):
    """The demo claim: independent loads issue together in the first wavefront."""

    def test_the_three_loads_share_step_zero(self) -> None:
        code, out, _ = invoke("schedule", str(HEADLINE))
        self.assertEqual(code, cli.EXIT_OK)
        first = next(line for line in out.splitlines() if line.startswith("step 0"))
        for index in ("0001", "0002", "0005"):
            self.assertIn(index, first)

    def test_the_dependent_add_is_strictly_later(self) -> None:
        _, out, _ = invoke("schedule", str(HEADLINE))
        steps = [line for line in out.splitlines() if line.startswith("step ")]
        self.assertIn("0003", steps[1])
        self.assertNotIn("0003", steps[0])


class ExitCodeTests(unittest.TestCase):
    """Usage problems exit 2, execution failures exit 1, and neither traces back."""

    def test_missing_file_is_a_usage_error(self) -> None:
        code, _, err = invoke("run", "no_such_file.lineir")
        self.assertEqual(code, cli.EXIT_USAGE)
        self.assertIn("no such file", err)

    def test_bad_json_input_is_a_usage_error(self) -> None:
        code, _, err = invoke("run", str(HEADLINE), "--input", "A=[1,2")
        self.assertEqual(code, cli.EXIT_USAGE)
        self.assertIn("not valid JSON", err)

    def test_bad_grid_is_a_usage_error(self) -> None:
        code, _, err = invoke("run", str(HEADLINE), "--grid", "a,b")
        self.assertEqual(code, cli.EXIT_USAGE)
        self.assertIn("--grid", err)

    def test_max_parallel_below_one_is_a_usage_error(self) -> None:
        code, _, err = invoke("schedule", str(HEADLINE), "--max-parallel", "0")
        self.assertEqual(code, cli.EXIT_USAGE)
        self.assertIn("--max-parallel", err)

    def test_no_command_prints_help(self) -> None:
        code, out, _ = invoke()
        self.assertEqual(code, cli.EXIT_USAGE)
        self.assertIn("usage", out.lower())

    def test_verify_of_a_mutated_program_fails(self) -> None:
        mutated = HEADLINE.read_text(encoding="utf-8").replace(
            "0003 | add          |", "0003 | sub          |", 1
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mutant.lineir"
            path.write_text(mutated, encoding="utf-8")
            code, out, _ = invoke("verify", str(path))
        self.assertEqual(code, cli.EXIT_FAIL)
        self.assertIn("NOT EQUIVALENT", out)


class DiagnosticTests(unittest.TestCase):
    """The unbound-buffer diagnostic must survive a program the effects table rejects."""

    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name) / "broken.lineir"
        self.path.write_text(MALFORMED, encoding="utf-8")

    def tearDown(self) -> None:
        self.folder.cleanup()

    def test_the_hint_survives_a_malformed_memory_op(self) -> None:
        code, _, err = invoke("run", str(self.path))
        self.assertEqual(code, cli.EXIT_FAIL)
        self.assertIn("unknown buffer 'Q'", err)
        self.assertIn("this program touches Q", err)

    def test_buffer_hint_does_not_raise_on_malformed_ops(self) -> None:
        program = parse_lineir(MALFORMED)
        self.assertIn("Q", cli._buffer_hint(program))

    def test_output_buffers_skips_malformed_ops(self) -> None:
        program = parse_lineir(MALFORMED + "0003 | store | buf=C value=a\n")
        self.assertEqual(cli._output_buffers(program), ["C"])


class InputBindingTests(unittest.TestCase):
    """--input accepts scalars, lists and the object form, with optional dtype pinning."""

    def test_list_input_builds_a_tile(self) -> None:
        name, value = cli.parse_input("A=[1.0, 2.0]")
        self.assertEqual(name, "A")
        self.assertEqual(value.to_nested(), [1.0, 2.0])

    def test_object_form_pins_shape_and_dtype(self) -> None:
        _, value = cli.parse_input('A={"shape": [4], "value": 0, "dtype": "f32"}')
        self.assertEqual(value.shape, (4,))
        self.assertEqual(value.dtype, "f32")

    def test_dtype_suffix_forces_a_scalar_tile(self) -> None:
        _, value = cli.parse_input("alpha:f64=2.5")
        self.assertEqual(value.dtype, "f64")
        self.assertEqual(value.item(), 2.5)

    def test_a_bare_scalar_stays_a_python_value(self) -> None:
        self.assertEqual(cli.parse_input("n=8"), ("n", 8))

    def test_a_non_identifier_name_is_rejected(self) -> None:
        with self.assertRaises(cli.CliError):
            cli.parse_input("not a name=1")

    def test_inputs_reach_the_kernel(self) -> None:
        code, out, _ = invoke(
            "run",
            str(EXAMPLES / "vector_add.lineir"),
            "--input",
            "x_ptr=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]",
            "--input",
            "y_ptr=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]",
            "--input",
            'out_ptr={"shape": [8], "value": 0, "dtype": "f32"}',
        )
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("3.0", out)


class OutputFileTests(unittest.TestCase):
    """--out writes the rendering to disk instead of stdout."""

    def test_trace_out_writes_a_file(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "trace.txt"
            code, out, _ = invoke("trace", str(HEADLINE), "--out", str(target))
            self.assertEqual(code, cli.EXIT_OK)
            self.assertEqual(out, "")
            self.assertIn("opcode", target.read_text(encoding="utf-8"))

    def test_schedule_out_writes_a_file(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "sched.txt"
            code, _, _ = invoke("schedule", str(HEADLINE), "--out", str(target))
            self.assertEqual(code, cli.EXIT_OK)
            self.assertIn("critical path", target.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
