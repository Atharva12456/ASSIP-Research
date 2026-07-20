from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tile_interp.ir import IRError, Op, Program
from tile_interp.lineir import HEADER, parse_lineir, parse_lineir_file, to_lineir

TRITON_ADD = """# unified-tile-ir line-format v0
# source_lang=triton
# source=examples/triton_add.py
# max_ops=220
0000 | kernel       | name=add_kernel params="x_ptr,y_ptr,out_ptr,n_elements,BLOCK"
0001 | program_id   | out=pid axis=0
0002 | mul          | out=block_start lhs=pid rhs=BLOCK
0003 | arange       | out=offs start=0 stop=BLOCK
0004 | add          | out=offsets lhs=block_start rhs=offs
0005 | lt           | out=mask lhs=offsets rhs=n_elements
0006 | load         | out=x ptr="x_ptr + offsets" mask=mask other=0.0
0007 | load         | out=y ptr="y_ptr + offsets" mask=mask other=0.0
0008 | add          | out=z lhs=x rhs=y
0009 | store        | ptr="out_ptr + offsets" value=z mask=mask
0010 | return       | value=z"""


class HeaderTests(unittest.TestCase):
    """The '#' headers carry provenance and participate in equality."""

    def test_defaults_when_headers_are_absent(self) -> None:
        program = parse_lineir("0000 | kernel | name=k")
        self.assertEqual(program.source_lang, "unknown")
        self.assertEqual(program.source_name, "<memory>")
        self.assertEqual(program.max_ops, 220)

    def test_source_name_argument_is_used_when_no_header(self) -> None:
        program = parse_lineir("0000 | kernel | name=k", "given.py")
        self.assertEqual(program.source_name, "given.py")

    def test_source_header_beats_the_argument(self) -> None:
        text = f"{HEADER}\n# source=real.py\n0000 | kernel | name=k"
        self.assertEqual(parse_lineir(text, "given.py").source_name, "real.py")

    def test_all_headers_are_read(self) -> None:
        program = parse_lineir(TRITON_ADD)
        self.assertEqual(program.source_lang, "triton")
        self.assertEqual(program.source_name, "examples/triton_add.py")
        self.assertEqual(program.max_ops, 220)

    def test_quoted_source_header(self) -> None:
        text = f'{HEADER}\n# source="examples\\\\triton_add.py"\n0000 | kernel | name=k'
        self.assertEqual(parse_lineir(text).source_name, "examples\\triton_add.py")

    def test_non_integer_max_ops_raises(self) -> None:
        with self.assertRaises(IRError):
            parse_lineir("# max_ops=lots\n0000 | kernel | name=k")

    def test_comment_without_equals_is_ignored(self) -> None:
        program = parse_lineir("# just a note\n0000 | kernel | name=k")
        self.assertEqual(len(program), 1)

    def test_emitted_headers(self) -> None:
        lines = to_lineir(parse_lineir(TRITON_ADD)).splitlines()
        self.assertEqual(lines[0], HEADER)
        self.assertEqual(lines[1], "# source_lang=triton")
        self.assertEqual(lines[2], "# source=examples/triton_add.py")
        self.assertEqual(lines[3], "# max_ops=220")


class ParseTests(unittest.TestCase):
    """Ops, opcodes, and attribute parsing."""

    def test_op_count_and_indices(self) -> None:
        program = parse_lineir(TRITON_ADD)
        self.assertEqual(len(program), 11)
        self.assertEqual([op.index for op in program], list(range(11)))

    def test_opcodes(self) -> None:
        program = parse_lineir(TRITON_ADD)
        self.assertEqual(program.ops[0].opcode, "kernel")
        self.assertEqual(program.ops[6].opcode, "load")
        self.assertEqual(program.ops[10].opcode, "return")

    def test_kernel_name_and_params(self) -> None:
        program = parse_lineir(TRITON_ADD)
        self.assertEqual(program.kernel_name(), "add_kernel")
        self.assertEqual(
            program.params(), ["x_ptr", "y_ptr", "out_ptr", "n_elements", "BLOCK"]
        )

    def test_anonymous_without_a_kernel_op(self) -> None:
        program = parse_lineir("0000 | add | out=c lhs=a rhs=b")
        self.assertEqual(program.kernel_name(), "anonymous")
        self.assertEqual(program.params(), [])

    def test_quoted_value_with_spaces(self) -> None:
        program = parse_lineir('0000 | load | out=x ptr="x_ptr + offsets"')
        self.assertEqual(program.ops[0].get("ptr"), "x_ptr + offsets")

    def test_quoted_value_containing_a_pipe(self) -> None:
        program = parse_lineir('0000 | assign | out=q value="a | b"')
        self.assertEqual(program.ops[0].get("value"), "a | b")

    def test_empty_quoted_value(self) -> None:
        program = parse_lineir('0000 | kernel | name=k params=""')
        self.assertEqual(program.ops[0].get("params"), "")

    def test_empty_attribute_section(self) -> None:
        program = parse_lineir("0000 | endif |")
        self.assertEqual(program.ops[0].attrs, {})

    def test_missing_attribute_section(self) -> None:
        program = parse_lineir("0000 | endif")
        self.assertEqual(program.ops[0].opcode, "endif")
        self.assertEqual(program.ops[0].attrs, {})

    def test_whitespace_only_attribute_section(self) -> None:
        self.assertEqual(parse_lineir("0000 | endfor |    ").ops[0].attrs, {})

    def test_blank_lines_are_skipped(self) -> None:
        program = parse_lineir("0000 | kernel | name=k\n\n\n0001 | endif |\n")
        self.assertEqual(len(program), 2)

    def test_attribute_order_is_preserved(self) -> None:
        program = parse_lineir("0000 | load | zeta=1 alpha=2 mid=3")
        self.assertEqual(list(program.ops[0].attrs), ["zeta", "alpha", "mid"])

    def test_op_lookup(self) -> None:
        program = parse_lineir(TRITON_ADD)
        self.assertEqual(program.op(5).opcode, "lt")
        with self.assertRaises(IRError):
            program.op(99)

    def test_find_by_opcode(self) -> None:
        self.assertEqual(len(parse_lineir(TRITON_ADD).find("load")), 2)


class ParseErrorTests(unittest.TestCase):
    """Malformed lines raise IRError, never a bare ValueError."""

    def test_missing_pipe(self) -> None:
        with self.assertRaises(IRError):
            parse_lineir("0000")

    def test_non_numeric_index(self) -> None:
        with self.assertRaises(IRError):
            parse_lineir("abc | kernel |")

    def test_missing_opcode(self) -> None:
        with self.assertRaises(IRError):
            parse_lineir("0000 |  | name=k")

    def test_attribute_without_equals(self) -> None:
        with self.assertRaises(IRError):
            parse_lineir("0000 | kernel | dangling")

    def test_unterminated_quote(self) -> None:
        with self.assertRaises(IRError):
            parse_lineir('0000 | kernel | name="open')


class RoundTripTests(unittest.TestCase):
    """parse_lineir(to_lineir(p)) == p, op-for-op and attr-for-attr."""

    def assert_round_trips(self, text: str) -> Program:
        program = parse_lineir(text)
        again = parse_lineir(to_lineir(program))
        self.assertEqual(again, program)
        self.assertEqual(len(again), len(program))
        for left, right in zip(program.ops, again.ops):
            self.assertEqual(left.index, right.index)
            self.assertEqual(left.opcode, right.opcode)
            self.assertEqual(list(left.attrs.items()), list(right.attrs.items()))
        return program

    def test_triton_kernel(self) -> None:
        self.assert_round_trips(TRITON_ADD)

    def test_round_trip_is_idempotent(self) -> None:
        program = parse_lineir(TRITON_ADD)
        once = to_lineir(program)
        twice = to_lineir(parse_lineir(once))
        self.assertEqual(once, twice)

    def test_values_needing_quotes(self) -> None:
        self.assert_round_trips(
            '0000 | kernel | name=k params="a,b,c"\n'
            '0001 | load   | out=v ptr="base + i * stride + j"\n'
            '0002 | assign | out=s value="hello world"\n'
            '0003 | assign | out=e value=""\n'
            '0004 | assign | out=p value="a\\\\b"'
        )

    def test_empty_attr_sections(self) -> None:
        self.assert_round_trips("0000 | if | cond=c\n0001 | else |\n0002 | endif |")

    def test_control_flow_program(self) -> None:
        self.assert_round_trips(
            "0000 | kernel | name=loops params=n\n"
            "0001 | for    | target=i iter=range(n)\n"
            "0002 | while  | cond=go\n"
            "0003 | endwhile |\n"
            "0004 | endfor |\n"
            "0005 | return |"
        )

    def test_attr_order_survives_the_round_trip(self) -> None:
        program = self.assert_round_trips("0000 | load | zeta=1 alpha=2 mid=3")
        self.assertEqual(list(program.ops[0].attrs), ["zeta", "alpha", "mid"])

    def test_empty_program(self) -> None:
        self.assert_round_trips(f"{HEADER}\n# source_lang=triton\n# source=empty.py")

    def test_max_ops_survives(self) -> None:
        program = self.assert_round_trips("# max_ops=42\n0000 | kernel | name=k")
        self.assertEqual(program.max_ops, 42)

    def test_hand_built_program(self) -> None:
        program = Program(
            "cutile",
            "examples/cutile_scale.cu",
            [
                Op(0, "kernel", {"name": "scale", "params": "A,B,alpha"}),
                Op(1, "load", {"out": "a", "buf": "A"}),
                Op(2, "mul", {"out": "b", "lhs": "a", "rhs": "alpha"}),
                Op(3, "store", {"buf": "B", "value": "b"}),
            ],
            64,
        )
        self.assertEqual(parse_lineir(to_lineir(program)), program)

    def test_no_trailing_newline(self) -> None:
        self.assertFalse(to_lineir(parse_lineir(TRITON_ADD)).endswith("\n"))


class FileTests(unittest.TestCase):
    """parse_lineir_file uses pathlib and records the resolved path."""

    def test_reads_a_file(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.lineir"
            path.write_text(TRITON_ADD, encoding="utf-8")
            program = parse_lineir_file(path)
            self.assertEqual(len(program), 11)
            self.assertEqual(program.source_name, "examples/triton_add.py")

    def test_source_name_falls_back_to_the_path(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bare.lineir"
            path.write_text("0000 | kernel | name=k", encoding="utf-8")
            self.assertEqual(parse_lineir_file(path).source_name, str(path))

    def test_accepts_a_string_path(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bare.lineir"
            path.write_text("0000 | kernel | name=k", encoding="utf-8")
            self.assertEqual(len(parse_lineir_file(str(path))), 1)


class OpAccessorTests(unittest.TestCase):
    """Op.get / require / flag / list_attr."""

    def test_get_with_default(self) -> None:
        op = Op(0, "add", {"out": "c"})
        self.assertEqual(op.get("out"), "c")
        self.assertIsNone(op.get("lhs"))
        self.assertEqual(op.get("lhs", "zero"), "zero")

    def test_require_raises_for_missing(self) -> None:
        op = Op(3, "add", {"out": "c"})
        self.assertEqual(op.require("out"), "c")
        with self.assertRaises(IRError):
            op.require("lhs")

    def test_flag(self) -> None:
        self.assertTrue(Op(0, "add", {"inplace": "true"}).flag("inplace"))
        self.assertFalse(Op(0, "add", {"inplace": "false"}).flag("inplace"))
        self.assertFalse(Op(0, "add", {}).flag("inplace"))
        self.assertTrue(Op(0, "add", {}).flag("inplace", True))

    def test_list_attr(self) -> None:
        op = Op(0, "kernel", {"params": "a, b ,c"})
        self.assertEqual(op.list_attr("params"), ["a", "b", "c"])
        self.assertEqual(op.list_attr("missing"), [])


class IndexOrderTests(unittest.TestCase):
    """The op index is an identity: it must be unique and it must increase."""

    def test_duplicate_index_is_rejected(self) -> None:
        with self.assertRaises(IRError):
            parse_lineir("0000 | assign | out=a value=1\n0000 | assign | out=b value=2")

    def test_descending_index_is_rejected(self) -> None:
        with self.assertRaises(IRError):
            parse_lineir("0005 | assign | out=a value=1\n0002 | assign | out=b value=2")

    def test_gaps_are_allowed_while_increasing(self) -> None:
        program = parse_lineir("0000 | assign | out=a value=1\n0007 | assign | out=b value=2")
        self.assertEqual([op.index for op in program.ops], [0, 7])

    def test_the_rejection_keeps_file_order_equal_to_index_order(self) -> None:
        program = parse_lineir(TRITON_ADD)
        self.assertEqual([op.index for op in program.ops], list(range(len(program))))


if __name__ == "__main__":
    unittest.main()
