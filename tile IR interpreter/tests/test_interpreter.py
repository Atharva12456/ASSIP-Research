from __future__ import annotations

import unittest

from tile_interp.expr import ExprError
from tile_interp.interpreter import (
    ExecContext,
    Interpreter,
    InterpreterError,
    execute,
    match_blocks,
)
from tile_interp.ir import IRError
from tile_interp.lineir import parse_lineir
from tile_interp.memory import Memory, MemoryError_
from tile_interp.semantics import OPCODES, UnsupportedOpcode
from tile_interp.values import ShapeError, Tile

TRITON_ADD = """# unified-tile-ir line-format v0
# source_lang=triton
# source=examples/triton_add.py
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


def run(text: str, **bindings: object):
    """Parse and execute a snippet in index order."""
    return Interpreter(parse_lineir(text)).run(**bindings)


def one(line: str, **bindings: object):
    """Execute a single op line."""
    return run("0000 | " + line, **bindings)


class ArithmeticOpcodeTests(unittest.TestCase):
    """add sub mul div floordiv mod pow."""

    def test_scalar_arithmetic(self) -> None:
        cases = {
            "add": 9,
            "sub": 5,
            "mul": 14,
            "floordiv": 3,
            "mod": 1,
            "pow": 49,
        }
        for opcode, expected in cases.items():
            with self.subTest(opcode=opcode):
                result = one(f"{opcode} | out=c lhs=a rhs=b", a=7, b=2)
                self.assertEqual(result.env["c"], expected)

    def test_true_division_is_float(self) -> None:
        self.assertEqual(one("div | out=c lhs=a rhs=b", a=7, b=2).env["c"], 3.5)

    def test_tile_arithmetic_broadcasts(self) -> None:
        left = Tile.from_flat([1.0, 2.0, 3.0], (3,), "f32")
        result = one("add | out=c lhs=a rhs=b", a=left, b=10)
        self.assertEqual(result.env["c"].to_nested(), [11.0, 12.0, 13.0])

    def test_inplace_accumulation(self) -> None:
        text = (
            "0000 | assign | out=t value=0\n"
            "0001 | add    | out=t lhs=t rhs=5 inplace=true\n"
            "0002 | add    | out=t lhs=t rhs=5 inplace=true"
        )
        self.assertEqual(run(text).env["t"], 10)

    def test_missing_operand_raises_irerror(self) -> None:
        with self.assertRaises(IRError):
            one("add | out=c lhs=a", a=1)

    def test_division_by_zero_raises_exprerror(self) -> None:
        with self.assertRaises(ExprError):
            one("div | out=c lhs=a rhs=b", a=1, b=0)


class ComparisonAndLogicTests(unittest.TestCase):
    """lt le gt ge eq ne, plus and/or."""

    def test_scalar_comparisons(self) -> None:
        cases = {"lt": False, "le": False, "gt": True, "ge": True, "eq": False, "ne": True}
        for opcode, expected in cases.items():
            with self.subTest(opcode=opcode):
                self.assertIs(one(f"{opcode} | out=c lhs=a rhs=b", a=7, b=2).env["c"], expected)

    def test_tile_comparison_produces_a_mask(self) -> None:
        offsets = Tile.from_flat([0, 1, 2, 3], (4,), "i32")
        mask = one("lt | out=m lhs=o rhs=n", o=offsets, n=2).env["m"]
        self.assertEqual(mask.dtype, "bool")
        self.assertEqual(mask.to_nested(), [True, True, False, False])

    def test_and_over_args(self) -> None:
        self.assertIs(one('and | out=c args="p,q"', p=True, q=False).env["c"], False)
        self.assertIs(one('and | out=c args="p,q"', p=True, q=True).env["c"], True)

    def test_or_over_args(self) -> None:
        self.assertIs(one('or | out=c args="p,q"', p=False, q=False).env["c"], False)
        self.assertIs(one('or | out=c args="p,q"', p=False, q=True).env["c"], True)

    def test_logic_falls_back_to_lhs_rhs(self) -> None:
        self.assertIs(one("and | out=c lhs=p rhs=q", p=True, q=True).env["c"], True)

    def test_logic_on_tiles(self) -> None:
        left = Tile.from_flat([True, True, False], (3,), "bool")
        right = Tile.from_flat([True, False, False], (3,), "bool")
        result = one('and | out=c args="p,q"', p=left, q=right).env["c"]
        self.assertEqual(result.to_nested(), [True, False, False])


class MemoryOpcodeTests(unittest.TestCase):
    """load store arange program_id fill."""

    def test_load_whole_buffer(self) -> None:
        buffer = Tile.from_flat([1.0, 2.0, 3.0], (3,), "f32")
        self.assertEqual(one("load | out=v buf=A", A=buffer).env["v"].to_nested(), [1.0, 2.0, 3.0])

    def test_load_gathers_through_an_index(self) -> None:
        result = one(
            "load | out=v buf=A index=idx",
            A=Tile.from_flat([1.0, 2.0, 3.0, 4.0], (4,), "f32"),
            idx=Tile.from_flat([3, 1], (2,), "i32"),
        )
        self.assertEqual(result.env["v"].shape, (2,))
        self.assertEqual(result.env["v"].to_nested(), [4.0, 2.0])

    def test_masked_load_uses_other(self) -> None:
        result = one(
            "load | out=v buf=A index=idx mask=m other=-1.0",
            A=Tile.from_flat([1.0, 2.0], (2,), "f32"),
            idx=Tile.from_flat([0, 99], (2,), "i32"),
            m=Tile.from_flat([True, False], (2,), "bool"),
        )
        self.assertEqual(result.env["v"].to_nested(), [1.0, -1.0])

    def test_load_through_a_pointer_expression(self) -> None:
        result = one(
            'load | out=v ptr="A + offsets"',
            A=Tile.from_flat([5.0, 6.0, 7.0], (3,), "f32"),
            offsets=Tile.from_flat([2, 0], (2,), "i32"),
        )
        self.assertEqual(result.env["v"].to_nested(), [7.0, 5.0])

    def test_load_with_multi_term_pointer_expression(self) -> None:
        result = one(
            'load | out=v ptr="A + i * n + j"',
            A=Tile.from_flat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), "f32"),
            i=1,
            n=3,
            j=2,
        )
        self.assertEqual(result.env["v"].item(), 6.0)

    def test_out_of_range_index_raises(self) -> None:
        with self.assertRaises(MemoryError_):
            one(
                "load | out=v buf=A index=idx",
                A=Tile.from_flat([1.0], (1,), "f32"),
                idx=Tile.from_flat([7], (1,), "i32"),
            )

    def test_store_whole_buffer(self) -> None:
        result = one(
            "store | buf=A value=v",
            A=Tile.zeros(3, "f32"),
            v=Tile.from_flat([1.0, 2.0, 3.0], (3,), "f32"),
        )
        self.assertEqual(result.memory.buffer("A").tile.to_nested(), [1.0, 2.0, 3.0])

    def test_masked_store_leaves_lanes_untouched(self) -> None:
        result = one(
            "store | buf=A value=v index=idx mask=m",
            A=Tile.zeros(4, "f32"),
            v=Tile.from_flat([9.0, 9.0], (2,), "f32"),
            idx=Tile.from_flat([0, 3], (2,), "i32"),
            m=Tile.from_flat([True, False], (2,), "bool"),
        )
        self.assertEqual(result.memory.buffer("A").tile.to_nested(), [9.0, 0.0, 0.0, 0.0])

    def test_store_to_an_undeclared_buffer_raises(self) -> None:
        with self.assertRaises(MemoryError_):
            one("store | buf=Nowhere value=1")

    def test_load_or_store_without_a_target_raises(self) -> None:
        with self.assertRaises(IRError):
            one("load | out=v")

    def test_arange(self) -> None:
        result = one("arange | out=v start=0 stop=n", n=4)
        self.assertEqual(result.env["v"].to_nested(), [0, 1, 2, 3])
        self.assertEqual(result.env["v"].dtype, "i32")

    def test_program_id(self) -> None:
        program = parse_lineir("0000 | program_id | out=p axis=1")
        result = Interpreter(program, grid=(4, 4), program_ids=(2, 3)).run()
        self.assertEqual(result.env["p"], 3)

    def test_program_id_bad_axis_raises_irerror(self) -> None:
        with self.assertRaises(IRError):
            one("program_id | out=p axis=5")

    def test_fill_with_a_value(self) -> None:
        result = one('fill | out=z args="(4,), 1.5"')
        self.assertEqual(result.env["z"].to_nested(), [1.5, 1.5, 1.5, 1.5])

    def test_fill_defaults_to_zeros(self) -> None:
        result = one('fill | out=z args="(2, 3)"')
        self.assertEqual(result.env["z"].shape, (2, 3))
        self.assertEqual(result.env["z"].dtype, "f32")
        self.assertEqual(result.env["z"].data, [0.0] * 6)

    def test_fill_needs_args(self) -> None:
        with self.assertRaises(IRError):
            one("fill | out=z")


class MathOpcodeTests(unittest.TestCase):
    """dot exp sqrt max min select."""

    def test_dot(self) -> None:
        left = Tile.from_nested([[1.0, 2.0], [3.0, 4.0]])
        right = Tile.from_nested([[5.0, 6.0], [7.0, 8.0]])
        result = one("dot | out=z lhs=a rhs=b", a=left, b=right)
        self.assertEqual(result.env["z"].to_nested(), [[19.0, 22.0], [43.0, 50.0]])

    def test_exp(self) -> None:
        result = one("exp | out=z value=v", v=Tile.from_flat([0.0, 1.0], (2,), "f32"))
        self.assertAlmostEqual(result.env["z"].data[0], 1.0)
        self.assertAlmostEqual(result.env["z"].data[1], 2.718281828459045)

    def test_sqrt(self) -> None:
        result = one("sqrt | out=z value=v", v=Tile.from_flat([4.0, 9.0], (2,), "f32"))
        self.assertEqual(result.env["z"].to_nested(), [2.0, 3.0])

    def test_sqrt_of_a_scalar(self) -> None:
        self.assertEqual(one("sqrt | out=z value=16.0").env["z"], 4.0)

    def test_sqrt_of_a_negative_raises(self) -> None:
        with self.assertRaises(ExprError):
            one("sqrt | out=z value=-1.0")

    def test_max_and_min(self) -> None:
        left = Tile.from_flat([1, 5], (2,), "i32")
        right = Tile.from_flat([4, 2], (2,), "i32")
        self.assertEqual(one("max | out=z lhs=a rhs=b", a=left, b=right).env["z"].to_nested(), [4, 5])
        self.assertEqual(one("min | out=z lhs=a rhs=b", a=left, b=right).env["z"].to_nested(), [1, 2])

    def test_select(self) -> None:
        result = one(
            "select | out=z cond=m true=a false=b",
            m=Tile.from_flat([True, False], (2,), "bool"),
            a=Tile.from_flat([1.0, 2.0], (2,), "f32"),
            b=Tile.from_flat([9.0, 9.0], (2,), "f32"),
        )
        self.assertEqual(result.env["z"].to_nested(), [1.0, 9.0])

    def test_select_on_scalars(self) -> None:
        self.assertEqual(one("select | out=z cond=true true=1 false=2").env["z"], 1)
        self.assertEqual(one("select | out=z cond=false true=1 false=2").env["z"], 2)

    def test_transpose_reverses_the_axes(self) -> None:
        a = Tile.from_nested([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = one("transpose | out=t value=a", a=a)
        self.assertEqual(result.env["t"].shape, (3, 2))
        self.assertEqual(result.env["t"].to_nested(), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

    def test_transpose_below_two_dimensions_is_a_copy(self) -> None:
        result = one("transpose | out=t value=a", a=Tile.from_flat([1.0, 2.0], (2,), "f32"))
        self.assertEqual(result.env["t"].to_nested(), [1.0, 2.0])

    def test_reduce_over_every_element(self) -> None:
        a = Tile.from_nested([[1.0, 2.0], [3.0, 4.0]])
        result = one("reduce | out=s value=a op=sum", a=a)
        self.assertEqual(result.env["s"].shape, ())
        self.assertEqual(result.env["s"].item(), 10.0)

    def test_reduce_along_an_axis_drops_it(self) -> None:
        a = Tile.from_nested([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        rows = one("reduce | out=s value=a op=sum axis=1", a=a).env["s"]
        cols = one("reduce | out=s value=a op=sum axis=0", a=a).env["s"]
        self.assertEqual(rows.to_nested(), [6.0, 15.0])
        self.assertEqual(cols.to_nested(), [5.0, 7.0, 9.0])

    def test_reduce_combiners(self) -> None:
        a = Tile.from_nested([[1.0, 4.0], [3.0, 2.0]])
        self.assertEqual(one("reduce | out=s value=a op=max axis=1", a=a).env["s"].to_nested(), [4.0, 3.0])
        self.assertEqual(one("reduce | out=s value=a op=min axis=1", a=a).env["s"].to_nested(), [1.0, 2.0])
        self.assertEqual(one("reduce | out=s value=a op=prod", a=a).env["s"].item(), 24.0)

    def test_reduce_keepdims_stays_broadcastable(self) -> None:
        """The softmax shape: a keepdims row reduction must broadcast back over its source."""
        a = Tile.from_nested([[1.0, 3.0], [7.0, 5.0]])
        kept = one("reduce | out=s value=a op=max axis=1 keepdims=true", a=a).env["s"]
        self.assertEqual(kept.shape, (2, 1))
        shifted = a.zip_with(kept, lambda x, y: x - y)
        self.assertEqual(shifted.to_nested(), [[-2.0, 0.0], [0.0, -2.0]])

    def test_reduce_keepdims_over_every_axis(self) -> None:
        a = Tile.from_nested([[1.0, 2.0], [3.0, 4.0]])
        kept = one("reduce | out=s value=a op=sum keepdims=true", a=a).env["s"]
        self.assertEqual(kept.shape, (1, 1))
        self.assertEqual(kept.to_nested(), [[10.0]])

    def test_reshape_takes_a_dimension_string(self) -> None:
        a = Tile.from_flat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (6,), "f32")
        result = one("reshape | out=r value=a shape=2x3", a=a)
        self.assertEqual(result.env["r"].shape, (2, 3))
        self.assertEqual(result.env["r"].to_nested(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def test_broadcast_stretches_a_size_one_axis(self) -> None:
        a = Tile.from_flat([1.0, 2.0], (2, 1), "f32")
        result = one("broadcast | out=b value=a shape=2x3", a=a)
        self.assertEqual(result.env["b"].to_nested(), [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

    def test_broadcast_rejects_an_incompatible_shape(self) -> None:
        a = Tile.from_flat([1.0, 2.0, 3.0], (3,), "f32")
        with self.assertRaises(ShapeError):
            one("broadcast | out=b value=a shape=2x2", a=a)

    def test_mma_accumulates_into_its_third_operand(self) -> None:
        left = Tile.from_nested([[1.0, 2.0], [3.0, 4.0]])
        right = Tile.from_nested([[5.0, 6.0], [7.0, 8.0]])
        acc = Tile.from_nested([[1.0, 1.0], [1.0, 1.0]])
        result = one("mma | out=z lhs=a rhs=b acc=c", a=left, b=right, c=acc)
        self.assertEqual(result.env["z"].to_nested(), [[20.0, 23.0], [44.0, 51.0]])

    def test_mma_without_an_accumulator_is_a_plain_product(self) -> None:
        left = Tile.from_nested([[1.0, 2.0], [3.0, 4.0]])
        right = Tile.from_nested([[1.0, 0.0], [0.0, 1.0]])
        result = one("mma | out=z lhs=a rhs=b", a=left, b=right)
        self.assertEqual(result.env["z"].to_nested(), [[1.0, 2.0], [3.0, 4.0]])

    def test_reduce_rejects_an_unknown_combiner(self) -> None:
        a = Tile.from_nested([[1.0, 2.0]])
        with self.assertRaises(UnsupportedOpcode):
            one("reduce | out=s value=a op=median", a=a)

    def test_reduce_rejects_an_out_of_range_axis(self) -> None:
        a = Tile.from_nested([[1.0, 2.0]])
        with self.assertRaises(ShapeError):
            one("reduce | out=s value=a op=sum axis=4", a=a)


class MetaOpcodeTests(unittest.TestCase):
    """kernel return assign call."""

    def test_kernel_binds_nothing(self) -> None:
        result = one('kernel | name=k params="a,b"')
        self.assertEqual(result.env, {})

    def test_assign(self) -> None:
        self.assertEqual(one("assign | out=a value=41").env["a"], 41)

    def test_assign_to_several_targets(self) -> None:
        env = one('assign | out="a,b" value=7').env
        self.assertEqual(env["a"], 7)
        self.assertEqual(env["b"], 7)

    def test_assign_unpacks_a_tuple_target(self) -> None:
        env = one('assign | out="(a, b)" value=pair', pair=[1, 2]).env
        self.assertEqual(env["a"], 1)
        self.assertEqual(env["b"], 2)

    def test_unpack_arity_mismatch_raises(self) -> None:
        with self.assertRaises(IRError):
            one('assign | out="(a, b)" value=pair', pair=[1, 2, 3])

    def test_return_value(self) -> None:
        result = run("0000 | assign | out=a value=5\n0001 | return | value=a")
        self.assertEqual(result.returned, 5)

    def test_return_halts_execution(self) -> None:
        result = run(
            "0000 | assign | out=a value=1\n"
            "0001 | return | value=a\n"
            "0002 | assign | out=b value=2"
        )
        self.assertEqual(result.order, [0, 1])
        self.assertNotIn("b", result.env)

    def test_bare_return(self) -> None:
        self.assertIsNone(run("0000 | return |").returned)

    def test_call_builtin(self) -> None:
        total = one("call | out=z callee=sum args=v", v=Tile.from_flat([1.0, 2.0, 3.0], (3,), "f32"))
        self.assertEqual(total.env["z"].item(), 6.0)

    def test_call_with_a_dotted_prefix(self) -> None:
        result = one("call | out=z callee=tl.trans args=v", v=Tile.from_nested([[1, 2], [3, 4]]))
        self.assertEqual(result.env["z"].to_nested(), [[1, 3], [2, 4]])

    def test_call_unknown_callee_raises(self) -> None:
        with self.assertRaises(UnsupportedOpcode):
            one("call | out=z callee=frobnicate args=v", v=1)

    def test_call_without_a_callee_raises(self) -> None:
        with self.assertRaises(UnsupportedOpcode):
            one('call | expr="tl.debug_barrier()"')


class UnknownOpcodeTests(unittest.TestCase):
    """An unknown opcode must fail loudly, never silently pass."""

    def test_run_raises(self) -> None:
        with self.assertRaises(UnsupportedOpcode):
            run("0000 | frobnicate | out=q")

    def test_run_in_order_raises(self) -> None:
        program = parse_lineir("0000 | frobnicate | out=q")
        with self.assertRaises(UnsupportedOpcode):
            Interpreter(program).run_in_order([0])

    def test_nothing_is_bound(self) -> None:
        program = parse_lineir("0000 | assign | out=a value=1\n0001 | frobnicate | out=q")
        with self.assertRaises(UnsupportedOpcode):
            Interpreter(program).run()


class ControlFlowTests(unittest.TestCase):
    """if/else/endif, for/endfor and while/endwhile, arbitrarily nested."""

    def test_if_taken(self) -> None:
        text = (
            "0000 | assign | out=t value=0\n"
            "0001 | if     | cond=c\n"
            "0002 | assign | out=t value=1\n"
            "0003 | endif  |"
        )
        self.assertEqual(run(text, c=True).env["t"], 1)
        self.assertEqual(run(text, c=False).env["t"], 0)

    def test_if_else(self) -> None:
        text = (
            "0000 | if     | cond=c\n"
            "0001 | assign | out=t value=1\n"
            "0002 | else   |\n"
            "0003 | assign | out=t value=2\n"
            "0004 | endif  |"
        )
        self.assertEqual(run(text, c=True).env["t"], 1)
        self.assertEqual(run(text, c=False).env["t"], 2)

    def test_then_branch_order_includes_the_markers(self) -> None:
        text = (
            "0000 | if     | cond=c\n"
            "0001 | assign | out=t value=1\n"
            "0002 | else   |\n"
            "0003 | assign | out=t value=2\n"
            "0004 | endif  |"
        )
        self.assertEqual(run(text, c=True).order, [0, 1, 2, 4])
        self.assertEqual(run(text, c=False).order, [0, 3, 4])

    def test_for_over_range(self) -> None:
        text = (
            "0000 | assign | out=t value=0\n"
            "0001 | for    | target=i iter=range(4)\n"
            "0002 | add    | out=t lhs=t rhs=i\n"
            "0003 | endfor |"
        )
        self.assertEqual(run(text).env["t"], 6)

    def test_for_over_a_tile(self) -> None:
        text = (
            "0000 | assign | out=t value=0\n"
            "0001 | for    | target=v iter=data\n"
            "0002 | add    | out=t lhs=t rhs=v\n"
            "0003 | endfor |"
        )
        result = run(text, data=Tile.from_flat([2, 3, 5], (3,), "i32"))
        self.assertEqual(result.env["t"], 10)

    def test_for_over_a_c_style_clause(self) -> None:
        text = (
            "0000 | assign | out=t value=0\n"
            '0001 | for    | iter="int j = 0; j < 4; j += 1"\n'
            "0002 | add    | out=t lhs=t rhs=j\n"
            "0003 | endfor |"
        )
        self.assertEqual(run(text).env["t"], 6)

    def test_c_style_inclusive_bound(self) -> None:
        text = (
            "0000 | assign | out=t value=0\n"
            '0001 | for    | iter="int j = 1; j <= 3; ++j"\n'
            "0002 | add    | out=t lhs=t rhs=j\n"
            "0003 | endfor |"
        )
        self.assertEqual(run(text).env["t"], 6)

    def test_while_loop(self) -> None:
        text = (
            "0000 | assign | out=k value=0\n"
            "0001 | while  | cond=\"k < 5\"\n"
            "0002 | add    | out=k lhs=k rhs=1\n"
            "0003 | endwhile |"
        )
        self.assertEqual(run(text).env["k"], 5)

    def test_while_guard_fires(self) -> None:
        text = (
            "0000 | assign | out=k value=0\n"
            "0001 | while  | cond=true\n"
            "0002 | add    | out=k lhs=k rhs=1\n"
            "0003 | endwhile |"
        )
        program = parse_lineir(text)
        with self.assertRaises(InterpreterError) as caught:
            Interpreter(program, max_iterations=25).run()
        self.assertIn("25", str(caught.exception))

    def test_the_shipped_default_guard_also_fires(self) -> None:
        text = (
            "0000 | assign | out=k value=0\n"
            "0001 | while  | cond=true\n"
            "0002 | add    | out=k lhs=k rhs=1\n"
            "0003 | endwhile |"
        )
        with self.assertRaises(InterpreterError) as caught:
            Interpreter(parse_lineir(text)).run()
        self.assertIn("10000", str(caught.exception))

    def test_for_guard_fires(self) -> None:
        text = (
            "0000 | for    | target=i iter=range(500)\n"
            "0001 | assign | out=t value=i\n"
            "0002 | endfor |"
        )
        with self.assertRaises(InterpreterError):
            Interpreter(parse_lineir(text), max_iterations=10).run()

    def test_if_inside_for_inside_for(self) -> None:
        text = (
            "0000 | kernel | name=nested params=\n"
            "0001 | assign | out=total value=0\n"
            "0002 | for    | target=i iter=range(3)\n"
            "0003 | for    | target=j iter=range(3)\n"
            '0004 | if     | cond="i < j"\n'
            "0005 | add    | out=total lhs=total rhs=1\n"
            "0006 | else   |\n"
            "0007 | sub    | out=total lhs=total rhs=1\n"
            "0008 | endif  |\n"
            "0009 | endfor |\n"
            "0010 | endfor |\n"
            "0011 | return | value=total"
        )
        result = run(text)
        self.assertEqual(result.returned, -3)
        self.assertEqual(result.env["total"], -3)

    def test_while_inside_for(self) -> None:
        text = (
            "0000 | assign | out=t value=0\n"
            "0001 | for    | target=i iter=range(3)\n"
            "0002 | assign | out=k value=0\n"
            '0003 | while  | cond="k < 2"\n'
            "0004 | add    | out=k lhs=k rhs=1\n"
            "0005 | add    | out=t lhs=t rhs=1\n"
            "0006 | endwhile |\n"
            "0007 | endfor |"
        )
        self.assertEqual(run(text).env["t"], 6)

    def test_deeply_nested_loops_do_not_recurse(self) -> None:
        lines = ["0000 | assign | out=t value=0"]
        depth = 40
        index = 1
        for _ in range(depth):
            lines.append(f"{index:04d} | for | target=i{index} iter=range(1)")
            index += 1
        lines.append(f"{index:04d} | add | out=t lhs=t rhs=1")
        index += 1
        for _ in range(depth):
            lines.append(f"{index:04d} | endfor |")
            index += 1
        self.assertEqual(run("\n".join(lines)).env["t"], 1)

    def test_empty_loop_body(self) -> None:
        text = "0000 | for | target=i iter=range(0)\n0001 | assign | out=t value=1\n0002 | endfor |"
        self.assertNotIn("t", run(text).env)

    def test_unbalanced_control_flow_raises(self) -> None:
        with self.assertRaises(InterpreterError):
            run("0000 | if | cond=true")
        with self.assertRaises(InterpreterError):
            run("0000 | endif |")
        with self.assertRaises(InterpreterError):
            run("0000 | for | target=i iter=range(1)\n0001 | endif |")
        with self.assertRaises(InterpreterError):
            run("0000 | else |")

    def test_double_else_raises(self) -> None:
        text = "0000 | if | cond=true\n0001 | else |\n0002 | else |\n0003 | endif |"
        with self.assertRaises(InterpreterError):
            run(text)

    def test_match_blocks_pairs_markers(self) -> None:
        program = parse_lineir(
            "0000 | if    | cond=c\n0001 | else  |\n0002 | endif |\n"
            "0003 | for   | target=i iter=range(1)\n0004 | endfor |"
        )
        blocks = match_blocks(program)
        self.assertEqual(blocks.close[0], 2)
        self.assertEqual(blocks.open[2], 0)
        self.assertEqual(blocks.alt[0], 1)
        self.assertEqual(blocks.alt_end[1], 2)
        self.assertEqual(blocks.close[3], 4)


class RunInOrderTests(unittest.TestCase):
    """run_in_order proves a reordering is sound, and refuses control flow."""

    def setUp(self) -> None:
        self.program = parse_lineir(TRITON_ADD)

    def inputs(self) -> dict[str, object]:
        return {
            "x_ptr": Tile.from_flat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (6,), "f32"),
            "y_ptr": Tile.from_flat([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], (6,), "f32"),
            "out_ptr": Tile.zeros(6, "f32"),
            "n_elements": 6,
            "BLOCK": 8,
        }

    def test_index_order_matches_run(self) -> None:
        sequential = Interpreter(self.program).run(**self.inputs())
        reordered = Interpreter(self.program).run_in_order(range(11), **self.inputs())
        self.assertEqual(reordered.memory.snapshot(), sequential.memory.snapshot())

    def test_swapping_the_two_loads_is_sound(self) -> None:
        sequential = Interpreter(self.program).run(**self.inputs())
        order = [0, 1, 2, 3, 4, 5, 7, 6, 8, 9, 10]
        reordered = Interpreter(self.program).run_in_order(order, **self.inputs())
        self.assertEqual(reordered.memory.snapshot(), sequential.memory.snapshot())
        self.assertEqual(reordered.order, order)

    def test_control_flow_is_rejected(self) -> None:
        text = (
            "0000 | assign | out=t value=0\n"
            "0001 | if     | cond=true\n"
            "0002 | assign | out=t value=1\n"
            "0003 | endif  |"
        )
        program = parse_lineir(text)
        with self.assertRaises(InterpreterError) as caught:
            Interpreter(program).run_in_order([0, 1, 2, 3])
        self.assertIn("control flow", str(caught.exception))

    def test_every_control_opcode_is_rejected(self) -> None:
        for opcode, attrs in (
            ("if", "cond=true"),
            ("else", ""),
            ("endif", ""),
            ("for", "target=i iter=range(1)"),
            ("endfor", ""),
            ("while", "cond=false"),
            ("endwhile", ""),
        ):
            with self.subTest(opcode=opcode):
                program = parse_lineir(f"0000 | {opcode} | {attrs}")
                with self.assertRaises(InterpreterError):
                    Interpreter(program).run_in_order([0])

    def test_short_order_is_rejected(self) -> None:
        program = parse_lineir("0000 | assign | out=a value=1\n0001 | assign | out=b value=2")
        with self.assertRaises(InterpreterError):
            Interpreter(program).run_in_order([0])

    def test_duplicate_order_is_rejected(self) -> None:
        program = parse_lineir("0000 | assign | out=a value=1\n0001 | assign | out=b value=2")
        with self.assertRaises(InterpreterError):
            Interpreter(program).run_in_order([0, 0])


class TritonKernelTests(unittest.TestCase):
    """A full emitter-shaped kernel executes end to end."""

    def test_masked_vector_add(self) -> None:
        program = parse_lineir(TRITON_ADD)
        result = Interpreter(program).run(
            x_ptr=Tile.from_flat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (6,), "f32"),
            y_ptr=Tile.from_flat([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], (6,), "f32"),
            out_ptr=Tile.zeros(6, "f32"),
            n_elements=6,
            BLOCK=8,
        )
        self.assertEqual(
            result.memory.buffer("out_ptr").tile.to_nested(),
            [11.0, 22.0, 33.0, 44.0, 55.0, 66.0],
        )
        self.assertEqual(result.order, list(range(11)))

    def test_second_program_id_writes_the_second_block(self) -> None:
        program = parse_lineir(TRITON_ADD)
        result = Interpreter(program, grid=(2,), program_ids=(1,)).run(
            x_ptr=Tile.from_flat([1.0, 2.0, 3.0, 4.0], (4,), "f32"),
            y_ptr=Tile.from_flat([10.0, 20.0, 30.0, 40.0], (4,), "f32"),
            out_ptr=Tile.zeros(4, "f32"),
            n_elements=4,
            BLOCK=2,
        )
        self.assertEqual(result.memory.buffer("out_ptr").tile.to_nested(), [0.0, 0.0, 33.0, 44.0])


class InterpreterStateTests(unittest.TestCase):
    """Seeded memory, reusability, and the ExecContext surface."""

    def test_tile_bindings_declare_buffers(self) -> None:
        result = one("assign | out=q value=1", A=Tile.zeros(2, "f32"))
        self.assertTrue(result.memory.has("A"))

    def test_scalar_bindings_do_not_declare_buffers(self) -> None:
        result = one("assign | out=q value=1", n=5)
        self.assertFalse(result.memory.has("n"))

    def test_seeded_memory_is_not_mutated(self) -> None:
        memory = Memory()
        memory.declare("A", Tile.zeros(2, "f32"))
        program = parse_lineir("0000 | store | buf=A value=v")
        interpreter = Interpreter(program, memory=memory)
        interpreter.run(v=Tile.from_flat([1.0, 2.0], (2,), "f32"))
        self.assertEqual(memory.buffer("A").tile.to_nested(), [0.0, 0.0])

    def test_interpreter_is_reusable(self) -> None:
        program = parse_lineir("0000 | add | out=c lhs=a rhs=b")
        interpreter = Interpreter(program)
        self.assertEqual(interpreter.run(a=1, b=2).env["c"], 3)
        self.assertEqual(interpreter.run(a=10, b=20).env["c"], 30)

    def test_exec_context_value_and_bind(self) -> None:
        program = parse_lineir("0000 | assign | out=a value=1")
        ctx = ExecContext(program, Memory())
        ctx.bind("n", 4)
        self.assertEqual(ctx.value("n + 1"), 5)
        self.assertIsNone(ctx.value(None))
        self.assertEqual(ctx.value("", 7), 7)

    def test_exec_result_output_helper(self) -> None:
        result = one("store | buf=A value=v", A=Tile.zeros(2, "f32"), v=1.0)
        self.assertEqual(result.output("A").to_nested(), [1.0, 1.0])

    def test_execute_helper(self) -> None:
        program = parse_lineir("0000 | add | out=c lhs=a rhs=b")
        self.assertEqual(execute(program, a=2, b=3).env["c"], 5)

    def test_max_steps_guard(self) -> None:
        text = (
            "0000 | assign | out=k value=0\n"
            "0001 | while  | cond=true\n"
            "0002 | add    | out=k lhs=k rhs=1\n"
            "0003 | endwhile |"
        )
        program = parse_lineir(text)
        with self.assertRaises(InterpreterError):
            Interpreter(program, max_iterations=10**9, max_steps=50).run()

    def test_shape_mismatch_propagates(self) -> None:
        with self.assertRaises((ShapeError, ExprError)):
            one(
                "add | out=c lhs=a rhs=b",
                a=Tile.from_flat([1.0, 2.0], (2,), "f32"),
                b=Tile.from_flat([1.0, 2.0, 3.0], (3,), "f32"),
            )


ARITHMETIC_BATTERY = """0000 | kernel   | name=k params="A,C"
0001 | arange   | out=i start=0 stop=4
0002 | load     | out=a buf=A
0003 | add      | out=s lhs=a rhs=a
0004 | sub      | out=t lhs=s rhs=a
0005 | mul      | out=u lhs=t rhs=2
0006 | div      | out=v lhs=u rhs=2
0007 | floordiv | out=w lhs=i rhs=2
0008 | mod      | out=y lhs=i rhs=2
0009 | pow      | out=z lhs=i rhs=2
0010 | store    | buf=C value=v
0011 | return   | value=z"""

PREDICATE_BATTERY = """0000 | arange | out=i start=0 stop=4
0001 | lt     | out=m1 lhs=i rhs=2
0002 | le     | out=m2 lhs=i rhs=2
0003 | gt     | out=m3 lhs=i rhs=2
0004 | ge     | out=m4 lhs=i rhs=2
0005 | eq     | out=m5 lhs=i rhs=2
0006 | ne     | out=m6 lhs=i rhs=2
0007 | and    | out=c1 args="m1,m2"
0008 | or     | out=c2 args="m3,m4"
0009 | select | out=p cond=m1 true=i false=m5"""

MATH_BATTERY = """0000 | program_id | out=pid axis=0
0001 | fill       | out=f args="4,1.0"
0002 | exp        | out=e value=f
0003 | sqrt       | out=q value=f
0004 | max        | out=mx lhs=f rhs=e
0005 | min        | out=mn lhs=f rhs=e
0006 | assign     | out=g value=f
0007 | call       | out=h callee=sum args=f"""

DOT_BATTERY = """0000 | assign | out=L value="[[1.0,2.0],[3.0,4.0]]"
0001 | assign | out=R value="[[1.0,0.0],[0.0,1.0]]"
0002 | dot    | out=D lhs=L rhs=R"""

REDUCTION_BATTERY = """0000 | assign    | out=M value="[[1.0,2.0],[3.0,4.0]]"
0001 | transpose | out=Mt value=M
0002 | reduce    | out=Rows value=M op=sum axis=1
0003 | reduce    | out=Top value=M op=max axis=1 keepdims=true
0004 | reduce    | out=All value=M op=sum
0005 | reshape   | out=Flat value=M shape=4
0006 | broadcast | out=Wide value=Top shape=2x2
0007 | mma       | out=Prod lhs=M rhs=M acc=M"""

CONTROL_BATTERY = """0000 | assign   | out=t value=0
0001 | for      | target=i iter=range(3)
0002 | if       | cond="i > 0"
0003 | add      | out=t lhs=t rhs=i
0004 | else     |
0005 | assign   | out=t value=0
0006 | endif    |
0007 | endfor   |
0008 | assign   | out=n value=0
0009 | while    | cond="n < 2"
0010 | add      | out=n lhs=n rhs=1
0011 | endwhile |"""


class OpcodeCoverageTests(unittest.TestCase):
    """Every opcode in the catalog must actually execute somewhere in this suite."""

    def batteries(self) -> list[tuple[str, dict[str, object]]]:
        return [
            (
                ARITHMETIC_BATTERY,
                {
                    "A": Tile.from_flat([1.0, 2.0, 3.0, 4.0], (4,), "f32"),
                    "C": Tile.zeros(4, "f32"),
                },
            ),
            (PREDICATE_BATTERY, {}),
            (MATH_BATTERY, {}),
            (DOT_BATTERY, {}),
            (REDUCTION_BATTERY, {}),
            (CONTROL_BATTERY, {}),
        ]

    def executed_opcodes(self) -> set[str]:
        seen: set[str] = set()
        for text, bindings in self.batteries():
            seen |= {event.opcode for event in run(text, **bindings).trace}
        return seen

    def test_every_catalog_opcode_executes(self) -> None:
        missing = sorted(set(OPCODES) - self.executed_opcodes())
        self.assertEqual(missing, [], f"opcodes never executed: {missing}")

    def test_the_batteries_run_nothing_outside_the_catalog(self) -> None:
        self.assertEqual(self.executed_opcodes() - set(OPCODES), set())

    def test_arithmetic_battery_results(self) -> None:
        result = run(
            ARITHMETIC_BATTERY,
            A=Tile.from_flat([1.0, 2.0, 3.0, 4.0], (4,), "f32"),
            C=Tile.zeros(4, "f32"),
        )
        self.assertEqual(result.memory.buffer("C").tile.to_nested(), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result.returned.to_nested(), [0, 1, 4, 9])

    def test_control_battery_results(self) -> None:
        result = run(CONTROL_BATTERY)
        self.assertEqual(result.env["t"], 3)
        self.assertEqual(result.env["n"], 2)

    def test_dot_battery_is_an_identity_product(self) -> None:
        self.assertEqual(
            run(DOT_BATTERY).env["D"].to_nested(), [[1.0, 2.0], [3.0, 4.0]]
        )


class RecorderReuseTests(unittest.TestCase):
    """A supplied recorder is per-run state; two runs must not share one trace."""

    PROGRAM = "0000 | assign | out=a value=1\n0001 | add | out=b lhs=a rhs=2"

    def test_a_second_run_does_not_append_to_the_first_trace(self) -> None:
        from tile_interp.trace import TraceRecorder

        recorder = TraceRecorder()
        interpreter = Interpreter(parse_lineir(self.PROGRAM), recorder=recorder)
        first = interpreter.run()
        second = interpreter.run()
        self.assertEqual(len(first.trace), 2)
        self.assertEqual(len(second.trace), 2)
        self.assertEqual(second.trace.op_indices(), second.order)

    def test_captured_values_do_not_leak_between_runs(self) -> None:
        from tile_interp.trace import TraceRecorder

        recorder = TraceRecorder(capture_values=True)
        interpreter = Interpreter(parse_lineir(self.PROGRAM), recorder=recorder)
        interpreter.run()
        interpreter.run()
        self.assertEqual(sorted(recorder.values), [0, 1])


if __name__ == "__main__":
    unittest.main()
