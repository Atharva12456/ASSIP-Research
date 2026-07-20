from __future__ import annotations

import unittest

from tile_interp.expr import (
    ExprError,
    apply_binary,
    apply_boolean,
    apply_compare,
    apply_unary,
    eval_attr,
    parse_names,
    subscript,
    truthy,
)
from tile_interp.values import ShapeError, Tile


def env() -> dict[str, object]:
    """A representative environment: scalars, a vector, and a matrix."""
    return {
        "n": 10,
        "half": 0.5,
        "flag": True,
        "x": Tile.from_flat([1, 2, 3, 4], (4,), "i32"),
        "y": Tile.from_flat([10.0, 20.0, 30.0, 40.0], (4,), "f32"),
        "m": Tile.from_nested([[1, 2, 3], [4, 5, 6]]),
        "items": [7, 8, 9],
    }


class WhitelistedNodeTests(unittest.TestCase):
    """Every node type the contract whitelists must evaluate."""

    def setUp(self) -> None:
        self.env = env()

    def test_name(self) -> None:
        self.assertEqual(eval_attr("n", self.env), 10)

    def test_constant_int(self) -> None:
        self.assertEqual(eval_attr("42", self.env), 42)

    def test_constant_float(self) -> None:
        self.assertEqual(eval_attr("2.5", self.env), 2.5)

    def test_constant_string(self) -> None:
        self.assertEqual(eval_attr("'abc'", self.env), "abc")

    def test_sibling_bool_spelling(self) -> None:
        self.assertIs(eval_attr("true", self.env), True)
        self.assertIs(eval_attr("false", self.env), False)
        self.assertIs(eval_attr("True", self.env), True)
        self.assertIs(eval_attr("False", self.env), False)

    def test_unary_usub(self) -> None:
        self.assertEqual(eval_attr("-n", self.env), -10)

    def test_unary_uadd(self) -> None:
        self.assertEqual(eval_attr("+n", self.env), 10)

    def test_unary_not(self) -> None:
        self.assertIs(eval_attr("not flag", self.env), False)

    def test_binop_add(self) -> None:
        self.assertEqual(eval_attr("n + 1", self.env), 11)

    def test_binop_sub(self) -> None:
        self.assertEqual(eval_attr("n - 1", self.env), 9)

    def test_binop_mult(self) -> None:
        self.assertEqual(eval_attr("n * 3", self.env), 30)

    def test_binop_div(self) -> None:
        self.assertEqual(eval_attr("n / 4", self.env), 2.5)

    def test_binop_floordiv(self) -> None:
        self.assertEqual(eval_attr("n // 4", self.env), 2)

    def test_binop_mod(self) -> None:
        self.assertEqual(eval_attr("n % 4", self.env), 2)

    def test_binop_pow(self) -> None:
        self.assertEqual(eval_attr("n ** 2", self.env), 100)

    def test_boolop_and(self) -> None:
        self.assertIs(eval_attr("flag and n > 3", self.env), True)

    def test_boolop_or(self) -> None:
        self.assertIs(eval_attr("n > 100 or flag", self.env), True)

    def test_compare_all_operators(self) -> None:
        self.assertIs(eval_attr("n < 11", self.env), True)
        self.assertIs(eval_attr("n <= 10", self.env), True)
        self.assertIs(eval_attr("n > 11", self.env), False)
        self.assertIs(eval_attr("n >= 10", self.env), True)
        self.assertIs(eval_attr("n == 10", self.env), True)
        self.assertIs(eval_attr("n != 10", self.env), False)

    def test_chained_compare(self) -> None:
        self.assertIs(eval_attr("1 < n < 100", self.env), True)
        self.assertIs(eval_attr("1 < n < 5", self.env), False)

    def test_subscript_integer(self) -> None:
        self.assertEqual(eval_attr("x[2]", self.env), 3)

    def test_subscript_negative(self) -> None:
        self.assertEqual(eval_attr("x[-1]", self.env), 4)

    def test_subscript_slice(self) -> None:
        self.assertEqual(eval_attr("x[1:3]", self.env).to_nested(), [2, 3])

    def test_subscript_tuple_key(self) -> None:
        self.assertEqual(eval_attr("m[1, 0]", self.env), 4)

    def test_subscript_row(self) -> None:
        self.assertEqual(eval_attr("m[0]", self.env).to_nested(), [1, 2, 3])

    def test_subscript_python_list(self) -> None:
        self.assertEqual(eval_attr("items[1]", self.env), 8)

    def test_tuple_literal(self) -> None:
        self.assertEqual(eval_attr("(1, 2, 3)", self.env), (1, 2, 3))

    def test_list_literal(self) -> None:
        self.assertEqual(eval_attr("[1, 2, 3]", self.env), [1, 2, 3])

    def test_nested_expression(self) -> None:
        self.assertEqual(eval_attr("(n - 1) * 2 + 3", self.env), 21)


class RejectedConstructTests(unittest.TestCase):
    """The whitelist is a security boundary; every escape hatch must be refused."""

    def setUp(self) -> None:
        self.env = env()

    def assert_refused(self, text: str) -> None:
        with self.assertRaises(ExprError, msg=f"{text!r} was not refused"):
            eval_attr(text, self.env)

    def test_attribute_access_refused(self) -> None:
        self.assert_refused("x.shape")
        self.assert_refused("n.real")

    def test_function_call_refused(self) -> None:
        self.assert_refused("len(x)")
        self.assert_refused("print(1)")
        self.assert_refused("tl.load(x)")

    def test_lambda_refused(self) -> None:
        self.assert_refused("lambda: 1")
        self.assert_refused("(lambda v: v)(1)")

    def test_comprehensions_refused(self) -> None:
        self.assert_refused("[i for i in items]")
        self.assert_refused("{i for i in items}")
        self.assert_refused("{i: i for i in items}")
        self.assert_refused("(i for i in items)")

    def test_dunder_names_refused(self) -> None:
        self.assert_refused("__import__")
        self.assert_refused("__builtins__")
        self.assert_refused("__class__")

    def test_dunder_refused_even_when_bound(self) -> None:
        hostile = dict(self.env)
        hostile["__secret__"] = "boom"
        with self.assertRaises(ExprError):
            eval_attr("__secret__", hostile)

    def test_dict_and_set_literals_refused(self) -> None:
        self.assert_refused("{1: 2}")
        self.assert_refused("{1, 2}")

    def test_conditional_expression_refused(self) -> None:
        self.assert_refused("1 if flag else 2")

    def test_fstring_refused(self) -> None:
        self.assert_refused('f"{n}"')

    def test_walrus_refused(self) -> None:
        self.assert_refused("(q := 1)")

    def test_starred_refused(self) -> None:
        self.assert_refused("[*items]")

    def test_bitwise_operators_refused(self) -> None:
        self.assert_refused("n & 1")
        self.assert_refused("n | 1")
        self.assert_refused("n ^ 1")
        self.assert_refused("n << 1")
        self.assert_refused("~n")

    def test_membership_and_identity_refused(self) -> None:
        self.assert_refused("n in items")
        self.assert_refused("n is None")

    def test_syntax_error_refused(self) -> None:
        self.assert_refused("n +")
        self.assert_refused("((")

    def test_statement_refused(self) -> None:
        self.assert_refused("q = 1")
        self.assert_refused("import os")

    def test_empty_expression_refused(self) -> None:
        self.assert_refused("")
        self.assert_refused("   ")

    def test_non_string_refused(self) -> None:
        with self.assertRaises(ExprError):
            eval_attr(5, self.env)  # type: ignore[arg-type]

    def test_unknown_name_refused(self) -> None:
        self.assert_refused("nowhere")

    def test_division_by_zero_becomes_exprerror(self) -> None:
        self.assert_refused("n / 0")
        self.assert_refused("n // 0")

    def test_bad_operand_types_become_exprerror(self) -> None:
        hostile = dict(self.env)
        hostile["text"] = "abc"
        with self.assertRaises(ExprError):
            eval_attr("text - 1", hostile)

    def test_out_of_range_subscript_refused(self) -> None:
        self.assert_refused("x[99]")
        self.assert_refused("m[0, 1, 2]")


class TileOperandTests(unittest.TestCase):
    """Tile op Tile and Tile op scalar both work and broadcast."""

    def setUp(self) -> None:
        self.env = env()

    def test_tile_plus_scalar(self) -> None:
        result = eval_attr("x + 1", self.env)
        self.assertEqual(result.to_nested(), [2, 3, 4, 5])

    def test_scalar_plus_tile(self) -> None:
        result = eval_attr("1 + x", self.env)
        self.assertEqual(result.to_nested(), [2, 3, 4, 5])

    def test_tile_plus_tile(self) -> None:
        result = eval_attr("x + y", self.env)
        self.assertEqual(result.to_nested(), [11.0, 22.0, 33.0, 44.0])
        self.assertEqual(result.dtype, "f32")

    def test_tile_division_promotes_to_float(self) -> None:
        result = eval_attr("x / 2", self.env)
        self.assertEqual(result.dtype, "f64")
        self.assertEqual(result.to_nested(), [0.5, 1.0, 1.5, 2.0])

    def test_broadcasting_across_ranks(self) -> None:
        local = dict(self.env)
        local["col"] = Tile.from_nested([[10], [20]])
        local["row"] = Tile.from_nested([1, 2, 3])
        result = eval_attr("col + row", local)
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result.to_nested(), [[11, 12, 13], [21, 22, 23]])

    def test_tile_comparison_yields_bool_tile(self) -> None:
        result = eval_attr("x < 3", self.env)
        self.assertEqual(result.dtype, "bool")
        self.assertEqual(result.to_nested(), [True, True, False, False])

    def test_tile_boolop_yields_bool_tile(self) -> None:
        result = eval_attr("(x > 1) and (x < 4)", self.env)
        self.assertEqual(result.dtype, "bool")
        self.assertEqual(result.to_nested(), [False, True, True, False])

    def test_unary_negate_tile(self) -> None:
        result = eval_attr("-x", self.env)
        self.assertEqual(result.to_nested(), [-1, -2, -3, -4])

    def test_broadcast_mismatch_surfaces(self) -> None:
        local = dict(self.env)
        local["short"] = Tile.from_nested([1, 2])
        with self.assertRaises((ExprError, ShapeError)):
            eval_attr("x + short", local)


class ParseNamesTests(unittest.TestCase):
    """parse_names feeds the scheduler and must never raise."""

    def test_simple_names(self) -> None:
        self.assertEqual(parse_names("a + b"), ["a", "b"])

    def test_source_order_without_duplicates(self) -> None:
        self.assertEqual(parse_names("a + b + a"), ["a", "b"])

    def test_literals_are_not_names(self) -> None:
        self.assertEqual(parse_names("true"), [])
        self.assertEqual(parse_names("a and false"), ["a"])

    def test_ignores_the_whitelist(self) -> None:
        self.assertEqual(sorted(parse_names("tl.load(x)")), ["tl", "x"])

    def test_leading_identifier_of_a_pointer_expression_comes_first(self) -> None:
        self.assertEqual(parse_names("A + i * n + j")[0], "A")
        self.assertEqual(parse_names("base + stride * row + col")[0], "base")

    def test_syntax_error_returns_empty(self) -> None:
        self.assertEqual(parse_names("int i = 0; i < n; i += 1"), [])
        self.assertEqual(parse_names("(("), [])

    def test_empty_and_none(self) -> None:
        self.assertEqual(parse_names(""), [])
        self.assertEqual(parse_names("   "), [])
        self.assertEqual(parse_names(None), [])  # type: ignore[arg-type]

    def test_pointer_expression(self) -> None:
        self.assertEqual(parse_names("x_ptr + offsets"), ["x_ptr", "offsets"])


class OperatorHelperTests(unittest.TestCase):
    """The apply_* helpers are the single source of truth for arithmetic."""

    def test_apply_binary_scalars(self) -> None:
        self.assertEqual(apply_binary("add", 2, 3), 5)
        self.assertEqual(apply_binary("sub", 2, 3), -1)
        self.assertEqual(apply_binary("mul", 2, 3), 6)
        self.assertEqual(apply_binary("div", 3, 2), 1.5)
        self.assertEqual(apply_binary("floordiv", 7, 2), 3)
        self.assertEqual(apply_binary("mod", 7, 2), 1)
        self.assertEqual(apply_binary("pow", 2, 3), 8)

    def test_apply_binary_tiles(self) -> None:
        left = Tile.from_flat([1, 2], (2,), "i32")
        self.assertEqual(apply_binary("add", left, 1).to_nested(), [2, 3])

    def test_apply_binary_unknown_operator(self) -> None:
        with self.assertRaises(ExprError):
            apply_binary("xor", 1, 2)

    def test_apply_compare_scalars_return_bool(self) -> None:
        self.assertIs(apply_compare("lt", 1, 2), True)
        self.assertIs(apply_compare("ne", 1, 1), False)

    def test_apply_compare_tile_returns_bool_tile(self) -> None:
        left = Tile.from_flat([1, 5], (2,), "i32")
        result = apply_compare("gt", left, 3)
        self.assertEqual(result.dtype, "bool")
        self.assertEqual(result.to_nested(), [False, True])

    def test_apply_boolean_over_many_operands(self) -> None:
        self.assertIs(apply_boolean("and", [True, True, False]), False)
        self.assertIs(apply_boolean("or", [False, False, True]), True)

    def test_apply_boolean_single_operand_casts(self) -> None:
        self.assertIs(apply_boolean("and", [3]), True)
        result = apply_boolean("or", [Tile.from_flat([0, 2], (2,), "i32")])
        self.assertEqual(result.dtype, "bool")
        self.assertEqual(result.to_nested(), [False, True])

    def test_apply_boolean_needs_operands(self) -> None:
        with self.assertRaises(ExprError):
            apply_boolean("and", [])
        with self.assertRaises(ExprError):
            apply_boolean("nand", [True])

    def test_apply_unary(self) -> None:
        self.assertEqual(apply_unary("neg", 3), -3)
        self.assertEqual(apply_unary("pos", -3), -3)
        self.assertIs(apply_unary("not", 0), True)
        with self.assertRaises(ExprError):
            apply_unary("inv", 3)

    def test_apply_unary_on_bool_tile_promotes(self) -> None:
        result = apply_unary("neg", Tile.from_flat([True, False], (2,), "bool"))
        self.assertEqual(result.dtype, "i32")
        self.assertEqual(result.to_nested(), [-1, 0])


class TruthyTests(unittest.TestCase):
    """truthy is what drives if and while conditions."""

    def test_python_scalars(self) -> None:
        self.assertIs(truthy(1), True)
        self.assertIs(truthy(0), False)
        self.assertIs(truthy(None), False)

    def test_size_one_tile_uses_its_element(self) -> None:
        self.assertIs(truthy(Tile.scalar(0)), False)
        self.assertIs(truthy(Tile.scalar(3)), True)
        self.assertIs(truthy(Tile.from_flat([1], (1,), "i32")), True)

    def test_empty_tile_is_false(self) -> None:
        self.assertIs(truthy(Tile.from_flat([], (0,), "i32")), False)

    def test_larger_tile_requires_every_lane(self) -> None:
        self.assertIs(truthy(Tile.from_flat([True, True], (2,), "bool")), True)
        self.assertIs(truthy(Tile.from_flat([True, False], (2,), "bool")), False)


class SubscriptHelperTests(unittest.TestCase):
    """subscript works on tiles and plain sequences and refuses everything else."""

    def test_tile(self) -> None:
        self.assertEqual(subscript(Tile.from_nested([1, 2, 3]), 1), 2)

    def test_sequence(self) -> None:
        self.assertEqual(subscript([1, 2, 3], 2), 3)
        self.assertEqual(subscript((1, 2, 3), 0), 1)

    def test_bad_target(self) -> None:
        with self.assertRaises(ExprError):
            subscript(5, 0)

    def test_bad_key(self) -> None:
        with self.assertRaises(ExprError):
            subscript([1, 2, 3], 9)


if __name__ == "__main__":
    unittest.main()
