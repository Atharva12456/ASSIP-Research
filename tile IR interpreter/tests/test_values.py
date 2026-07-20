from __future__ import annotations

import unittest

from tile_interp.values import (
    ShapeError,
    Tile,
    arith_dtype,
    broadcast_shapes,
    float_dtype,
    infer_dtype,
    result_dtype,
)


class BroadcastShapeTests(unittest.TestCase):
    """broadcast_shapes must follow numpy's right-aligned rules exactly."""

    def test_identical_shapes(self) -> None:
        self.assertEqual(broadcast_shapes((2, 3), (2, 3)), (2, 3))

    def test_stretches_leading_one(self) -> None:
        self.assertEqual(broadcast_shapes((1, 3), (4, 3)), (4, 3))

    def test_stretches_trailing_one(self) -> None:
        self.assertEqual(broadcast_shapes((4, 1), (4, 3)), (4, 3))

    def test_ragged_ranks_right_align(self) -> None:
        self.assertEqual(broadcast_shapes((3,), (2, 3)), (2, 3))
        self.assertEqual(broadcast_shapes((5, 1, 3), (4, 3)), (5, 4, 3))

    def test_zero_dim_is_neutral(self) -> None:
        self.assertEqual(broadcast_shapes((), (2, 3)), (2, 3))
        self.assertEqual(broadcast_shapes((2, 3), ()), (2, 3))
        self.assertEqual(broadcast_shapes((), ()), ())

    def test_outer_product_shape(self) -> None:
        self.assertEqual(broadcast_shapes((3, 1), (4,)), (3, 4))

    def test_genuine_mismatch_raises(self) -> None:
        with self.assertRaises(ShapeError):
            broadcast_shapes((2, 3), (2, 4))
        with self.assertRaises(ShapeError):
            broadcast_shapes((3,), (2,))

    def test_zero_length_axis_only_matches_itself(self) -> None:
        self.assertEqual(broadcast_shapes((0,), (1,)), (0,))
        with self.assertRaises(ShapeError):
            broadcast_shapes((0,), (3,))


class BroadcastToTests(unittest.TestCase):
    """Tile.broadcast_to materialises the stretched data in row-major order."""

    def test_row_vector_to_matrix(self) -> None:
        tile = Tile.from_nested([1, 2, 3])
        self.assertEqual(tile.broadcast_to((2, 3)).to_nested(), [[1, 2, 3], [1, 2, 3]])

    def test_column_vector_to_matrix(self) -> None:
        tile = Tile.from_nested([[1], [2]])
        self.assertEqual(tile.broadcast_to((2, 3)).to_nested(), [[1, 1, 1], [2, 2, 2]])

    def test_scalar_to_matrix(self) -> None:
        self.assertEqual(Tile.scalar(7).broadcast_to((2, 2)).to_nested(), [[7, 7], [7, 7]])

    def test_shrinking_rank_raises(self) -> None:
        with self.assertRaises(ShapeError):
            Tile.from_nested([[1, 2], [3, 4]]).broadcast_to((2,))

    def test_incompatible_axis_raises(self) -> None:
        with self.assertRaises(ShapeError):
            Tile.from_nested([1, 2, 3]).broadcast_to((2, 4))


class DtypeTests(unittest.TestCase):
    """Promotion is max-rank over bool < i32 < i64 < f32 < f64."""

    def test_rank_order(self) -> None:
        self.assertEqual(result_dtype("bool", "i32"), "i32")
        self.assertEqual(result_dtype("i32", "i64"), "i64")
        self.assertEqual(result_dtype("i64", "f32"), "f32")
        self.assertEqual(result_dtype("f32", "f64"), "f64")
        self.assertEqual(result_dtype("f64", "bool"), "f64")

    def test_promotion_is_symmetric(self) -> None:
        for left in ("bool", "i32", "i64", "f32", "f64"):
            for right in ("bool", "i32", "i64", "f32", "f64"):
                self.assertEqual(result_dtype(left, right), result_dtype(right, left))

    def test_arith_promotes_bool_pairs_to_int(self) -> None:
        self.assertEqual(arith_dtype("bool", "bool"), "i32")
        self.assertEqual(arith_dtype("bool", "f32"), "f32")

    def test_float_dtype_lifts_integers(self) -> None:
        self.assertEqual(float_dtype("i32"), "f64")
        self.assertEqual(float_dtype("i64"), "f64")
        self.assertEqual(float_dtype("bool"), "f64")
        self.assertEqual(float_dtype("f32"), "f32")

    def test_unknown_dtype_raises(self) -> None:
        with self.assertRaises(ShapeError):
            result_dtype("f16", "f32")
        with self.assertRaises(ShapeError):
            Tile.zeros((2,), "int8")

    def test_infer_dtype(self) -> None:
        self.assertEqual(infer_dtype(True), "bool")
        self.assertEqual(infer_dtype(3), "i64")
        self.assertEqual(infer_dtype(3.0), "f64")
        with self.assertRaises(ShapeError):
            infer_dtype("nope")

    def test_int_division_produces_float(self) -> None:
        left = Tile.from_flat([1, 2, 3], (3,), "i32")
        right = Tile.from_flat([2, 2, 2], (3,), "i32")
        quotient = left.zip_with(right, lambda a, b: a / b, float_dtype(result_dtype("i32", "i32")))
        self.assertEqual(quotient.dtype, "f64")
        self.assertEqual(quotient.to_nested(), [0.5, 1.0, 1.5])

    def test_floordiv_of_ints_stays_int(self) -> None:
        left = Tile.from_flat([7, 8], (2,), "i32")
        right = Tile.from_flat([2, 3], (2,), "i32")
        floored = left.zip_with(right, lambda a, b: a // b, arith_dtype("i32", "i32"))
        self.assertEqual(floored.dtype, "i32")
        self.assertEqual(floored.to_nested(), [3, 2])


class ConstructionTests(unittest.TestCase):
    """Constructors, shape bookkeeping, and conversions."""

    def test_scalar_is_zero_dimensional(self) -> None:
        tile = Tile.scalar(2.5)
        self.assertEqual(tile.shape, ())
        self.assertEqual(tile.ndim, 0)
        self.assertEqual(tile.size, 1)
        self.assertEqual(tile.item(), 2.5)
        self.assertEqual(tile.to_nested(), 2.5)

    def test_zeros_defaults_to_f32(self) -> None:
        tile = Tile.zeros((2, 3))
        self.assertEqual(tile.dtype, "f32")
        self.assertEqual(tile.shape, (2, 3))
        self.assertEqual(tile.data, [0.0] * 6)

    def test_full_infers_dtype(self) -> None:
        self.assertEqual(Tile.full((2,), 3).dtype, "i64")
        self.assertEqual(Tile.full((2,), 3.0).dtype, "f64")
        self.assertEqual(Tile.full((2,), 3, "f32").dtype, "f32")

    def test_arange(self) -> None:
        self.assertEqual(Tile.arange(0, 5).to_nested(), [0, 1, 2, 3, 4])
        self.assertEqual(Tile.arange(0, 5).dtype, "i32")
        self.assertEqual(Tile.arange(1, 7, 2).to_nested(), [1, 3, 5])
        self.assertEqual(Tile.arange(0, 0).shape, (0,))
        with self.assertRaises(ShapeError):
            Tile.arange(0, 4, 0)

    def test_from_nested_round_trips(self) -> None:
        nested = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        tile = Tile.from_nested(nested)
        self.assertEqual(tile.shape, (3, 2))
        self.assertEqual(tile.to_nested(), nested)

    def test_from_nested_rejects_ragged(self) -> None:
        with self.assertRaises(ShapeError):
            Tile.from_nested([[1, 2], [3]])
        with self.assertRaises(ShapeError):
            Tile.from_nested([[1, 2], 3])

    def test_from_flat_checks_size(self) -> None:
        self.assertEqual(Tile.from_flat([1, 2, 3, 4], (2, 2)).to_nested(), [[1, 2], [3, 4]])
        with self.assertRaises(ShapeError):
            Tile.from_flat([1, 2, 3], (2, 2))

    def test_item_requires_size_one(self) -> None:
        with self.assertRaises(ShapeError):
            Tile.from_nested([1, 2]).item()
        self.assertEqual(Tile.from_nested([[5]]).item(), 5)

    def test_reshape_preserves_row_major_data(self) -> None:
        tile = Tile.from_nested([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(tile.reshape((3, 2)).to_nested(), [[1, 2], [3, 4], [5, 6]])
        self.assertEqual(tile.reshape((6,)).to_nested(), [1, 2, 3, 4, 5, 6])
        with self.assertRaises(ShapeError):
            tile.reshape((4, 2))

    def test_copy_is_independent(self) -> None:
        tile = Tile.from_nested([1, 2, 3])
        clone = tile.copy()
        clone.data[0] = 99
        self.assertEqual(tile.data[0], 1)

    def test_astype(self) -> None:
        tile = Tile.from_flat([1.7, -2.2], (2,), "f64")
        self.assertEqual(tile.astype("i32").to_nested(), [1, -2])
        self.assertEqual(tile.astype("bool").to_nested(), [True, True])


class ElementwiseTests(unittest.TestCase):
    """map and zip_with, including the broadcasting path."""

    def test_map_keeps_shape(self) -> None:
        tile = Tile.from_nested([[1, 2], [3, 4]])
        self.assertEqual(tile.map(lambda x: x * 2).to_nested(), [[2, 4], [6, 8]])

    def test_map_can_change_dtype(self) -> None:
        tile = Tile.from_nested([1, 2, 3])
        self.assertEqual(tile.map(lambda x: x > 1, "bool").to_nested(), [False, True, True])

    def test_zip_with_broadcasts(self) -> None:
        row = Tile.from_nested([1, 2, 3])
        column = Tile.from_nested([[10], [20]])
        summed = row.zip_with(column, lambda a, b: a + b)
        self.assertEqual(summed.shape, (2, 3))
        self.assertEqual(summed.to_nested(), [[11, 12, 13], [21, 22, 23]])

    def test_zip_with_promotes_dtype(self) -> None:
        left = Tile.from_flat([1, 2], (2,), "i32")
        right = Tile.from_flat([0.5, 0.5], (2,), "f64")
        self.assertEqual(left.zip_with(right, lambda a, b: a + b).dtype, "f64")

    def test_zip_with_mismatch_raises(self) -> None:
        with self.assertRaises(ShapeError):
            Tile.from_nested([1, 2, 3]).zip_with(Tile.from_nested([1, 2]), lambda a, b: a + b)

    def test_zip_with_requires_tile(self) -> None:
        with self.assertRaises(ShapeError):
            Tile.from_nested([1, 2]).zip_with(3, lambda a, b: a + b)  # type: ignore[arg-type]


class ReduceTests(unittest.TestCase):
    """reduce over all elements and over each individual axis."""

    def setUp(self) -> None:
        self.tile = Tile.from_nested([[1, 2, 3], [4, 5, 6]])

    def test_reduce_all(self) -> None:
        total = self.tile.reduce(lambda a, b: a + b)
        self.assertEqual(total.shape, ())
        self.assertEqual(total.item(), 21)

    def test_reduce_axis_zero(self) -> None:
        result = self.tile.reduce(lambda a, b: a + b, 0)
        self.assertEqual(result.shape, (3,))
        self.assertEqual(result.to_nested(), [5, 7, 9])

    def test_reduce_axis_one(self) -> None:
        result = self.tile.reduce(lambda a, b: a + b, 1)
        self.assertEqual(result.shape, (2,))
        self.assertEqual(result.to_nested(), [6, 15])

    def test_reduce_negative_axis(self) -> None:
        self.assertEqual(self.tile.reduce(lambda a, b: a + b, -1).to_nested(), [6, 15])

    def test_reduce_max_over_axis(self) -> None:
        self.assertEqual(self.tile.reduce(max, 0).to_nested(), [4, 5, 6])
        self.assertEqual(self.tile.reduce(max, 1).to_nested(), [3, 6])

    def test_reduce_with_init(self) -> None:
        self.assertEqual(self.tile.reduce(lambda a, b: a + b, None, 100).item(), 121)

    def test_reduce_three_dimensional(self) -> None:
        tile = Tile.from_flat(list(range(24)), (2, 3, 4))
        self.assertEqual(tile.reduce(lambda a, b: a + b, 0).shape, (3, 4))
        self.assertEqual(tile.reduce(lambda a, b: a + b, 1).shape, (2, 4))
        self.assertEqual(tile.reduce(lambda a, b: a + b, 2).shape, (2, 3))
        self.assertEqual(tile.reduce(lambda a, b: a + b, 2).to_nested()[0][0], 0 + 1 + 2 + 3)
        self.assertEqual(tile.reduce(lambda a, b: a + b).item(), sum(range(24)))

    def test_reduce_bad_axis_raises(self) -> None:
        with self.assertRaises(ShapeError):
            self.tile.reduce(lambda a, b: a + b, 2)

    def test_reduce_empty_needs_init(self) -> None:
        empty = Tile.from_flat([], (0,), "i32")
        with self.assertRaises(ShapeError):
            empty.reduce(lambda a, b: a + b)
        self.assertEqual(empty.reduce(lambda a, b: a + b, None, 0).item(), 0)


class MatmulTests(unittest.TestCase):
    """matmul against results computed by hand."""

    def test_two_by_three_times_three_by_two(self) -> None:
        left = Tile.from_nested([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        right = Tile.from_nested([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
        product = left.matmul(right)
        self.assertEqual(product.shape, (2, 2))
        self.assertEqual(product.to_nested(), [[58.0, 64.0], [139.0, 154.0]])

    def test_identity(self) -> None:
        matrix = Tile.from_nested([[1.0, 2.0], [3.0, 4.0]])
        identity = Tile.from_nested([[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(matrix.matmul(identity).to_nested(), matrix.to_nested())

    def test_matrix_times_vector(self) -> None:
        matrix = Tile.from_nested([[1.0, 2.0], [3.0, 4.0]])
        vector = Tile.from_nested([1.0, 1.0])
        product = matrix.matmul(vector)
        self.assertEqual(product.shape, (2,))
        self.assertEqual(product.to_nested(), [3.0, 7.0])

    def test_vector_times_matrix(self) -> None:
        matrix = Tile.from_nested([[1.0, 2.0], [3.0, 4.0]])
        vector = Tile.from_nested([1.0, 1.0])
        product = vector.matmul(matrix)
        self.assertEqual(product.shape, (2,))
        self.assertEqual(product.to_nested(), [4.0, 6.0])

    def test_vector_dot_vector_is_scalar(self) -> None:
        left = Tile.from_nested([1.0, 2.0, 3.0])
        right = Tile.from_nested([4.0, 5.0, 6.0])
        product = left.matmul(right)
        self.assertEqual(product.shape, ())
        self.assertEqual(product.item(), 32.0)

    def test_integer_matmul_stays_integer(self) -> None:
        left = Tile.from_flat([1, 2, 3, 4], (2, 2), "i32")
        self.assertEqual(left.matmul(left).dtype, "i32")
        self.assertEqual(left.matmul(left).to_nested(), [[7, 10], [15, 22]])

    def test_inner_dimension_mismatch_raises(self) -> None:
        with self.assertRaises(ShapeError):
            Tile.from_nested([[1.0, 2.0]]).matmul(Tile.from_nested([[1.0, 2.0]]))

    def test_zero_dimensional_operand_raises(self) -> None:
        with self.assertRaises(ShapeError):
            Tile.scalar(2.0).matmul(Tile.from_nested([1.0, 2.0]))

    def test_three_dimensional_operand_raises(self) -> None:
        with self.assertRaises(ShapeError):
            Tile.zeros((2, 2, 2)).matmul(Tile.zeros((2, 2)))


class TransposeTests(unittest.TestCase):
    """transpose reverses the axes."""

    def test_two_dimensional(self) -> None:
        tile = Tile.from_nested([[1, 2, 3], [4, 5, 6]])
        flipped = tile.transpose()
        self.assertEqual(flipped.shape, (3, 2))
        self.assertEqual(flipped.to_nested(), [[1, 4], [2, 5], [3, 6]])

    def test_double_transpose_is_identity(self) -> None:
        tile = Tile.from_nested([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(tile.transpose().transpose().to_nested(), tile.to_nested())

    def test_one_dimensional_is_a_copy(self) -> None:
        tile = Tile.from_nested([1, 2, 3])
        flipped = tile.transpose()
        self.assertEqual(flipped.to_nested(), [1, 2, 3])
        flipped.data[0] = 9
        self.assertEqual(tile.data[0], 1)


class DigestTests(unittest.TestCase):
    """digest is a stable content hash, not an identity hash."""

    def test_same_content_from_separate_constructions(self) -> None:
        a = Tile.from_nested([[1.0, 2.0], [3.0, 4.0]])
        b = Tile.from_flat([1.0, 2.0, 3.0, 4.0], (2, 2), "f64")
        self.assertIsNot(a, b)
        self.assertEqual(a.digest(), b.digest())

    def test_digest_is_eight_hex_chars(self) -> None:
        digest = Tile.from_nested([1, 2, 3]).digest()
        self.assertEqual(len(digest), 8)
        self.assertTrue(all(char in "0123456789abcdef" for char in digest))

    def test_repeated_calls_agree(self) -> None:
        tile = Tile.from_nested([1.5, 2.5])
        self.assertEqual(tile.digest(), tile.digest())

    def test_different_values_differ(self) -> None:
        a = Tile.from_nested([1.0, 2.0])
        b = Tile.from_nested([1.0, 2.5])
        self.assertNotEqual(a.digest(), b.digest())

    def test_different_shapes_differ(self) -> None:
        a = Tile.from_flat([1, 2, 3, 4], (2, 2), "i32")
        b = Tile.from_flat([1, 2, 3, 4], (4,), "i32")
        self.assertNotEqual(a.digest(), b.digest())

    def test_different_dtypes_differ(self) -> None:
        a = Tile.from_flat([1, 2], (2,), "i32")
        b = Tile.from_flat([1, 2], (2,), "i64")
        self.assertNotEqual(a.digest(), b.digest())

    def test_rounding_noise_hashes_the_same(self) -> None:
        a = Tile.from_nested([0.1 + 0.2])
        b = Tile.from_nested([0.3])
        self.assertEqual(a.digest(), b.digest())

    def test_negative_zero_matches_zero(self) -> None:
        a = Tile.from_nested([-0.0])
        b = Tile.from_nested([0.0])
        self.assertEqual(a.digest(), b.digest())

    def test_describe_format(self) -> None:
        self.assertTrue(Tile.zeros((2, 3), "f32").describe().startswith("2x3:f32@"))
        self.assertTrue(Tile.scalar(1.0).describe().startswith("scalar:f64@"))


class AllcloseTests(unittest.TestCase):
    """allclose compares numerically, never with ==."""

    def test_exact_match(self) -> None:
        self.assertTrue(Tile.from_nested([1.0, 2.0]).allclose(Tile.from_nested([1.0, 2.0])))

    def test_within_tolerance(self) -> None:
        a = Tile.from_nested([1.0])
        b = Tile.from_nested([1.0 + 1e-15])
        self.assertTrue(a.allclose(b))

    def test_outside_tolerance(self) -> None:
        a = Tile.from_nested([1.0])
        b = Tile.from_nested([1.001])
        self.assertFalse(a.allclose(b))

    def test_broadcasting_comparison(self) -> None:
        self.assertTrue(Tile.scalar(1.0).allclose(Tile.from_nested([1.0, 1.0])))

    def test_shape_mismatch_is_false_not_an_error(self) -> None:
        self.assertFalse(Tile.from_nested([1.0, 2.0]).allclose(Tile.from_nested([1.0, 2.0, 3.0])))

    def test_non_tile_is_false(self) -> None:
        self.assertFalse(Tile.from_nested([1.0]).allclose(1.0))  # type: ignore[arg-type]

    def test_nan_never_matches(self) -> None:
        nan = Tile.from_nested([float("nan")])
        self.assertFalse(nan.allclose(nan))

    def test_infinities_match_themselves(self) -> None:
        pos = Tile.from_nested([float("inf")])
        neg = Tile.from_nested([float("-inf")])
        self.assertTrue(pos.allclose(Tile.from_nested([float("inf")])))
        self.assertFalse(pos.allclose(neg))


class EqualityTests(unittest.TestCase):
    """Tile equality is plain structural equality, safe for assertEqual."""

    def test_equal_tiles(self) -> None:
        self.assertEqual(Tile.from_nested([1, 2]), Tile.from_nested([1, 2]))

    def test_dtype_participates(self) -> None:
        self.assertNotEqual(
            Tile.from_flat([1, 2], (2,), "i32"), Tile.from_flat([1, 2], (2,), "i64")
        )

    def test_equality_returns_a_bool(self) -> None:
        self.assertIsInstance(Tile.from_nested([1, 2]) == Tile.from_nested([1, 2]), bool)


if __name__ == "__main__":
    unittest.main()
