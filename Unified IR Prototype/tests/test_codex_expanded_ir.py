import unittest

from unified_ir.line_format import to_line_format
from unified_ir.parsers import parse_source


class CodexExpandedIRTests(unittest.TestCase):
    def test_expanded_cutile_program_covers_many_ops(self):
        source = """
kernel expanded_ops(x, y, z, n, BLOCK) {
  pid = program_id(0);
  block_start = pid * BLOCK;
  offsets0 = arange(0, BLOCK);
  offsets = block_start + offsets0;
  zero = 0;
  one = 1;
  two = 2;
  three = 3;
  four = 4;
  lane0 = offsets + zero;
  lane1 = lane0 + one;
  lane2 = lane1 - two;
  lane3 = lane2 * three;
  lane4 = lane3 / four;
  mask_lt = offsets < n;
  mask_le = offsets <= n;
  mask_gt = n > offsets;
  mask_ge = offsets >= zero;
  mask_eq = lane0 == lane1;
  mask_ne = lane0 != lane2;
  mask_and = mask_lt && mask_ge;
  mask_or = mask_eq || mask_ne;
  x_tile = load(x + offsets, mask=mask_lt, other=0.0);
  y_tile = load(y + offsets, mask=mask_lt, other=0.0);
  sum_tile = x_tile + y_tile;
  diff_tile = sum_tile - y_tile;
  prod_tile = diff_tile * x_tile;
  div_tile = prod_tile / two;
  call_add = add(div_tile, sum_tile);
  call_sub = sub(call_add, y_tile);
  call_mul = mul(call_sub, x_tile);
  call_div = div(call_mul, two);
  max_tile = max(call_div, sum_tile);
  min_tile = min(max_tile, diff_tile);
  selected_tile = select(mask_and, max_tile, min_tile);
  exp_tile = exp(selected_tile);
  sqrt_tile = sqrt(exp_tile);
  dot_tile = dot(x_tile, y_tile);
  filled = full(BLOCK, 0.0);
  final_seed = sqrt_tile + dot_tile;
  final_tile = final_seed + filled;
  custom_tile_op(final_tile);
  store(z + offsets, final_tile, mask=mask_and);
}
""".strip()
        program = parse_source(source, "expanded_ops.cutile", "cutile")
        text = to_line_format(program)
        opcodes = {op.opcode for op in program.ops}
        expected = {
            "kernel",
            "program_id",
            "mul",
            "arange",
            "add",
            "assign",
            "sub",
            "div",
            "lt",
            "le",
            "gt",
            "ge",
            "eq",
            "ne",
            "and",
            "or",
            "load",
            "max",
            "min",
            "select",
            "exp",
            "sqrt",
            "dot",
            "fill",
            "call",
            "store",
        }

        self.assertGreaterEqual(len(source.splitlines()), 45)
        self.assertGreaterEqual(len(program.ops), 40)
        self.assertLessEqual(len(program.ops), 220)
        self.assertTrue(expected.issubset(opcodes), expected - opcodes)
        self.assertIn("out=final_tile", text)
        self.assertIn("| store", text)

    def test_expanded_triton_program_covers_python_ast_ops(self):
        source = """
import triton
import triton.language as tl


@triton.jit
def expanded_triton(a, b, c, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    base = pid * BLOCK
    offsets0 = tl.arange(0, BLOCK)
    offsets = base + offsets0
    x = tl.load(a + offsets)
    y = tl.load(b + offsets)
    addv = x + y
    subv = addv - y
    mulv = subv * x
    divv = mulv / 2.0
    floordivv = pid // 2
    modv = pid % 2
    powv = pid ** 2
    lt_mask = offsets < n
    ge_mask = offsets >= 0
    both = lt_mask and ge_mask
    chosen = tl.where(lt_mask, addv, subv)
    ex = tl.exp(chosen)
    root = tl.sqrt(ex)
    mx = tl.maximum(root, chosen)
    mn = tl.minimum(mx, root)
    prod = tl.dot(x, y)
    final = mn + prod
    tl.store(c + offsets, final)
""".strip()
        program = parse_source(source, "expanded_triton.py", "triton")
        opcodes = {op.opcode for op in program.ops}
        expected = {
            "kernel",
            "program_id",
            "mul",
            "arange",
            "add",
            "load",
            "sub",
            "div",
            "floordiv",
            "mod",
            "pow",
            "lt",
            "ge",
            "and",
            "select",
            "exp",
            "sqrt",
            "max",
            "min",
            "dot",
            "store",
        }

        self.assertGreaterEqual(len(source.splitlines()), 25)
        self.assertGreaterEqual(len(program.ops), 20)
        self.assertLessEqual(len(program.ops), 220)
        self.assertTrue(expected.issubset(opcodes), expected - opcodes)


if __name__ == "__main__":
    unittest.main()
