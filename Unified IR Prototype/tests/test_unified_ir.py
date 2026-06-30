import unittest
from pathlib import Path

from unified_ir.parsers import parse_source
from unified_ir.line_format import to_line_format


ROOT = Path(__file__).resolve().parents[1]


class UnifiedIRTests(unittest.TestCase):
    def test_triton_add_emits_add_load_store(self):
        source_path = ROOT / "examples" / "triton_add.py"
        program = parse_source(source_path.read_text(encoding="utf-8"), str(source_path), "triton")
        text = to_line_format(program)
        self.assertIn("name=vector_add", text)
        self.assertIn("name=scaled_vector_add", text)
        self.assertIn("| add", text)
        self.assertIn("out=scaled_x", text)
        self.assertIn("| load", text)
        self.assertIn("| store", text)
        self.assertLessEqual(len(program.ops), 220)

    def test_cutile_add_emits_add_load_store(self):
        source_path = ROOT / "examples" / "cutile_add.cutile"
        program = parse_source(source_path.read_text(encoding="utf-8"), str(source_path), "cutile")
        text = to_line_format(program)
        self.assertIn("source_lang=cutile", text)
        self.assertIn("name=vector_add", text)
        self.assertIn("name=scaled_vector_add", text)
        self.assertIn("out=z_tile", text)
        self.assertIn("out=scaled_x", text)
        self.assertIn("| add", text)
        self.assertIn("| load", text)
        self.assertIn("| store", text)

    def test_operator_budget_is_enforced(self):
        source = "kernel too_big() {\n" + "\n".join(f"a{i} = a{i} + 1;" for i in range(5)) + "\n}"
        with self.assertRaises(ValueError):
            parse_source(source, "too_big.cutile", "cutile", max_ops=3)


if __name__ == "__main__":
    unittest.main()
