# Unified IR Prototype

This is a small proof-of-concept unified tile IR emitter for Triton-like Python kernels and cuTile-like kernels. It outputs a simple line format where each operator is one line:

```text
0007 | add          | out=z lhs=x rhs=y
```

The prototype is intentionally narrow and fast. It has no third-party dependencies and defaults to a maximum of 220 emitted operator lines with `--max-ops 220`.

## What This Program Does

The program does not execute Triton or cuTile kernels. It reads source code and converts it into a shared line-format IR.

```text
Triton-like source   cuTile-like source
        \              /
         \            /
          unified_ir parser
                |
        line-format tile IR
```

For example, this cuTile input:

```cutile
kernel simple_add() {
  result = 2 + 3;
}
```

emits IR like:

```text
0000 | kernel       | name=simple_add
0001 | add          | out=result lhs=2 rhs=3
```

That means the prototype recognized the operation `result = add(2, 3)`. It does not compute the final value `5`.

## Folder Layout

```text
Unified IR Prototype/
  examples/
    triton_add.py
    cutile_add.cutile
  tests/
    test_unified_ir.py
    test_codex_expanded_ir.py
  unified_ir/
    cli.py
    parsers.py
    ir.py
    line_format.py
    operator_catalog.py
  uir.py
  README.md
```

The important folders are:

- `examples/`: input files you can convert into IR
- `tests/`: unit tests that check the converter
- `unified_ir/`: the implementation of the parser and IR emitter
- `uir.py`: helper script for running the tool from the repo root

## Running From PowerShell

The easiest starting point is the prototype folder:

```powershell
cd "C:\Users\athar\Downloads\ASSIP Research\Unified IR Prototype"
```

Then run one of the example files.

Run the Triton example:

```powershell
python -m unified_ir --lang triton examples/triton_add.py
```

Run the cuTile example:

```powershell
python -m unified_ir --lang cutile examples/cutile_add.cutile
```

You can also run from the repo root with `uir.py`:

```powershell
cd "C:\Users\athar\Downloads\ASSIP Research"
python ".\Unified IR Prototype\uir.py" --lang triton ".\Unified IR Prototype\examples\triton_add.py"
python ".\Unified IR Prototype\uir.py" --lang cutile ".\Unified IR Prototype\examples\cutile_add.cutile"
```

## What The Command Means

```powershell
python -m unified_ir --lang cutile examples/cutile_add.cutile
```

Breakdown:

- `python`: runs Python
- `-m unified_ir`: runs the `unified_ir` package as a program
- `--lang cutile`: tells the parser the input is cuTile-style code
- `examples/cutile_add.cutile`: the input file to convert

Use `--lang triton` for Triton-like Python files and `--lang cutile` for cuTile-like files. You can also use `--lang auto`, but explicit language selection is easier when presenting.

## Saving IR Output

By default, output prints in the terminal. To save the IR to a file, use `--out`:

```powershell
python -m unified_ir --lang auto examples/triton_add.py --out add.lineir
```

Then open `add.lineir` to inspect the generated line-format IR.

## Running Your Own File

Create your own file in `examples/`.

Example cuTile file:

```cutile
kernel my_add() {
  result = 2 + 3;
}
```

Save it as:

```text
examples/my_add.cutile
```

Run it:

```powershell
python -m unified_ir --lang cutile examples/my_add.cutile
```

Example Triton-like file:

```python
import triton
import triton.language as tl


@triton.jit
def my_add(x_ptr, y_ptr, z_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    z = x + y
    tl.store(z_ptr + offsets, z)
```

Save it as:

```text
examples/my_triton_add.py
```

Run it through the converter:

```powershell
python -m unified_ir --lang triton examples/my_triton_add.py
```

Do not run the Triton example directly with `python examples/my_triton_add.py`. That would try to import and execute real Triton. This prototype only needs to parse the file.

## Changing The Operator Limit

The default max IR size is 220 operator lines. To test a smaller or larger limit, use `--max-ops`:

```powershell
python -m unified_ir --lang cutile examples/cutile_add.cutile --max-ops 50
```

If the parser emits more operators than the limit, it raises an error.

## Supported Operators

Show supported prototype opcodes:

```powershell
python -m unified_ir --dump-ops
```

The main supported groups are:

- arithmetic: `add`, `sub`, `mul`, `div`, `floordiv`, `mod`, `pow`
- memory/tile setup: `load`, `store`, `arange`, `program_id`, `fill`
- comparisons: `lt`, `le`, `gt`, `ge`, `eq`, `ne`
- logical/control: `and`, `or`, `if`, `for`, `while`
- math/select: `dot`, `exp`, `sqrt`, `max`, `min`, `select`
- utility: `assign`, `call`, `kernel`, `return`

## Running Tests

From the prototype folder:

```powershell
cd "C:\Users\athar\Downloads\ASSIP Research\Unified IR Prototype"
```

Run all tests:

```powershell
python -m unittest discover -s tests
```

Expected result:

```text
Ran 5 tests
OK
```

Run only the base test file:

```powershell
python -m unittest tests.test_unified_ir
```

Run only the expanded Codex test file:

```powershell
python -m unittest tests.test_codex_expanded_ir
```

Run one specific test:

```powershell
python -m unittest tests.test_unified_ir.UnifiedIRTests.test_triton_add_emits_add_load_store
```

## Test Coverage Summary

Current test files:

- `tests/test_unified_ir.py`: 3 base tests
- `tests/test_codex_expanded_ir.py`: 2 expanded tests

Current suite:

- 5 total unit tests
- expanded cuTile test: 45 source lines, 44 emitted IR ops, 26 unique opcodes
- expanded Triton test: 30 source lines, 25 emitted IR ops, 21 unique opcodes
- full test suite runs in under 0.01 seconds on the current examples

## Supported Prototype Subset

Triton input is parsed as Python AST and recognizes:

- `@triton.jit` kernel functions
- `tl.program_id`, `tl.arange`, `tl.load`, `tl.store`
- arithmetic operators such as `+`, `-`, `*`, `/`, and `@`
- comparisons such as `<`, `<=`, `>`, `>=`, `==`, `!=`
- simple calls such as `tl.dot`, `tl.where`, `tl.exp`, `tl.sqrt`, `tl.maximum`, and `tl.minimum`

cuTile input is a small line-oriented syntax shaped like:

```cutile
kernel vector_add(x, y, z, n_elements, BLOCK_SIZE) {
  pid = program_id(0);
  offsets0 = arange(0, BLOCK_SIZE);
  x_tile = load(x + offsets, mask=mask, other=0.0);
  y_tile = load(y + offsets, mask=mask, other=0.0);
  z_tile = x_tile + y_tile;
  store(z + offsets, z_tile, mask=mask);
}
```

The parser also accepts typed assignments such as `auto z_tile = add(x_tile, y_tile);` and `tile z_tile = x_tile + y_tile;`.

## Line Format

Every output has a short header followed by numbered operator lines:

```text
# unified-tile-ir line-format v0
# source_lang=triton
# source=examples/triton_add.py
# max_ops=220
0000 | kernel       | name=vector_add params=x_ptr,y_ptr,z_ptr,n_elements,BLOCK_SIZE
0001 | program_id   | out=pid axis=0
```

Fields after the final pipe are key-value attributes. Values with spaces are JSON quoted so the output stays line-oriented and easy to inspect.
