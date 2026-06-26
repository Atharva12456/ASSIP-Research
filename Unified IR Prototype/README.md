# Unified IR Prototype

This is a small proof-of-concept unified tile IR emitter for Triton-like Python kernels and cuTile-like kernels. It outputs a simple line format where each operator is one line:

```text
0007 | add          | out=z lhs=x rhs=y
```

The prototype is intentionally narrow and fast. It has no third-party dependencies and defaults to a maximum of 220 emitted operator lines with `--max-ops 220`.

## Run It

From this folder:

```powershell
python -m unified_ir --lang triton examples/triton_add.py
python -m unified_ir --lang cutile examples/cutile_add.cutile
```

From the repo root:

```powershell
python ".\Unified IR Prototype\uir.py" --lang triton ".\Unified IR Prototype\examples\triton_add.py"
python ".\Unified IR Prototype\uir.py" --lang cutile ".\Unified IR Prototype\examples\cutile_add.cutile"
```

Write the IR to a file:

```powershell
python -m unified_ir --lang auto examples/triton_add.py --out add.lineir
```

Show supported prototype opcodes:

```powershell
python -m unified_ir --dump-ops
```

Run tests:

```powershell
python -m unittest discover -s tests
```

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
