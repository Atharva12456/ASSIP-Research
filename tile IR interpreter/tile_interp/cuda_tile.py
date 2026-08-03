"""Translate NVIDIA CUDA Tile IR text into the line-format tile IR.

The two IRs describe the same thing differently. CUDA Tile IR is SSA with typed
values, explicit pointer tiles, and loops that carry values; the line format is a
flat op list over named buffers. The interesting part of the translation is
pointers: rather than modelling them at runtime, this tracks provenance
statically, so a load through a pointer tile becomes a plain indexed load from
the buffer that pointer came from.

    %a_ptr      = broadcast %a_ptr_base_scalar     ->  (buffer A, offset 0)
    %a_tile_ptr = offset %a_ptr, %offs             ->  (buffer A, offset %offs)
    load_ptr_tko %a_tile_ptr                       ->  load ptr="A + %offs"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .ir import Op, Program

BINARY = {
    "addi": "add", "addf": "add",
    "subi": "sub", "subf": "sub",
    "muli": "mul", "mulf": "mul",
    "divi": "div", "divf": "div",
    "maxi": "max", "maxf": "max",
    "mini": "min", "minf": "min",
}

_COMMENT = re.compile(r"//.*")
_TYPE_SHAPE = re.compile(r"tile<([0-9x]*?)x?(?:ptr<)?[a-z0-9]+>?>")


class CuTileError(Exception):
    """Raised when the CUDA Tile IR text cannot be translated."""


@dataclass(slots=True)
class Pointer:
    """A pointer tile: which buffer it addresses, and the offset value name."""

    buffer: str
    offset: str | None = None

    def expr(self) -> str:
        return f"{self.buffer} + {self.offset}" if self.offset else self.buffer


@dataclass(slots=True)
class Statement:
    """One parsed CUDA Tile IR operation."""

    results: list[str]
    opcode: str
    operands: list[str]
    literal: str | None
    types: str
    body: list["Statement"] = field(default_factory=list)
    header: str = ""
    text: str = ""


def _strip_comments(text: str) -> str:
    return "\n".join(_COMMENT.sub("", line) for line in text.splitlines())


def _result_shape(types: str) -> tuple[int, ...] | None:
    """The shape of the last tile type in a type annotation, if it has one."""
    matches = re.findall(r"tile<([^>]*(?:<[^>]*>)?[^>]*)>", types)
    if not matches:
        return None
    body = matches[-1]
    dims = re.match(r"^((?:\d+x)*)", body).group(1)
    if not dims:
        return ()
    return tuple(int(part) for part in dims.rstrip("x").split("x"))


def _shape_text(shape: tuple[int, ...] | None) -> str | None:
    if not shape:
        return None
    return "x".join(str(dim) for dim in shape)


def _split_statements(text: str) -> list[Statement]:
    """Split a block into statements, recursing into brace-delimited loop bodies."""
    statements: list[Statement] = []
    buffer: list[str] = []
    depth = 0
    pending_header = ""
    body_chars: list[str] = []

    for char in text:
        if depth:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    inner = _split_statements("".join(body_chars))
                    statements.append(
                        Statement([], "for", [], None, "", inner, pending_header)
                    )
                    body_chars = []
                    buffer = []
                    continue
            body_chars.append(char)
            continue
        if char == "{":
            depth = 1
            # the header may span several physical lines already sitting in _PARTIAL
            pending_header = " ".join(_PARTIAL + ["".join(buffer)]).strip().replace("#", "__")
            _PARTIAL.clear()
            buffer = []
            continue
        if char == "\n":
            line = "".join(buffer).strip()
            if line:
                statements.extend(_flush(line))
            buffer = []
            continue
        buffer.append(char)

    tail = "".join(buffer).strip()
    if tail:
        statements.extend(_flush(tail))
    return statements


_PARTIAL = []


def _flush(line: str) -> list[Statement]:
    """Accumulate physical lines until one parses as a complete statement."""
    if not _PARTIAL and line.lstrip().startswith((":", "->")):
        # a type annotation trailing a statement that already parsed; nothing to do
        return []
    _PARTIAL.append(line)
    joined = " ".join(_PARTIAL).strip()
    if joined.endswith((",", "=", "->", ":")):
        return []
    parsed = _parse_statement(joined)
    if parsed is None:
        return []
    if parsed.opcode == "for":
        # a loop header: keep accumulating until its opening brace arrives
        return []
    _PARTIAL.clear()
    return [parsed]


def _parse_statement(line: str) -> Statement | None:
    if not line or line in {"}", "{"}:
        return None
    line = line.replace("#", "__")  # SSA result selector: %v#1 -> %v__1

    results: list[str] = []
    rest = line
    if "=" in line:
        head, _, tail = line.partition("=")
        head_s = head.strip()
        multi = re.fullmatch(r"%([\w.]+):(\d+)", head_s)
        if multi:  # a multi-result op, %name:N binds N results
            results = [f"{multi.group(1)}__{i}" for i in range(int(multi.group(2)))]
            rest = tail.strip()
        elif re.fullmatch(r"[\s%\w,.]+", head_s) and "%" in head_s:
            results = re.findall(r"%([\w.]+)", head_s)
            rest = tail.strip()

    rest = rest.strip()
    if not rest:
        return None

    # pull a scalar literal out first; its own colon would break the type split
    literal = None
    literal_match = re.search(r"<\s*[a-z]\w*\s*:\s*([-\d.e+]+)\s*>", rest)
    if literal_match:
        literal = literal_match.group(1)
        rest = (rest[: literal_match.start()] + " " + rest[literal_match.end():]).strip()

    full = rest  # keep the type annotation; some ops carry structure inside it
    head, _, types = rest.partition(":")
    head = head.strip()

    tokens = head.split()
    if not tokens:
        return None
    opcode = tokens[0]
    if opcode.startswith("cuda_tile."):
        opcode = opcode[len("cuda_tile.") :]
    operands = re.findall(r"%([\w.]+)", head)
    return Statement(results, opcode, operands, literal, types.strip(), text=full)


class CuTileTranslator:
    """Lowers parsed CUDA Tile IR statements into a line-format IRProgram."""

    def __init__(self, *, max_ops: int = 400) -> None:
        self.max_ops = max_ops
        self.program: Program | None = None
        self.pointers: dict[str, Pointer] = {}
        self.views: dict[str, str] = {}  # view name -> underlying buffer name
        self.params: list[str] = []
        self.notes: list[str] = []

    def translate(self, source: str, source_name: str) -> IRProgram:
        text = _strip_comments(source)
        _PARTIAL.clear()

        if not text.strip():
            raise CuTileError(
                "the file is empty; paste a CUDA Tile IR kernel into it "
                "(see examples/vector_add.tileir for the smallest working one)"
            )
        entry = re.search(r"entry\s+@([\w.]+)\s*\((.*?)\)\s*\{", text, re.DOTALL)
        if entry is None:
            raise CuTileError(
                "no 'entry @name(...) {' block found; a CUDA Tile IR kernel needs "
                "'cuda_tile.module @m { entry @name(%a: tile<ptr<f32>>, ...) { ... } }' "
                "(see examples/vector_add.tileir)"
            )

        name = entry.group(1)
        # Split the parameter list and tell pointers from scalars by their type.
        # A pointer parameter names a buffer; a scalar (a shape or a stride) is an
        # ordinary runtime value bound from the inputs.
        is_pointer: dict[str, bool] = {}
        self.params = []
        for arg in entry.group(2).split(","):
            if not arg.strip():
                continue
            param = arg.split(":")[0].strip().lstrip("%")
            self.params.append(param)
            is_pointer[param] = "ptr<" in arg

        body_start = entry.end()
        body_end = _matching_brace(text, body_start - 1)
        if body_end is None:
            raise CuTileError("unbalanced braces in the entry body")

        self.program = Program("cuda_tile", source_name, max_ops=self.max_ops)
        self._add("kernel", name=name, params=self.params)

        for param in self.params:
            if is_pointer[param]:
                self.pointers[param] = Pointer(_buffer_name(param))

        for statement in _split_statements(text[body_start:body_end]):
            self._statement(statement)
        return self.program

    # ------------------------------------------------------------------ ops

    def _statement(self, st: Statement) -> None:
        handler = getattr(self, f"_op_{st.opcode}", None)
        if handler is not None:
            handler(st)
            return
        if st.opcode in BINARY:
            self._binary(st)
            return
        self.notes.append(f"unhandled opcode {st.opcode!r}")

    def _binary(self, st: Statement) -> None:
        if len(st.operands) < 2 or not st.results:
            raise CuTileError(f"{st.opcode} needs two operands and a result")
        self._add(BINARY[st.opcode], out=st.results[0],
                  lhs=st.operands[0], rhs=st.operands[1])

    def _op_constant(self, st: Statement) -> None:
        shape = _result_shape(st.types)
        out = st.results[0]
        if shape:
            # the shape must reach fill as one bracketed list, not flattened ints
            dims = ",".join(str(dim) for dim in shape)
            self._add("fill", out=out, args=f"[{dims}]", value=st.literal)
        else:
            self._add("assign", out=out, value=st.literal)

    def _op_get_tile_block_id(self, st: Statement) -> None:
        for axis, result in enumerate(st.results):
            self._add("program_id", out=result, axis=axis)

    def _op_iota(self, st: Statement) -> None:
        shape = _result_shape(st.types) or (1,)
        self._add("arange", out=st.results[0], start=0, stop=shape[0])

    def _op_reshape(self, st: Statement) -> None:
        out, source = st.results[0], st.operands[0]
        if source in self.pointers:
            self.pointers[out] = self.pointers[source]
            return
        self._add("reshape", out=out, value=source,
                  shape=_shape_text(_result_shape(st.types)) or "1")

    def _op_broadcast(self, st: Statement) -> None:
        out, source = st.results[0], st.operands[0]
        if source in self.pointers:
            self.pointers[out] = self.pointers[source]
            return
        self._add("broadcast", out=out, value=source,
                  shape=_shape_text(_result_shape(st.types)) or "1")

    def _op_offset(self, st: Statement) -> None:
        out, base, delta = st.results[0], st.operands[0], st.operands[1]
        pointer = self.pointers.get(base)
        if pointer is None:
            raise CuTileError(f"offset on {base!r}, which is not a pointer")
        if pointer.offset is None:
            self.pointers[out] = Pointer(pointer.buffer, delta)
            return
        merged = f"{out}_off"
        self._add("add", out=merged, lhs=pointer.offset, rhs=delta)
        self.pointers[out] = Pointer(pointer.buffer, merged)

    def _op_load_ptr_tko(self, st: Statement) -> None:
        pointer = self.pointers.get(st.operands[0])
        if pointer is None:
            raise CuTileError(f"load through {st.operands[0]!r}, which is not a pointer")
        self._add("load", out=st.results[0], ptr=pointer.expr())

    def _op_store_ptr_tko(self, st: Statement) -> None:
        pointer = self.pointers.get(st.operands[0])
        if pointer is None:
            raise CuTileError(f"store through {st.operands[0]!r}, which is not a pointer")
        self._add("store", ptr=pointer.expr(), value=st.operands[1])

    def _op_mmaf(self, st: Statement) -> None:
        acc = st.operands[2] if len(st.operands) > 2 else None
        self._add("mma", out=st.results[0], lhs=st.operands[0],
                  rhs=st.operands[1], acc=acc)

    def _op_assume(self, st: Statement) -> None:
        """An alignment hint; pass the value through unchanged."""
        out, operand = st.results[0], st.operands[0]
        if operand in self.pointers:
            self.pointers[out] = self.pointers[operand]
        elif operand in self.views:
            self.views[out] = self.views[operand]
        else:
            self._add("assign", out=out, value=operand)

    def _op_make_tensor_view(self, st: Statement) -> None:
        operand = st.operands[0]
        buffer = self._buffer_of(operand)
        if buffer is None:
            raise CuTileError(f"make_tensor_view on {operand!r}, which is not a pointer")
        out = st.results[0]
        self._add(
            "tensor_view", out=out, buf=buffer,
            shape=_bracket_list(st.text, "shape"),
            strides=_bracket_list(st.text, "strides"),
        )
        self.views[out] = buffer

    def _op_make_partition_view(self, st: Statement) -> None:
        view = st.operands[0]
        out = st.results[0]
        self._add(
            "partition_view", out=out, view=view,
            tile=_paren_dims(st.text), dim_map=_bracket_list(st.text, "dim_map"),
        )
        self.views[out] = self.views.get(view)

    def _op_get_index_space_shape(self, st: Statement) -> None:
        # bracket the targets so the interpreter unpacks the result tuple
        self._add("index_space", out="(" + ",".join(st.results) + ")", view=st.operands[0])

    def _op_load_view_tko(self, st: Statement) -> None:
        view, index = _subscript(st.text)
        buffer = self.views.get(view)
        self._add("load_view", out=st.results[0], view=view, buf=buffer, index=index)

    def _op_store_view_tko(self, st: Statement) -> None:
        view, index = _subscript(st.text)
        blocked = {view, *index.split(",")}
        value = next((o for o in st.operands if o not in blocked), None)
        if value is None:
            raise CuTileError(f"store_view_tko found no value operand in {st.text!r}")
        buffer = self.views.get(view)
        self._add("store_view", view=view, buf=buffer, value=value, index=index)

    def _buffer_of(self, name: str) -> str | None:
        if name in self.views:
            return self.views[name]
        pointer = self.pointers.get(name)
        return pointer.buffer if pointer is not None else None

    def _op_for(self, st: Statement) -> None:
        """Lower an SSA loop with iter_values into the flat for/endfor form."""
        header = st.header
        results = re.findall(r"%([\w.]+)", header.split("for", 1)[0])
        induction = re.search(r"for\s+%([\w.]+)\s+in", header)
        bounds = re.search(r"\(\s*%([\w.]+)\s+to\s+%([\w.]+)\s*(?:,\s*step\s+%([\w.]+))?\s*\)",
                           header)
        carried = re.findall(r"%([\w.]+)\s*=\s*%([\w.]+)", header)

        if induction is None or bounds is None:
            raise CuTileError("could not parse the loop header")

        start, stop, step = bounds.group(1), bounds.group(2), bounds.group(3) or "1"

        # Seed each carried name. A carried POINTER needs its offset materialized
        # into a mutable variable: the load op's text is fixed at translation time,
        # so the offset it names has to be something the loop body can reassign.
        for name, initial in carried:
            pointer = self.pointers.get(initial)
            if pointer is None:
                self._add("assign", out=name, value=initial)
                continue
            slot = f"{name}_off"
            self._add("assign", out=slot, value=pointer.offset or "0")
            self.pointers[name] = Pointer(pointer.buffer, slot)

        self._add("for", target=induction.group(1),
                  iter=f"range({start}, {stop}, {step})")

        continues: list[str] = []
        for inner in st.body:
            if inner.opcode == "continue":
                continues = inner.operands
                continue
            self._statement(inner)

        # write the continue values back into the carried names
        for (name, _), value in zip(carried, continues):
            carried_pointer = self.pointers.get(name)
            next_pointer = self.pointers.get(value)
            if carried_pointer is not None and next_pointer is not None:
                if next_pointer.offset != carried_pointer.offset:
                    self._add("assign", out=carried_pointer.offset,
                              value=next_pointer.offset or "0")
            elif name != value:
                self._add("assign", out=name, value=value)

        self._add("endfor")

        # bind the loop results to the final carried values
        for result, (name, _) in zip(results, carried):
            if name in self.pointers:
                self.pointers[result] = self.pointers[name]
            elif result != name:
                self._add("assign", out=result, value=name)

    def _add(self, opcode: str, **attrs: object) -> None:
        assert self.program is not None
        if len(self.program.ops) >= self.program.max_ops:
            raise CuTileError(f"operator budget exceeded: {self.program.max_ops}")
        clean = {
            key: _stringify(value)
            for key, value in attrs.items()
            if value is not None and value != ""
        }
        self.program.ops.append(Op(len(self.program.ops), opcode, clean))


def _clean_names(text: str) -> str:
    """Comma-join the identifiers and integer literals in a bracketed list."""
    parts = [part.strip().lstrip("%") for part in text.split(",")]
    return ",".join(part for part in parts if part)


def _bracket_list(text: str, key: str) -> str:
    """The 'key = [a, b]' list from a statement, as 'a,b'."""
    match = re.search(rf"{key}\s*=\s*\[([^\]]*)\]", text)
    if match is None:
        raise CuTileError(f"expected '{key} = [...]' in {text!r}")
    return _clean_names(match.group(1))


def _paren_dims(text: str) -> str:
    """The 'tile=(128x64)' dimensions from a partition-view type, as '128,64'."""
    match = re.search(r"tile\s*=\s*\(([\dx]+)\)", text)
    if match is None:
        raise CuTileError(f"expected 'tile=(...)' in {text!r}")
    return ",".join(match.group(1).split("x"))


def _subscript(text: str) -> tuple[str, str]:
    """A '%view[%i, %j]' access, returning (view, 'i,j')."""
    match = re.search(r"%([\w.]+)\s*\[([^\]]*)\]", text)
    if match is None:
        raise CuTileError(f"expected a '%view[...]' access in {text!r}")
    return match.group(1), _clean_names(match.group(2))


def _stringify(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(_stringify(item) for item in value)
    return str(value)


def _buffer_name(param: str) -> str:
    """A_ptr_base_scalar -> A, so buffers read naturally in the trace."""
    head = param.split("_", 1)[0]
    return head.upper() if len(head) <= 2 else head


def _matching_brace(text: str, open_index: int) -> int | None:
    depth = 0
    for index in range(open_index, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def translate_cuda_tile(source: str, source_name: str = "<memory>",
                        *, max_ops: int = 400) -> IRProgram:
    """Translate CUDA Tile IR text into a line-format program."""
    return CuTileTranslator(max_ops=max_ops).translate(source, source_name)


def translate_cuda_tile_file(path: str | Path, *, max_ops: int = 400) -> IRProgram:
    """Translate a CUDA Tile IR file into a line-format program."""
    target = Path(path)
    return translate_cuda_tile(target.read_text(), str(target), max_ops=max_ops)
