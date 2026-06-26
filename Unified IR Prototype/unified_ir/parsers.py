from __future__ import annotations

import ast
import re
from pathlib import Path

from .ir import IRProgram
from .operator_catalog import BINOP_TO_IR, BOOL_TO_IR, CALL_TO_IR, COMPARE_TO_IR


class ParseError(Exception):
    """Raised when the prototype cannot parse the requested input."""


def parse_source(
    source: str,
    source_name: str,
    lang: str = "auto",
    *,
    max_ops: int = 220,
) -> IRProgram:
    resolved_lang = _detect_lang(source, source_name) if lang == "auto" else lang
    if resolved_lang == "triton":
        return TritonParser(max_ops=max_ops).parse(source, source_name)
    if resolved_lang == "cutile":
        return CuTileParser(max_ops=max_ops).parse(source, source_name)
    raise ParseError(f"unknown language: {resolved_lang}")


def _detect_lang(source: str, source_name: str) -> str:
    suffix = Path(source_name).suffix.lower()
    if suffix == ".py" or "triton.jit" in source or "triton.language" in source:
        return "triton"
    if suffix in {".cu", ".cuh", ".cutile", ".cpp", ".cc", ".h"}:
        return "cutile"
    if "cuTile" in source or "cutile" in source or "kernel " in source:
        return "cutile"
    raise ParseError("could not auto-detect language; pass --lang triton or --lang cutile")


class TritonParser:
    def __init__(self, *, max_ops: int = 220) -> None:
        self.max_ops = max_ops
        self.program: IRProgram | None = None

    def parse(self, source: str, source_name: str) -> IRProgram:
        try:
            tree = ast.parse(source, filename=source_name)
        except SyntaxError as exc:
            raise ParseError(f"Triton input is not valid Python: {exc}") from exc

        function = self._find_kernel(tree)
        if function is None:
            raise ParseError("no function found for Triton kernel")

        self.program = IRProgram("triton", source_name, max_ops=self.max_ops)
        params = [arg.arg for arg in function.args.args]
        self.program.add("kernel", name=function.name, params=params)
        for stmt in function.body:
            self._stmt(stmt)
        return self.program

    @staticmethod
    def _find_kernel(tree: ast.Module) -> ast.FunctionDef | None:
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        for function in functions:
            for decorator in function.decorator_list:
                if _expr_to_str(decorator) in {"triton.jit", "jit"}:
                    return function
        return functions[0] if functions else None

    def _stmt(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1:
                self._add("assign", out=",".join(_expr_to_str(t) for t in stmt.targets), value=_expr_to_str(stmt.value))
                return
            self._assign(_expr_to_str(stmt.targets[0]), stmt.value)
        elif isinstance(stmt, ast.AnnAssign):
            self._assign(_expr_to_str(stmt.target), stmt.value)
        elif isinstance(stmt, ast.AugAssign):
            opcode = BINOP_TO_IR.get(type(stmt.op), "assign")
            target = _expr_to_str(stmt.target)
            self._add(opcode, out=target, lhs=target, rhs=_expr_to_str(stmt.value), inplace=True)
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            self._call(stmt.value, out=None)
        elif isinstance(stmt, ast.Return):
            self._add("return", value=_expr_to_str(stmt.value) if stmt.value else None)
        elif isinstance(stmt, ast.If):
            self._add("if", cond=_expr_to_str(stmt.test))
            for child in stmt.body:
                self._stmt(child)
            if stmt.orelse:
                self._add("else")
                for child in stmt.orelse:
                    self._stmt(child)
            self._add("endif")
        elif isinstance(stmt, ast.For):
            self._add("for", target=_expr_to_str(stmt.target), iter=_expr_to_str(stmt.iter))
            for child in stmt.body:
                self._stmt(child)
            self._add("endfor")
        elif isinstance(stmt, ast.While):
            self._add("while", cond=_expr_to_str(stmt.test))
            for child in stmt.body:
                self._stmt(child)
            self._add("endwhile")
        elif isinstance(stmt, ast.Pass):
            return
        else:
            self._add("call", expr=_expr_to_str(stmt))

    def _assign(self, target: str, value: ast.AST | None) -> None:
        if value is None:
            self._add("assign", out=target)
        elif isinstance(value, ast.Call):
            self._call(value, out=target)
        elif isinstance(value, ast.BinOp):
            opcode = BINOP_TO_IR.get(type(value.op), "call")
            self._add(opcode, out=target, lhs=_expr_to_str(value.left), rhs=_expr_to_str(value.right))
        elif isinstance(value, ast.BoolOp):
            opcode = BOOL_TO_IR.get(type(value.op), "call")
            self._add(opcode, out=target, args=[_expr_to_str(item) for item in value.values])
        elif isinstance(value, ast.Compare):
            self._compare(target, value)
        else:
            self._add("assign", out=target, value=_expr_to_str(value))

    def _compare(self, target: str, value: ast.Compare) -> None:
        lhs = _expr_to_str(value.left)
        for index, op in enumerate(value.ops):
            rhs_node = value.comparators[index]
            rhs = _expr_to_str(rhs_node)
            opcode = COMPARE_TO_IR.get(type(op), "call")
            out = target if index == 0 else f"{target}_{index}"
            self._add(opcode, out=out, lhs=lhs, rhs=rhs)
            lhs = rhs

    def _call(self, call: ast.Call, *, out: str | None) -> None:
        callee = _expr_to_str(call.func)
        opcode = CALL_TO_IR.get(callee, "call")
        positional = [_expr_to_str(arg) for arg in call.args]
        kwargs = {kw.arg: _expr_to_str(kw.value) for kw in call.keywords if kw.arg}

        if opcode == "load":
            self._add("load", out=out, ptr=_get(positional, 0), **kwargs)
        elif opcode == "store":
            self._add("store", ptr=_get(positional, 0), value=_get(positional, 1), **kwargs)
        elif opcode == "arange":
            self._add("arange", out=out, start=_get(positional, 0), stop=_get(positional, 1), **kwargs)
        elif opcode == "program_id":
            axis = kwargs.pop("axis", None) or _get(positional, 0)
            self._add("program_id", out=out, axis=axis)
        elif opcode in {"add", "sub", "mul", "div", "dot", "max", "min"}:
            self._add(opcode, out=out, lhs=_get(positional, 0), rhs=_get(positional, 1), **kwargs)
        elif opcode == "select":
            self._add("select", out=out, cond=_get(positional, 0), true=_get(positional, 1), false=_get(positional, 2), **kwargs)
        elif opcode == "fill":
            self._add("fill", out=out, args=positional, **kwargs)
        elif opcode in {"exp", "sqrt"}:
            self._add(opcode, out=out, value=_get(positional, 0), **kwargs)
        else:
            self._add("call", out=out, callee=callee, args=positional, **kwargs)

    def _add(self, opcode: str, **attrs: object) -> None:
        assert self.program is not None
        self.program.add(opcode, **attrs)


class CuTileParser:
    def __init__(self, *, max_ops: int = 220) -> None:
        self.max_ops = max_ops
        self.program: IRProgram | None = None

    def parse(self, source: str, source_name: str) -> IRProgram:
        cleaned = _strip_c_comments(source)
        kernel_name, params = self._kernel_signature(cleaned)
        self.program = IRProgram("cutile", source_name, max_ops=self.max_ops)
        self.program.add("kernel", name=kernel_name, params=params)

        for statement in self._statements(cleaned):
            self._statement(statement)
        return self.program

    @staticmethod
    def _kernel_signature(source: str) -> tuple[str, list[str]]:
        patterns = [
            r"\bkernel\s+([A-Za-z_]\w*)\s*\(([^)]*)\)",
            r"\bvoid\s+([A-Za-z_]\w*)\s*\(([^)]*)\)",
            r"\b__global__\s+void\s+([A-Za-z_]\w*)\s*\(([^)]*)\)",
        ]
        for pattern in patterns:
            match = re.search(pattern, source)
            if match:
                params = [
                    _last_identifier(part.strip())
                    for part in match.group(2).split(",")
                    if part.strip()
                ]
                return match.group(1), params
        return "anonymous_kernel", []

    @staticmethod
    def _statements(source: str) -> list[str]:
        body_match = re.search(r"\{(?P<body>.*)\}", source, flags=re.DOTALL)
        body = body_match.group("body") if body_match else source
        rough_statements: list[str] = []
        for chunk in body.replace("{", ";\n").replace("}", ";\n").split(";"):
            statement = chunk.strip()
            if statement:
                rough_statements.append(statement)
        return rough_statements

    def _statement(self, statement: str) -> None:
        if statement in {"else"}:
            self._add("else")
            return
        if statement.startswith("if "):
            self._add("if", cond=_between_parens(statement) or statement[3:].strip())
            return
        if statement.startswith("for "):
            self._add("for", iter=_between_parens(statement) or statement[4:].strip())
            return
        if statement.startswith("while "):
            self._add("while", cond=_between_parens(statement) or statement[6:].strip())
            return
        if statement.startswith(("store(", "store_tile(", "cuTile.store(", "cutile.store(")):
            self._call_statement(statement, out=None)
            return

        decl = re.match(
            r"^(?:const\s+)?(?:auto|tile|Tile|Tensor|float|int|bool|half|bf16|fp32|fp16)\s+([A-Za-z_]\w*)\s*=\s*(.+)$",
            statement,
        )
        if decl:
            self._assign(decl.group(1), decl.group(2).strip())
            return

        assign = re.match(r"^([A-Za-z_]\w*)\s*=\s*(.+)$", statement)
        if assign:
            self._assign(assign.group(1), assign.group(2).strip())
            return

        self._call_statement(statement, out=None)

    def _assign(self, target: str, expr: str) -> None:
        call = _parse_call(expr)
        if call:
            self._emit_call(call[0], call[1], out=target)
            return

        binary = _split_binary(expr)
        if binary:
            lhs, symbol, rhs = binary
            self._add(_symbol_to_opcode(symbol), out=target, lhs=lhs, rhs=rhs)
            return

        self._add("assign", out=target, value=expr)

    def _call_statement(self, statement: str, *, out: str | None) -> None:
        call = _parse_call(statement)
        if call:
            self._emit_call(call[0], call[1], out=out)
        else:
            self._add("call", expr=statement)

    def _emit_call(self, callee: str, args_and_kwargs: list[str], *, out: str | None) -> None:
        opcode = CALL_TO_IR.get(callee, "call")
        positional, kwargs = _split_kwargs(args_and_kwargs)
        if opcode == "load":
            self._add("load", out=out, ptr=_get(positional, 0), **kwargs)
        elif opcode == "store":
            self._add("store", ptr=_get(positional, 0), value=_get(positional, 1), **kwargs)
        elif opcode == "arange":
            self._add("arange", out=out, start=_get(positional, 0), stop=_get(positional, 1), **kwargs)
        elif opcode == "program_id":
            axis = kwargs.pop("axis", None) or _get(positional, 0)
            self._add("program_id", out=out, axis=axis)
        elif opcode in {"add", "sub", "mul", "div", "dot", "max", "min"}:
            self._add(opcode, out=out, lhs=_get(positional, 0), rhs=_get(positional, 1), **kwargs)
        elif opcode == "select":
            self._add("select", out=out, cond=_get(positional, 0), true=_get(positional, 1), false=_get(positional, 2), **kwargs)
        elif opcode == "fill":
            self._add("fill", out=out, args=positional, **kwargs)
        elif opcode in {"exp", "sqrt"}:
            self._add(opcode, out=out, value=_get(positional, 0), **kwargs)
        else:
            self._add("call", out=out, callee=callee, args=positional, **kwargs)

    def _add(self, opcode: str, **attrs: object) -> None:
        assert self.program is not None
        self.program.add(opcode, **attrs)


def _expr_to_str(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return node.__class__.__name__


def _get(values: list[str], index: int) -> str | None:
    return values[index] if index < len(values) else None


def _strip_c_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    source = re.sub(r"//.*", "", source)
    return source


def _last_identifier(value: str) -> str:
    identifiers = re.findall(r"[A-Za-z_]\w*", value)
    return identifiers[-1] if identifiers else value


def _between_parens(value: str) -> str:
    match = re.search(r"\((.*)\)", value)
    return match.group(1).strip() if match else ""


def _parse_call(expr: str) -> tuple[str, list[str]] | None:
    expr = expr.strip()
    match = re.match(r"^([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)\s*\((.*)\)$", expr)
    if not match:
        return None
    return match.group(1), _split_args(match.group(2))


def _split_args(text: str) -> list[str]:
    args: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char == "," and depth == 0:
            arg = "".join(current).strip()
            if arg:
                args.append(arg)
            current = []
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        args.append(tail)
    return args


def _split_kwargs(items: list[str]) -> tuple[list[str], dict[str, str]]:
    positional: list[str] = []
    kwargs: dict[str, str] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if sep and re.match(r"^[A-Za-z_]\w*$", key.strip()):
            kwargs[key.strip()] = value.strip()
        else:
            positional.append(item)
    return positional, kwargs


def _split_binary(expr: str) -> tuple[str, str, str] | None:
    operators = ["<=", ">=", "==", "!=", "&&", "||", "+", "-", "*", "/", "<", ">"]
    depth = 0
    for index, char in enumerate(expr):
        if char in "([{":
            depth += 1
            continue
        if char in ")]}":
            depth -= 1
            continue
        if depth != 0:
            continue
        for symbol in operators:
            if expr.startswith(symbol, index):
                lhs = expr[:index].strip()
                rhs = expr[index + len(symbol) :].strip()
                if lhs and rhs:
                    return lhs, symbol, rhs
    return None


def _symbol_to_opcode(symbol: str) -> str:
    return {
        "+": "add",
        "-": "sub",
        "*": "mul",
        "/": "div",
        "<": "lt",
        "<=": "le",
        ">": "gt",
        ">=": "ge",
        "==": "eq",
        "!=": "ne",
        "&&": "and",
        "||": "or",
    }.get(symbol, "call")
