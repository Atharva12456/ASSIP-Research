from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(slots=True)
class IROp:
    """One line in the prototype tile IR."""

    index: int
    opcode: str
    attrs: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class IRProgram:
    """A small kernel-level IR container with a strict operator budget."""

    source_lang: str
    source_name: str
    max_ops: int = 220
    ops: list[IROp] = field(default_factory=list)

    def add(self, opcode: str, **attrs: object) -> IROp:
        if len(self.ops) >= self.max_ops:
            raise ValueError(
                f"operator budget exceeded: {self.max_ops} max line-format ops"
            )
        clean_attrs = {
            key: _stringify(value)
            for key, value in attrs.items()
            if value is not None and value != ""
        }
        op = IROp(index=len(self.ops), opcode=opcode, attrs=clean_attrs)
        self.ops.append(op)
        return op

    def extend(self, opcode_attrs: Iterable[tuple[str, dict[str, object]]]) -> None:
        for opcode, attrs in opcode_attrs:
            self.add(opcode, **attrs)


def _stringify(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(_stringify(item) for item in value)
    return str(value)
