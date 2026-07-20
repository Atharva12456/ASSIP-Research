from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

_TRUE_WORDS = frozenset({"true", "1", "yes", "on"})


class IRError(Exception):
    """Raised when IR text is malformed or a required attribute is missing."""


def split_list(text: str | None) -> list[str]:
    """Split a comma-separated attribute value into stripped, non-empty parts."""
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


@dataclass(slots=True)
class Op:
    """One line-format operation: an index, an opcode, and ordered string attrs."""

    index: int
    opcode: str
    attrs: dict[str, str] = field(default_factory=dict)

    def get(self, key: str, default: str | None = None) -> str | None:
        """Return the attribute value, or default when the key is absent."""
        return self.attrs.get(key, default)

    def require(self, key: str) -> str:
        """Return the attribute value, raising IRError when the key is absent."""
        try:
            return self.attrs[key]
        except KeyError:
            raise IRError(
                f"op {self.index:04d} ({self.opcode}) is missing required attr {key!r}"
            ) from None

    def flag(self, key: str, default: bool = False) -> bool:
        """Interpret an attribute as a boolean flag, e.g. inplace=true."""
        value = self.attrs.get(key)
        if value is None:
            return default
        return value.strip().lower() in _TRUE_WORDS

    def list_attr(self, key: str) -> list[str]:
        """Split a comma-separated attribute such as params or args."""
        return split_list(self.attrs.get(key))


@dataclass(slots=True)
class Program:
    """An ordered list of ops plus the provenance headers of the line format."""

    source_lang: str
    source_name: str
    ops: list[Op] = field(default_factory=list)
    max_ops: int = 220

    def __iter__(self) -> Iterator[Op]:
        return iter(self.ops)

    def __len__(self) -> int:
        return len(self.ops)

    def op(self, index: int) -> Op:
        """Return the op whose index attribute equals index."""
        if 0 <= index < len(self.ops) and self.ops[index].index == index:
            return self.ops[index]
        for candidate in self.ops:
            if candidate.index == index:
                return candidate
        raise IRError(f"no op with index {index}")

    def kernel_name(self) -> str:
        """Name from the first kernel op, or 'anonymous'."""
        for op in self.ops:
            if op.opcode == "kernel":
                return op.get("name") or "anonymous"
        return "anonymous"

    def params(self) -> list[str]:
        """Parameter names from the first kernel op."""
        for op in self.ops:
            if op.opcode == "kernel":
                return split_list(op.get("params"))
        return []

    def find(self, opcode: str) -> list[Op]:
        """All ops with the given opcode, in program order."""
        return [op for op in self.ops if op.opcode == opcode]
