from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .line_format import to_line_format
from .operator_catalog import SUPPORTED_OPS
from .parsers import ParseError, parse_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unified-ir",
        description="Emit a small line-format unified tile IR from Triton or cuTile-like kernels.",
    )
    parser.add_argument("source", nargs="?", help="Input source file. Reads stdin when omitted.")
    parser.add_argument(
        "--lang",
        choices=["auto", "triton", "cutile"],
        default="auto",
        help="Input language. Default: auto.",
    )
    parser.add_argument(
        "--out",
        help="Write line-format IR to this file instead of stdout.",
    )
    parser.add_argument(
        "--max-ops",
        type=int,
        default=220,
        help="Maximum emitted IR operator lines. Default: 220.",
    )
    parser.add_argument(
        "--dump-ops",
        action="store_true",
        help="Print supported prototype IR opcodes and exit.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"unified-ir prototype {__version__}",
    )
    return parser


def read_source(path_text: str | None) -> tuple[str, str]:
    if path_text is None:
        return sys.stdin.read(), "<stdin>"
    path = Path(path_text)
    return path.read_text(encoding="utf-8"), str(path)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.dump_ops:
        for op in sorted(SUPPORTED_OPS):
            print(op)
        return 0

    try:
        source, source_name = read_source(args.source)
        program = parse_source(source, source_name, args.lang, max_ops=args.max_ops)
        output = to_line_format(program)
    except (OSError, ParseError, ValueError) as exc:
        print(f"unified-ir: error: {exc}", file=sys.stderr)
        return 1

    if args.out:
        Path(args.out).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return 0
