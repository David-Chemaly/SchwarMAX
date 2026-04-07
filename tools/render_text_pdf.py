#!/usr/bin/env python3
"""
Render a plain-text or markdown file into a simple searchable PDF using only
the Python standard library.

The output is intentionally minimal: monospaced text with simple wrapping and
page breaks. This is useful when LaTeX / Pandoc are unavailable.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path


PAGE_WIDTH = 612
PAGE_HEIGHT = 792
LEFT = 54
TOP = 54
BOTTOM = 54
FONT_SIZE = 10
LEADING = 13
MAX_COLS = 92


def wrap_lines(text: str) -> list[str]:
    wrapped: list[str] = []
    in_code = False

    for raw in text.splitlines():
        line = raw.rstrip("\n")

        if line.startswith("```"):
            in_code = not in_code
            wrapped.append("")
            continue

        if not line.strip():
            wrapped.append("")
            continue

        if in_code:
            prefix = "    "
            code = line.expandtabs(4)
            if len(code) <= MAX_COLS - len(prefix):
                wrapped.append(prefix + code)
            else:
                for chunk in textwrap.wrap(
                    code,
                    width=MAX_COLS - len(prefix),
                    replace_whitespace=False,
                    drop_whitespace=False,
                    break_long_words=False,
                    break_on_hyphens=False,
                ):
                    wrapped.append(prefix + chunk)
            continue

        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            wrapped.append(heading.upper())
            wrapped.append("")
            continue

        indent = len(line) - len(line.lstrip(" "))
        prefix = " " * min(indent, 8)
        body = line.lstrip(" ")

        if body.startswith("- "):
            prefix += "- "
            body = body[2:]

        width = max(20, MAX_COLS - len(prefix))
        parts = textwrap.wrap(
            body,
            width=width,
            replace_whitespace=False,
            drop_whitespace=True,
            break_long_words=False,
            break_on_hyphens=False,
        )
        if not parts:
            wrapped.append(prefix.rstrip())
            continue

        wrapped.append(prefix + parts[0])
        cont_prefix = " " * len(prefix)
        for part in parts[1:]:
            wrapped.append(cont_prefix + part)

    return wrapped


def escape_pdf_text(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def paginate(lines: list[str]) -> list[list[str]]:
    usable_height = PAGE_HEIGHT - TOP - BOTTOM
    lines_per_page = usable_height // LEADING
    pages: list[list[str]] = []

    for i in range(0, len(lines), lines_per_page):
        pages.append(lines[i : i + lines_per_page])

    if not pages:
        pages = [[]]

    return pages


def build_page_stream(page_lines: list[str]) -> bytes:
    y0 = PAGE_HEIGHT - TOP
    out = ["BT", f"/F1 {FONT_SIZE} Tf", f"{LEFT} {y0} Td"]

    first = True
    for line in page_lines:
        safe = escape_pdf_text(line)
        if first:
            out.append(f"({safe}) Tj")
            first = False
        else:
            out.append(f"0 -{LEADING} Td")
            out.append(f"({safe}) Tj")

    if first:
        out.append("() Tj")

    out.append("ET")
    return "\n".join(out).encode("latin-1", errors="replace")


def render_pdf(lines: list[str], output_path: Path) -> None:
    pages = paginate(lines)

    objects: list[bytes] = []

    def add_object(data: bytes) -> int:
        objects.append(data)
        return len(objects)

    font_obj = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")

    page_obj_ids: list[int] = []
    content_obj_ids: list[int] = []

    for page_lines in pages:
        stream = build_page_stream(page_lines)
        content = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1")
            + stream
            + b"\nendstream"
        )
        content_obj_ids.append(add_object(content))
        page_obj_ids.append(0)

    pages_obj_index = len(objects) + 1

    for i, content_id in enumerate(content_obj_ids):
        page_dict = (
            f"<< /Type /Page /Parent {pages_obj_index} 0 R "
            f"/MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 {font_obj} 0 R >> >> "
            f"/Contents {content_id} 0 R >>"
        ).encode("latin-1")
        page_obj_ids[i] = add_object(page_dict)

    kids = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    pages_obj = add_object(
        f"<< /Type /Pages /Count {len(page_obj_ids)} /Kids [{kids}] >>".encode(
            "latin-1"
        )
    )
    catalog_obj = add_object(f"<< /Type /Catalog /Pages {pages_obj} 0 R >>".encode("latin-1"))

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]

    for i, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{i} 0 obj\n".encode("latin-1"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        pdf.extend(f"{off:010d} 00000 n \n".encode("latin-1"))

    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_obj} 0 R >>\n"
            f"startxref\n{xref_start}\n%%EOF\n"
        ).encode("latin-1")
    )

    output_path.write_bytes(pdf)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    text = args.input.read_text(encoding="utf-8")
    lines = wrap_lines(text)
    render_pdf(lines, args.output)


if __name__ == "__main__":
    main()
