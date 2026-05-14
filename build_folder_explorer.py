#!/usr/bin/env python3
"""
Build a single self-contained HTML explorer (same UI patterns as adaptive-rejection-sampler-explorer.html)
from a file or a directory tree.

Examples:
  python3 build_folder_explorer.py ./adaptive-rejection-sampler -o ./my-explorer.html
  python3 build_folder_explorer.py ./README.md -o readme-explorer.html
  python3 build_folder_explorer.py /path/to/task \\
      --template ./adaptive-rejection-sampler/adaptive-rejection-sampler-explorer.html \\
      --title "My task" --subtitle "Offline bundle"
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
from pathlib import Path


DEFAULT_EXCLUDE_DIR_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
        ".venv",
        "venv",
        ".tox",
        ".idea",
        ".vscode",
    }
)

DEFAULT_EXCLUDE_GLOBS = (
    "*.pyc",
    ".DS_Store",
)


def should_skip_file(rel: str, exclude_globs: tuple[str, ...]) -> bool:
    base = rel.rsplit("/", 1)[-1]
    for g in exclude_globs:
        if fnmatch.fnmatch(base, g) or fnmatch.fnmatch(rel, g):
            return True
    return False


def collect_files(
    root: Path,
    *,
    max_bytes: int,
    exclude_dirs: frozenset[str],
    exclude_globs: tuple[str, ...],
) -> dict[str, str]:
    """Return map posix_relpath -> utf-8 text (or placeholder for binary/skip)."""
    files: dict[str, str] = {}
    root = root.resolve()

    if root.is_file():
        rel = root.name
        files[rel] = read_one_file(root, max_bytes=max_bytes)
        return files

    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        try:
            rel = p.relative_to(root).as_posix()
        except ValueError:
            continue
        parts = rel.split("/")
        if any(part in exclude_dirs for part in parts[:-1]):
            continue
        if should_skip_file(rel, exclude_globs):
            continue
        files[rel] = read_one_file(p, max_bytes=max_bytes)
    return files


def read_one_file(path: Path, *, max_bytes: int) -> str:
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        return f"[Omitted: file is {len(raw)} bytes; limit is {max_bytes} bytes. Increase --max-bytes.]\n"
    if b"\x00" in raw[:8192]:
        return "[Binary file: null bytes in header; preview omitted.]\n"
    low = path.suffix.lower()
    if low in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".gz", ".tgz", ".enc"}:
        return f"[Binary / archive ({low}): not embedded as text.]\n"
    return raw.decode("utf-8", errors="replace")


def json_for_script_tag(obj: object) -> str:
    """Serialize JSON safe to embed inside <script type=application/json> (no raw </script>)."""
    s = json.dumps(obj, ensure_ascii=False)
    return s.replace("<", "\\u003c")


def patch_template(
    html: str,
    *,
    title: str,
    subtitle: str,
    pill: str,
    file_map: dict[str, str],
) -> str:
    embed = json_for_script_tag(file_map)
    start_tag = '<script type="application/json" id="file-embed">'
    i = html.find(start_tag)
    if i == -1:
        raise SystemExit("Template missing file-embed script tag.")
    j = html.find("</script>", i)
    if j == -1:
        raise SystemExit("Template malformed: no closing </script> for file-embed.")
    j += len("</script>")
    html = html[:i] + start_tag + embed + "</script>" + html[j:]

    html = re.sub(
        r"<title>.*?</title>",
        f"<title>{_escape_xml(title)}</title>",
        html,
        count=1,
        flags=re.DOTALL,
    )

    html = re.sub(
        r'(<div class="brand">\s*<h1>)(.*?)(</h1>\s*<p>)(.*?)(</p>\s*<span class="pill">)(.*?)(</span>\s*</div>)',
        rf"\g<1>{_escape_xml(title)}\g<3>{_escape_xml(subtitle)}\g<5>{_escape_xml(pill)}\g<7>",
        html,
        count=1,
        flags=re.DOTALL,
    )

    old_nav = r"""  const nav = document.getElementById("nav");
  const lbl = document.createElement("div");
  lbl.className = "section-label";
  lbl.textContent = "Agent run";
  nav.appendChild(lbl);
  nav.appendChild(makeFolder("agent", ["agent/trajectory.json", "agent/test-stdout.txt"]));

  const lbl2 = document.createElement("div");
  lbl2.className = "section-label";
  lbl2.textContent = "Task tree";
  nav.appendChild(lbl2);
  nav.appendChild(makeFileLink("instruction.md", "instruction.md"));
  nav.appendChild(makeFolder("environment", Object.keys(FILES).filter((k) => k.startsWith("environment/"))));
  nav.appendChild(makeFolder("solution", Object.keys(FILES).filter((k) => k.startsWith("solution/"))));
  nav.appendChild(makeFolder("tests", Object.keys(FILES).filter((k) => k.startsWith("tests/"))));"""

    new_nav = r"""  const nav = document.getElementById("nav");
  const lbl = document.createElement("div");
  lbl.className = "section-label";
  lbl.textContent = "Embedded files (" + Object.keys(FILES).length + ")";
  nav.appendChild(lbl);
  Object.keys(FILES)
    .sort(function (a, b) {
      return a.localeCompare(b);
    })
    .forEach(function (p) {
      nav.appendChild(makeFileLink(p, p));
    });"""

    if old_nav not in html:
        raise SystemExit(
            "Template no longer matches expected nav block. "
            "Pass --template pointing to adaptive-rejection-sampler-explorer.html "
            "from this repo, or update old_nav in build_folder_explorer.py."
        )
    html = html.replace(old_nav, new_nav, 1)
    return html


def _escape_xml(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def main() -> None:
    here = Path(__file__).resolve().parent
    default_template = here / "adaptive-rejection-sampler" / "adaptive-rejection-sampler-explorer.html"

    ap = argparse.ArgumentParser(
        description="Embed a folder (or single file) into one offline HTML explorer."
    )
    ap.add_argument(
        "input",
        type=Path,
        help="File or directory to embed (paths inside HTML are relative to this root).",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .html path (default: <input>/folder-explorer.html or <input>.explorer.html for a file).",
    )
    ap.add_argument(
        "--template",
        type=Path,
        default=default_template,
        help=f"HTML to use as shell (default: {default_template})",
    )
    ap.add_argument("--title", default="", help="Page / brand title (default: input name)")
    ap.add_argument(
        "--subtitle",
        default="",
        help="Brand subtitle (default: one-line hint about embedded bundle).",
    )
    ap.add_argument("--pill", default="embedded", help="Small pill label in sidebar brand.")
    ap.add_argument(
        "--max-bytes",
        type=int,
        default=5_000_000,
        help="Skip very large files beyond this size (default 5_000_000).",
    )
    ap.add_argument(
        "--exclude-glob",
        action="append",
        default=[],
        help="Extra glob(s) matched against basename or full relative path (repeatable).",
    )
    args = ap.parse_args()

    inp = args.input.expanduser().resolve()
    if not inp.exists():
        print(f"Not found: {inp}", file=sys.stderr)
        sys.exit(1)

    tpl = args.template.expanduser().resolve()
    if not tpl.is_file():
        print(f"Template not found: {tpl}", file=sys.stderr)
        sys.exit(1)

    title = args.title or inp.name
    subtitle = args.subtitle or (
        f"Single-file preview of {inp.name}. Open locally; no server required."
        if inp.is_dir()
        else f"Single-file preview. Open locally; no server required."
    )
    pill = args.pill

    exclude_globs = tuple(DEFAULT_EXCLUDE_GLOBS + tuple(args.exclude_glob))
    file_map = collect_files(
        inp,
        max_bytes=args.max_bytes,
        exclude_dirs=DEFAULT_EXCLUDE_DIR_NAMES,
        exclude_globs=exclude_globs,
    )
    if not file_map:
        print("No files collected.", file=sys.stderr)
        sys.exit(1)

    html = tpl.read_text(encoding="utf-8")
    out = patch_template(html, title=title, subtitle=subtitle, pill=pill, file_map=file_map)

    out_path = args.output
    if out_path is None:
        if inp.is_dir():
            out_path = inp / "folder-explorer.html"
        else:
            out_path = inp.parent / f"{inp.name}.explorer.html"

    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out, encoding="utf-8")
    print(f"Wrote {out_path} ({len(file_map)} file(s), {out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
