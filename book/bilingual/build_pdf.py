#!/usr/bin/env python3
"""Render the bilingual (EN + 中文) markdown chapters into a single PDF via XeLaTeX."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHAPTERS = ["03-gqa.md", "04-rmsnorm-and-mlp.md", "05-qwen3-model.md"]
OUT_TEX = HERE / "bilingual.tex"
OUT_PDF = HERE / "tiny-llm-week1-day3-5-bilingual.pdf"

PREAMBLE = r"""
\documentclass[11pt,a4paper]{article}
\usepackage{amsmath,amssymb}
\usepackage{fontspec}
\usepackage{xeCJK}
\usepackage{geometry}
\usepackage{xcolor}
\usepackage{url}
\usepackage{hyperref}
\usepackage{fvextra}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage{enumitem}

% Long URLs and identifiers must be able to break across lines.
\tolerance=2000
\emergencystretch=3em
\hbadness=10000

\defaultfontfeatures{Ligatures=TeX}
\setmainfont{Times New Roman}
\setsansfont{Helvetica Neue}
\setmonofont{Menlo}[Scale=0.88]
\setCJKmainfont{PingFang SC}
\setCJKsansfont{PingFang SC}
\setCJKmonofont{PingFang SC}
\xeCJKsetup{CJKmath=true}

\definecolor{zhcolor}{RGB}{26,26,26}
\definecolor{linkcolor}{RGB}{26,92,158}
\definecolor{codebg}{RGB}{244,245,247}
\definecolor{codeframe}{RGB}{216,220,226}
\hypersetup{colorlinks=true,linkcolor=linkcolor,urlcolor=linkcolor,
            citecolor=linkcolor,breaklinks=true}
\renewcommand{\UrlBreaks}{\do\/\do\-\do\.\do\_\do\?\do\&\do\+\do\=\do\#}

\geometry{top=2.2cm,bottom=2.2cm,left=2.3cm,right=2.3cm}
\sloppy

\titleformat{\section}{\Large\bfseries\sffamily}{}{0em}{}
\titleformat{\subsection}{\large\bfseries\sffamily}{}{0em}{}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\sffamily tiny-llm · Week 1 Day 3--5 双语对照}
\fancyhead[R]{\small\sffamily\thepage}
\renewcommand{\headrulewidth}{0.4pt}

\newenvironment{zhpar}{%
  \par\smallskip\begingroup\color{zhcolor}\small\ignorespaces%
}{\endgroup\par\smallskip}

\DefineVerbatimEnvironment{codeblock}{Verbatim}{
  frame=single,framesep=5pt,rulecolor=\color{codeframe},
  fillcolor=\color{codebg},fontsize=\small,formatcom=\color{black},
  baselinestretch=1.0,xleftmargin=2pt,breaklines=true,
  breakanywhere=true
}
"""


def escape(text: str) -> str:
    """Escape LaTeX specials, keeping already-escaped sequences intact."""
    out: list[str] = []
    for ch in text:
        if ch in "\\{}$&#^_%":
            out.append(
                {
                    "\\": r"\textbackslash{}",
                    "{": r"\{",
                    "}": r"\}",
                    "$": r"\$",
                    "&": r"\&",
                    "#": r"\#",
                    "^": r"\textasciicircum{}",
                    "_": r"\_",
                    "%": r"\%",
                }[ch]
            )
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "<":
            out.append(r"\textless{}")
        elif ch == ">":
            out.append(r"\textgreater{}")
        else:
            out.append(ch)
    return "".join(out)


INLINE_MATH = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")
LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
BOLD = re.compile(r"\*\*(.+?)\*\*")
CODE = re.compile(r"`([^`]+)`")
BULLET = re.compile(r"^[ \t]*[-*][ \t]+")
re.purge()


def inline(text: str) -> str:
    """Convert markdown inline markup to LaTeX, protecting verbatim spans.

    Order matters: links and bold recurse into their labels *before* inline
    code / math are stashed. Otherwise a recursive call receives text already
    containing this level's placeholders and returns them unexpanded, leaking
    raw NUL bytes into the .tex output.
    """
    spans: list[str] = []

    def stash(tex: str) -> str:
        spans.append(tex)
        return f"\x00{len(spans) - 1}\x00"

    def href_sub(m: re.Match[str]) -> str:
        return stash(r"\href{" + m.group(2) + "}{" + inline(m.group(1)) + "}")

    def bold_sub(m: re.Match[str]) -> str:
        return stash(r"\textbf{" + inline(m.group(1)) + "}")

    text = LINK.sub(href_sub, text)
    text = BOLD.sub(bold_sub, text)

    def code_sub(m: re.Match[str]) -> str:
        return stash(r"\texttt{" + escape(m.group(1)) + "}")

    def math_sub(m: re.Match[str]) -> str:
        return stash("$" + m.group(1) + "$")

    text = CODE.sub(code_sub, text)
    text = INLINE_MATH.sub(math_sub, text)

    # Emoji have no glyph in Times New Roman; drop them (their label follows).
    text = "".join(ch for ch in text if ord(ch) < 0x1F000)

    text = escape(text)

    for i, tex in enumerate(spans):
        text = text.replace(f"\x00{i}\x00", tex)
    return text


def convert(src: Path) -> str:
    """Convert one bilingual markdown chapter into LaTeX body."""
    lines = src.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # Fenced code block
        if line.startswith("```"):
            lang = line[3:].strip()
            i += 1
            body: list[str] = []
            while i < n and not lines[i].startswith("```"):
                body.append(lines[i])
                i += 1
            i += 1  # closing fence
            text = "\n".join(body)
            if lang in {"bash", "sh", "shell", "text", "plain", ""}:
                out.append("\\begin{codeblock}")
                out.append(text)
                out.append("\\end{codeblock}")
            else:
                out.append("\\begin{codeblock}")
                out.append(text)
                out.append("\\end{codeblock}")
            out.append("")
            continue

        # Display math
        if line.strip() == "$$":
            i += 1
            body = []
            while i < n and lines[i].strip() != "$$":
                body.append(lines[i])
                i += 1
            i += 1
            out.append("\\[")
            out.append("\n".join(body))
            out.append("\\]")
            out.append("")
            continue

        # Chinese translation paragraph (may contain its own bullet list)
        if line.startswith("zh>"):
            body = [line[3:].strip()]
            i += 1
            while i < n and lines[i].startswith("zh>"):
                body.append(lines[i][3:].strip())
                i += 1
            out.append("\\begin{zhpar}")
            body = [x for x in body if x]
            if any(BULLET.match(x) for x in body):
                # Preserve list structure: emit items, restoring any wrapped
                # continuation lines onto their bullet.
                items: list[str] = []
                for x in body:
                    if BULLET.match(x):
                        items.append(BULLET.sub("", x))
                    elif items:
                        items[-1] += " " + x
                    else:
                        items.append(x)
                out.append("\\begin{itemize}[leftmargin=1.6em,itemsep=0.15em,parsep=0pt]")
                for it in items:
                    out.append("\\item " + inline(it))
                out.append("\\end{itemize}")
            else:
                out.append(inline(" ".join(body)))
            out.append("\\end{zhpar}")
            continue


        # Headings
        if line.startswith("## "):
            out.append("\\subsection*{" + inline(line[3:].strip()) + "}")
            out.append("")
            i += 1
            continue
        if line.startswith("# "):
            out.append("\\section*{" + inline(line[2:].strip()) + "}")
            out.append("")
            i += 1
            continue
        # Bullet / list block
        if BULLET.match(line):
            items: list[str] = []
            while i < n and BULLET.match(lines[i]):
                item = re.sub(r"^[ \t]*[-*][ \t]+", "", lines[i])
                i += 1
                # Fold wrapped continuation lines into this item.
                while i < n and lines[i].strip() and not BULLET.match(lines[i]):
                    if lines[i].startswith(("```", "zh>", "# ")) or lines[i].strip() == "$$":
                        break
                    item += " " + lines[i].strip()
                    i += 1
                items.append(item)
            out.append("\\begin{itemize}[leftmargin=1.6em,itemsep=0.15em,parsep=0pt]")
            for it in items:
                out.append("\\item " + inline(it))
            out.append("\\end{itemize}")
            out.append("")
            continue

        # Paragraph: gather until blank line or a block start.
        # (An empty line reaching here is a no-op; skipping keeps the loop moving.)
        if not line.strip():
            i += 1
            continue
        para: list[str] = []
        while i < n:
            cur = lines[i]
            if (
                not cur.strip()
                or cur.startswith("```")
                or cur.startswith("zh>")
                or cur.startswith("# ")
                or cur.startswith("## ")
                or BULLET.match(cur)
            ):
                break
            para.append(cur)
            i += 1
        if para:
            out.append(inline(" ".join(para)))
            out.append("")
    return "\n".join(out)


def main() -> int:
    bodies: list[str] = []
    for name in CHAPTERS:
        path = HERE / name
        if not path.exists():
            print(f"missing chapter: {path}", file=sys.stderr)
            return 1
        bodies.append(convert(path))

    first = "Day 3--5"
    tex = (
        PREAMBLE
        + "\n\\begin{document}\n"
        + "\\begin{center}\n"
        + "{\\LARGE\\bfseries\\sffamily tiny-llm: Week 1 "
        + first
        + "}\\\\[0.4em]\n"
        + "{\\large\\sffamily Grouped Query Attention · RMSNorm 与 MLP · Qwen3 模型}\\\\[0.3em]\n"
        + "{\\small\\sffamily 中英双语对照 · Bilingual Edition}\n"
        + "\\end{center}\n\\vspace{0.6em}\n"
        + "\n\\clearpage\n".join(bodies)
        + "\n\\vspace{1em}\\begin{center}\\small\\sffamily "
        + "tiny-llm-book \\textcopyright{} 2025 by Alex Chi Z, CC BY-NC-SA 4.0. "
        + "英文原文 \\url{https://github.com/skyzh/tiny-llm}；中文译文为对照学习用途。\\end{center}\n"
        + "\\end{document}\n"
    )
    OUT_TEX.write_text(tex, encoding="utf-8")
    print(f"wrote {OUT_TEX}")

    for run in range(3):
        proc = subprocess.run(
            ["xelatex", "-interaction=nonstopmode", OUT_TEX.name],
            cwd=HERE,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if proc.returncode != 0:
            print(proc.stdout[-4000:], file=sys.stderr)
            print(proc.stderr[-2000:], file=sys.stderr)
            return proc.returncode

    produced = HERE / (OUT_TEX.stem + ".pdf")
    if produced.exists():
        produced.replace(OUT_PDF)
    print(f"wrote {OUT_PDF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
