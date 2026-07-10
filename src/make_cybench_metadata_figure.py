"""
make_metadata_figure.py
=======================
Generate a ``tcolorbox`` LaTeX figure comparing a multi-file tabular dataset
against agent-predicted Croissant-style metadata.

Abstract
--------
Reads (1) a directory of CSV files constituting one dataset and (2) a JSON
file of predicted metadata, and emits a self-contained ``figure*`` block.
The figure has a full-width context box summarising the input files and two
side-by-side boxes holding the descriptive and structural halves of the
prediction.

Nothing in the figure is hardcoded: shared join keys, per-file column lists,
field-type counts, and the example row are all derived from the inputs.

Keywords
--------
LaTeX, tcolorbox, Croissant, metadata, figure generation

License
-------
CC-BY 4.0 — https://creativecommons.org/licenses/by/4.0/

Usage
-----
    python make_metadata_figure.py \
        --csv-dir data/maize_IE \
        --json cybench_IE_multi_csv.json \
        --out fig_metadata_multifile.tex \
        --cite cybench2024
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()
REPO = os.getenv("REPO_PATH")
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Croissant dataType -> single-letter marker used in the context table.
TYPE_LETTER: dict[str, str] = {
    "Text": "t",
    "Number": "n",
    "Date": "d",
    "Integer": "i",
    "Boolean": "b",
}

# Fallback when a column is present in the CSVs but absent from the JSON.
PANDAS_TO_TYPE: dict[str, str] = {
    "object": "Text",
    "int64": "Integer",
    "float64": "Number",
    "bool": "Boolean",
    "datetime64[ns]": "Date",
}

# Characters that must be escaped in LaTeX text/`\texttt` mode.
_LATEX_ESCAPES: dict[str, str] = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

NUMERIC_TYPES = ("Number", "Integer")
 
 

def tex_escape(s: object) -> str:
    """Escape a string for safe inclusion in LaTeX text or \\texttt{}."""
    out = str(s)
    # Backslash first, so the replacements it introduces are not re-escaped.
    out = out.replace("\\", _LATEX_ESCAPES["\\"])
    for ch, repl in _LATEX_ESCAPES.items():
        if ch == "\\":
            continue
        out = out.replace(ch, repl)
    return out


def _tt(s: object) -> str:
    """Render as \\texttt{...} with escaping."""
    return r"\texttt{" + tex_escape(s) + "}"


# ---------------------------------------------------------------------------
# Input inspection
# ---------------------------------------------------------------------------

def read_csv_dir(csv_dir: Path, nrows: int = 5) -> dict[str, pd.DataFrame]:
    """Read every .csv in a directory (sorted by name); keep only `nrows`."""
    paths = sorted(csv_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No .csv files found in {csv_dir}")
    return {p.name: pd.read_csv(p, nrows=nrows) for p in paths}


def shared_columns(frames: dict[str, pd.DataFrame]) -> list[str]:
    """Columns present in EVERY file — the implicit join keys."""
    if not frames:
        return []
    common = set.intersection(*(set(df.columns) for df in frames.values()))
    # Preserve the column order of the first frame for stable output.
    first = next(iter(frames.values()))
    return [c for c in first.columns if c in common]


def field_types(meta: dict) -> dict[tuple[str, str], str]:
    """Map (file, column) -> Croissant dataType, from the predicted JSON."""
    types: dict[tuple[str, str], str] = {}
    for rs in meta.get("recordsets", meta.get("recordSet", [])):
        src = rs.get("source", "")
        for f in rs.get("fields", []):
            types[(src, f.get("source", ""))] = f.get("dataType", "")
    return types


def type_for(
    file: str,
    col: str,
    types: dict[tuple[str, str], str],
    df: pd.DataFrame,
) -> str:
    """Predicted dataType, falling back to the pandas dtype if unpredicted."""
    t = types.get((file, col))
    if t:
        return t
    return PANDAS_TO_TYPE.get(str(df[col].dtype), "Text")


def _fmt_number(value: object) -> str:
    """Render a number without scientific notation.
 
    Plain ``%g`` flips to sci notation once the exponent reaches the
    precision, so e.g. 1361240.75 would print as 1.36124e+06. A high
    precision avoids that for realistic magnitudes; the ``e`` check catches
    anything extreme and falls back to fixed-point."""
    if not isinstance(value, float):
        return str(value)
    text = f"{value:.10g}"
    if "e" in text or "E" in text:
        text = f"{value:f}".rstrip("0").rstrip(".")
    return text
 
 
def _fmt_cell(value: object, dtype: str) -> str:
    """Format one table cell. Negative numbers get a real minus sign, since a
    plain hyphen renders too short in text mode."""
    if pd.isna(value):
        return r"\textit{NA}"
    if dtype in NUMERIC_TYPES:
        text = _fmt_number(value)
        if text.startswith("-"):
            return "$-$" + tex_escape(text[1:])
        return tex_escape(text)
    return tex_escape(value)
 
 

def _data_table(
    name: str,
    df: pd.DataFrame,
    types: dict[tuple[str, str], str],
    max_rows: int = 5,
) -> str:
    """Render the first `max_rows` rows of one file as a booktabs tabular.
 
    Numeric columns are right-aligned. A trailing ellipsis row is emitted only
    when the file actually continues past `max_rows` (read_csv_dir loads one
    spare row so this is knowable)."""
    cols = list(df.columns)
    dtypes = [type_for(name, c, types, df) for c in cols]
 
    align = "".join("r" if d in NUMERIC_TYPES else "l" for d in dtypes)
    header = " & ".join(_tt(c) for c in cols)
 
    shown = df.head(max_rows)
    body = [
        " & ".join(_fmt_cell(v, d) for v, d in zip(row, dtypes)) + r" \\"
        for row in shown.itertuples(index=False, name=None)
    ]
    if len(df) > max_rows:
        body.append(" & ".join([r"$\cdots$"] * len(cols)) + r" \\")
 
    # Wide tables need a smaller face and tighter columns to stay in the box.
    size = r"\tiny" if len(cols) > 8 else r"\scriptsize"
    sep = "3pt" if len(cols) > 8 else "5pt"
 
    return "\n".join([
        r"{" + size,
        rf"\setlength{{\tabcolsep}}{{{sep}}}",
        rf"\begin{{tabular}}{{@{{}}{align}@{{}}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}\par}",
    ])
 
# ---------------------------------------------------------------------------
# Figure sections
# ---------------------------------------------------------------------------
def _context_box(
    frames: dict[str, pd.DataFrame],
    types: dict[tuple[str, str], str],
    keys: list[str],
    total_fields: int,
    excerpt_name: str,
    max_rows: int = 5,
) -> str:
    """Full-width box: the CSV files, their columns, and the first rows of the
    excerpt file."""
    key_str = ", ".join(_tt(k) for k in keys)

    used = {type_for(f, c, types, df)
            for f, df in frames.items() for c in df.columns}
    legend = ", ".join(
        rf"(\textsc{{{TYPE_LETTER[t]}}})\,=\,{t}"
        for t in TYPE_LETTER if t in used
    )

    ordered = sorted(frames.items(), key=lambda kv: kv[0])

    rows = []
    for fname, df in ordered:
        extra = [c for c in df.columns if c not in keys]
        cells = ", ".join(
            _tt(c) + rf" (\textsc{{{TYPE_LETTER[type_for(fname, c, types, df)]}}})"
            for c in extra
        ) or r"\textit{(keys only)}"
        rows.append(f"{_tt(fname)} & {len(df.columns)} & {cells} \\\\")
    table_body = "\n".join(rows)

    if excerpt_name not in frames:
        raise KeyError(f"{excerpt_name!r} not among the CSV files in the directory {list(frames.keys())}")
    ex_df = frames[excerpt_name]
    data_table = _data_table(excerpt_name, ex_df, types, max_rows=max_rows)
    shown = min(max_rows, len(ex_df))

    n_files = len(frames)
    return rf"""\begin{{tcolorbox}}[
    title={{Example multi-file dataset ({n_files} CSV files, {total_fields} columns in total)}},
    fonttitle=\bfseries\small,
    colback=gray!8, colframe=gray!50,
    boxrule=0.4pt, arc=2pt,
    left=4pt, right=4pt, top=3pt, bottom=3pt,
]
\scriptsize
Every file shares the join {'key' if len(keys) == 1 else 'keys'} {key_str};
remaining columns are listed below. Types: {legend}.

\smallskip
\begin{{tabular}}{{@{{}}llp{{0.52\linewidth}}@{{}}}}
\toprule
File & \# columns & Columns beyond the shared keys \\\\
\midrule
{table_body}
\bottomrule
\end{{tabular}}

\smallskip
\textbf{{First {shown} rows}} of {_tt(excerpt_name)}:

\vspace{{2pt}}
{data_table}
\end{{tcolorbox}}"""


def _fmt_spatial(sc: object) -> str:
    """Render spatialCoverage, tolerating bbox dicts or plain strings."""
    if isinstance(sc, dict):
        need = ("min_lat", "max_lat", "min_lon", "max_lon")
        if all(k in sc for k in need):
            return (rf"bounding box lat $[{sc['min_lat']:.3f},\ {sc['max_lat']:.3f}]$, "
                    rf"lon $[{sc['min_lon']:.3f},\ {sc['max_lon']:.3f}]$")
        return tex_escape(json.dumps(sc))
    return tex_escape(sc) if sc else r"\textit{(none)}"


def _fmt_temporal(tc: object) -> str:
    if isinstance(tc, dict) and "from" in tc and "to" in tc:
        return rf"{tex_escape(tc['from'])} \,--\, {tex_escape(tc['to'])}"
    return tex_escape(tc) if tc else r"\textit{(none)}"


def _descriptive_box(meta: dict, frames: dict[str, pd.DataFrame]) -> str:
    """Left box: name, description, keywords, language, coverage, filesets."""
    kw = ", ".join(tex_escape(k) for k in meta.get("keywords", []))
    lang = ", ".join(_tt(l) for l in meta.get("inLanguage", [])) or r"\textit{(none)}"

    fs = meta.get("filesets", meta.get("fileSets", {}))
    includes = fs.get("includes", []) if isinstance(fs, dict) else list(fs)
    excludes = fs.get("excludes", []) if isinstance(fs, dict) else []
    inc_str = '[' + ', '.join(_tt(f) for f in includes) + ']' if includes else r"\textit{(none)}"
    # inc_str = (rf"all {len(includes)} files" if len(includes) == len(frames)
    #            else ", ".join(_tt(f) for f in includes))
    exc_str = '[' + ', '.join(_tt(f) for f in excludes) + ']' if excludes else r"\textit{(none)}"

    lines = [
        rf"\textbf{{Name:}} {tex_escape(meta.get('name', ''))}",
        rf"\textbf{{Description:}} {tex_escape(meta.get('description', ''))}",
        rf"\textbf{{Keywords:}} {kw}",
        rf"\textbf{{inLanguage:}} {lang}",
        (rf"\textbf{{spatialCoverage:}} {_fmt_spatial(meta.get('spatialCoverage'))} \\"
         + "\n" + rf"\textbf{{spatial:}} {tex_escape(meta.get('spatial', ''))}"),
        (rf"\textbf{{temporalCoverage:}} {_fmt_temporal(meta.get('temporalCoverage'))} \\"
         + "\n" + rf"\textbf{{temporal:}} {tex_escape(meta.get('temporal', ''))}"),
        rf"\textbf{{filesets.includes:}} {inc_str} \quad \textbf{{excludes:}} {exc_str}",
    ]
    body = "\n\n\\smallskip\n".join(lines)

    return rf"""\begin{{tcolorbox}}[
    title={{Agentic metadata --- descriptive}},
    fonttitle=\bfseries\small,
    colback=blue!5, colframe=blue!40!black,
    boxrule=0.4pt, arc=2pt,
    left=4pt, right=4pt, top=3pt, bottom=3pt,
    width=0.49\linewidth,
    nobeforeafter,
    equal height group=meta,
]
\scriptsize
{body}
\end{{tcolorbox}}"""


def _ttlines(lines: list[str]) -> str:
    """A monospaced block. Avoids `verbatim`, which breaks inside a
    tcolorbox that belongs to an `equal height group` (typeset twice).
    Spaces become `~`, so callers MUST pre-wrap: a long line has no break
    points left and will overflow the box."""
    esc = [tex_escape(l).replace(" ", "~") for l in lines]
    return ("{\\ttfamily\\scriptsize\\setlength{\\parindent}{0pt}\n"
            + " \\\\\n".join(esc) + "\n\\par}")


def _wrap_example(text: str, width: int = 44, indent: str = "   ") -> list[str]:
    """Split an example row at ``, `` boundaries into lines <= `width` chars."""
    parts = text.split(", ")
    lines: list[str] = []
    cur = ""
    for i, part in enumerate(parts):
        piece = part if i == len(parts) - 1 else part + ","
        candidate = f"{cur} {piece}" if cur else piece
        if cur and len(candidate) > width:
            lines.append(cur)
            cur = indent + piece
        else:
            cur = candidate
    if cur:
        lines.append(cur)
    return lines


def _structural_box(meta: dict, excerpt: str | None = None) -> str:
    """Right box: recordsets summary plus one excerpt rendered verbatim-ish."""
    recordsets = meta.get("recordsets", meta.get("recordSet", []))
    n_rs = len(recordsets)
    n_fields = sum(len(rs.get("fields", [])) for rs in recordsets)

    counts = Counter(f.get("dataType", "?")
                     for rs in recordsets for f in rs.get("fields", []))
    counts_str = ", ".join(f"{t} {c}" for t, c in counts.most_common())

    keys = {rs.get("key") for rs in recordsets}
    key_note = (rf"Every set takes {_tt(next(iter(keys)))} as its {_tt('key')}"
                if len(keys) == 1 else
                rf"Declared {_tt('key')}s: " + ", ".join(_tt(k) for k in sorted(map(str, keys))))

    all_fields = [f for rs in recordsets for f in rs.get("fields", [])]
    no_array = all(not f.get("isArray") for f in all_fields)
    no_refs = all(f.get("references") is None for f in all_fields)
    flags = []
    if no_array:
        flags.append(rf"{_tt('isArray: false')}")
    if no_refs:
        flags.append(rf"{_tt('references: null')}")
    flag_note = (rf"All {n_fields} fields have " + " and ".join(flags) + "."
                 if flags else "")

    # Choose the excerpt: named file, else the median-width recordset.
    if excerpt is not None:
        rs = next((r for r in recordsets if r.get("source") == excerpt), None)
        if rs is None:
            raise ValueError(f"No recordset with source={excerpt!r}")
    else:
        rs = sorted(recordsets, key=lambda r: len(r.get("fields", [])))[len(recordsets) // 2]
    idx = recordsets.index(rs)

    width = max((len(f["source"]) for f in rs["fields"]), default=0)
    lines = [f"source: {rs['source']}", f"key:    {rs.get('key', '')}", "fields:"]
    lines += [f"  {f['source']:<{width}}  {f.get('dataType','')}" for f in rs["fields"]]
    if rs.get("examples"):
        wrapped = _wrap_example(f'"{rs["examples"][0]}"')
        lines += ["examples:"] + ["  " + w for w in wrapped]

    return rf"""\begin{{tcolorbox}}[
    title={{Agentic metadata --- structural (\texttt{{recordsets}})}},
    fonttitle=\bfseries\small,
    colback=blue!5, colframe=blue!40!black,
    boxrule=0.4pt, arc=2pt,
    left=4pt, right=4pt, top=3pt, bottom=3pt,
    width=0.49\linewidth,
    nobeforeafter,
    equal height group=meta,
]
\scriptsize
{n_rs} {_tt('recordsets')}, one per file, together declaring {n_fields} fields.
{key_note}. {flag_note}

\smallskip
\textbf{{Excerpt}} --- {_tt(f'recordsets[{idx}]')}:

\vspace{{2pt}}
{_ttlines(lines)}
\vspace{{2pt}}

\textbf{{Type assignment across all {n_fields} fields:}} {counts_str}.
\end{{tcolorbox}}"""


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def make_metadata_figure(
    csv_dir: str | None = None,
    json_path: str | None = None,
    caption: str | None = None,
    cite_key: str | None = None,
    label: str = "fig:example_metadata_multifile",
    excerpt: str | None = None,
) -> str:
    """
    Build the complete ``figure*`` block as a LaTeX string.

    Parameters
    ----------
    csv_dir : path to the directory holding the dataset's .csv files
    json_path : path to the predicted-metadata JSON
    caption : caption text; a neutral default is used if omitted
    cite_key : appended as ``Data from \\cite{key}.`` when given
    label : the ``\\label{...}`` for cross-referencing
    excerpt : filename whose recordset is shown in full; median-width if None

    Returns
    -------
    str : LaTeX source, ready to ``\\input`` or paste
    """
    if csv_dir is None:
        csv_dir = os.path.join(REPO, "outputs/cybench/maize-IE/")
    if json_path is None:
        json_path = os.path.join(REPO, "data/predictions/cybench_Qwen3.6_35B/cybench_IE_multi_csv.json")
    if cite_key is None:
        cite_key = "kallenberg2026cy"
    if excerpt is None:
        excerpt = "location_maize_IE.csv"

    csv_dir = Path(csv_dir)
    frames = read_csv_dir(csv_dir)
    meta = json.loads(Path(json_path).read_text())

    types = field_types(meta)
    keys = shared_columns(frames)
    total_fields = sum(len(df.columns) for df in frames.values())

    if caption is None:
        caption = (
            rf"\textbf{{Example of metadata generation for a multi-file dataset.}} "
            rf"The $\name$ receives {len(frames)} related CSV files (top) and emits a "
            rf"Croissant record (bottom), spanning descriptive fields (left) and "
            rf"per-file {_tt('recordsets')} (right). The generated metadata is formatted as a JSON file, which has been human-readable for this figure. No creator-annotated metadata "
            rf"exists for this dataset, as CY-Bench was annotated at the level of all its datasets, so no reference is shown."
        )
    if cite_key:
        caption += rf" Data from \cite{{{cite_key}}}."

    preamble = r"""% Requires in the preamble:
%   \usepackage{tcolorbox}
%   \usepackage{booktabs}
%   \tcbuselibrary{skins, breakable}
% Auto-generated by make_metadata_figure.py -- edits will be overwritten.
"""

    return "\n".join([
        preamble,
        r"\begin{figure*}[t]",
        r"\centering",
        "",
        r"% -- Context (full width) --",
        _context_box(frames, types, keys, total_fields, excerpt_name=excerpt),
        "",
        r"\smallskip",
        "",
        r"% -- Predicted metadata (side by side) --",
        _descriptive_box(meta, frames),
        r"\hfill",
        _structural_box(meta, excerpt=excerpt),
        "",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{figure*}",
        "",
    ])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv-dir", default=None, help="Directory of dataset .csv files")
    p.add_argument("--json", default=None, help="Predicted-metadata JSON")
    p.add_argument("--out", default="example_annotation_cybench.tex", help="Output .tex path")
    p.add_argument("--cite", default=None, help="BibTeX key for the data citation")
    p.add_argument("--label", default="fig:example_metadata_multifile")
    p.add_argument("--excerpt", default=None,
                   help="Filename whose recordset is shown in full")
    args = p.parse_args()

    tex = make_metadata_figure(
        csv_dir=args.csv_dir,
        json_path=args.json,
        cite_key=args.cite,
        label=args.label,
        excerpt=args.excerpt,
    )
    Path(args.out).write_text(tex)
    print(f"Written {args.out} ({len(tex.splitlines())} lines)")


if __name__ == "__main__":
    main()