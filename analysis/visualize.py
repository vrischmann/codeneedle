#!/usr/bin/env python3
"""Generate Plotly comparison dashboards from results/*.json.

Layout: one chart per page, grouped under a per-corpus subfolder.

    analysis/charts/
      index.html                          ← top-level: links per corpus
      <corpus>/
        index.html                        ← corpus dashboard with chart links
        leaderboard.html                  ← chart 1 standalone
        per-function.html                 ← chart 2 standalone
        recall-vs-position.html           ← chart 3 standalone

Each chart sizes itself to the data and reserves enough room for a vertical
legend with up to 20 model entries. Every chart is fully interactive — hover,
zoom, pan, click-to-toggle-trace, double-click-to-isolate.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


# This file lives in analysis/, so REPO_ROOT is one level up.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))   # so `import bench…` works regardless of cwd
PASS_THRESHOLD = 8    # matches bench/scorer.py
LEGEND_ROW_PX = 26    # how much vertical space each legend entry needs

# Stable color palette — assigned once per model so every chart uses the same color.
PALETTE = [
    "#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2",
    "#ff9da6", "#9d755d", "#bab0ac", "#b279a2", "#eeca3b",
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


@dataclass
class Run:
    path: Path
    model: str
    group_name: str
    data: dict


def _group_name(data: dict) -> str:
    files = data.get("files") or ([data["source"]] if data.get("source") else [])
    if not files:
        return "unknown"
    if len(files) == 1:
        return Path(files[0]).stem
    return "+".join(Path(f).stem for f in files[:3])


def load_runs(results_dir: Path) -> dict[str, list[Run]]:
    groups: dict[str, list[Run]] = defaultdict(list)
    for p in sorted(results_dir.glob("*.json")):
        if p.name.endswith("__debug.json"):
            continue

        try:
            data = json.loads(p.read_text())
        except Exception as e:
            print(f"skip {p.name}: {e}", file=sys.stderr)
            continue
        if isinstance(data, list) or not data.get("results"):
            continue
        group = _group_name(data)
        groups[group].append(Run(path=p, model=data.get("model", p.stem), group_name=group, data=data))
    return groups


def resolve_line_positions(runs: list[Run]) -> dict[str, int]:
    """Map function name → start_line. Re-extracts from source so the depth
    chart works for old dumps too. Looks up files by basename under fixtures/
    if the original absolute path no longer exists (e.g. after a repo rename).
    """
    from bench.extract import extract as bench_extract

    fixtures_dirs = [REPO_ROOT / "fixtures"]
    name_to_line: dict[str, int] = {}
    tried: set[Path] = set()
    for run in runs:
        for raw in run.data.get("files") or ([run.data.get("source")] if run.data.get("source") else []):
            if not raw:
                continue
            p = Path(raw)
            if not p.exists():
                for d in fixtures_dirs:
                    alt = d / p.name
                    if alt.exists():
                        p = alt
                        break
                else:
                    continue
            if p in tried:
                continue
            tried.add(p)
            try:
                for t in bench_extract(p):
                    name_to_line.setdefault(t.name, t.start_line)
            except Exception as e:
                print(f"  (couldn't re-extract {p.name} for depth chart: {e})", file=sys.stderr)
    return name_to_line


def assign_colors(runs: list[Run]) -> dict[str, str]:
    models = sorted({r.model for r in runs})
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}


def _legend_kwargs() -> dict:
    """Right-side vertical legend, padded box, room for many entries."""
    return dict(
        orientation="v",
        yanchor="top", y=1,
        xanchor="left", x=1.02,
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="#ddd",
        borderwidth=1,
        font=dict(size=12),
    )


def _avg_latency(results: list[dict]) -> float | None:
    """Average latency in seconds across non-error results, or None if no data."""
    lats = [r.get("latency_s") for r in results if r.get("latency_s") is not None and not r.get("error")]
    return sum(lats) / len(lats) if lats else None


def _throughput_stats(results: list[dict]) -> dict:
    """Compute average generation throughput from available data.

    Prefers server-reported `timings.predicted_per_second` (llama-server),
    falls back to computing `completion_tokens / latency_s` (both servers).
    Returns dict with keys: gen_tok_s, prompt_tok_s, avg_cache_n (all optional).
    """
    non_err = [r for r in results if not r.get("error")]
    if not non_err:
        return {}

    gen_speeds = []
    prompt_speeds = []
    cache_ns = []
    for r in non_err:
        timings = r.get("timings") or {}
        # Prefer server-reported speeds (more accurate).
        if timings.get("predicted_per_second") is not None:
            gen_speeds.append(float(timings["predicted_per_second"]))
        elif r.get("completion_tokens") and r.get("latency_s"):
            # Fallback: compute from token counts.
            gen_speeds.append(r["completion_tokens"] / r["latency_s"])

        if timings.get("prompt_per_second") is not None:
            prompt_speeds.append(float(timings["prompt_per_second"]))
        if timings.get("cache_n") is not None:
            cache_ns.append(int(timings["cache_n"]))

    out: dict = {}
    if gen_speeds:
        out["gen_tok_s"] = sum(gen_speeds) / len(gen_speeds)
    if prompt_speeds:
        out["prompt_tok_s"] = sum(prompt_speeds) / len(prompt_speeds)
    if cache_ns:
        out["avg_cache_n"] = sum(cache_ns) / len(cache_ns)
    return out


def _chart_height(*, content_rows: int, n_legend_entries: int, base: int = 420) -> int:
    """Pick a height tall enough for both the data rows and the legend.

    `content_rows` is the number of bars/lines/etc. shown vertically.
    `n_legend_entries` is the legend item count.
    """
    by_legend = LEGEND_ROW_PX * n_legend_entries + 120
    by_content = base + 20 * max(0, content_rows - 8)
    return max(base, by_legend, by_content)


# --- charts ---------------------------------------------------------------


def leaderboard(runs: list[Run], colors: dict[str, str]):
    """Horizontal bar chart, one trace per run (so each is independently
    toggleable from the legend). Sorted best → worst by primary lines matched.
    """
    import plotly.graph_objects as go

    rows = []
    for r in runs:
        matched = sum(x.get("primary_matched", 0) for x in r.data["results"])
        total = sum(x.get("primary_total", 0) for x in r.data["results"])
        passed = sum(1 for x in r.data["results"] if x.get("passed"))
        queries = len(r.data["results"])
        halluc = sum(x.get("hallucinated", 0) for x in r.data["results"])
        errored = sum(1 for x in r.data["results"] if x.get("error"))
        avg_lat = _avg_latency(r.data["results"])
        tp = _throughput_stats(r.data["results"])
        rows.append({
            "model": r.model, "stem": r.path.stem,
            "matched": matched, "total": total,
            "passed": passed, "queries": queries,
            "halluc": halluc, "errored": errored,
            "avg_latency": avg_lat,
            "gen_tok_s": tp.get("gen_tok_s"),
            "prompt_tok_s": tp.get("prompt_tok_s"),
            "avg_cache_n": tp.get("avg_cache_n"),
        })
    rows.sort(key=lambda d: d["matched"], reverse=True)

    if not rows:
        return None

    max_total = max(r["total"] for r in rows) or 1

    fig = go.Figure()
    for row in rows:
        lat_str = f" · {row['avg_latency']:.1f}s avg" if row['avg_latency'] is not None else ""
        gen_str = f" · {row['gen_tok_s']:.0f} tok/s gen" if row.get('gen_tok_s') is not None else ""
        annotation = (
            f"{row['matched']}/{row['total']} lines · "
            f"{row['passed']}/{row['queries']} pass · "
            f"{row['halluc']} halluc"
            + (f" · {row['errored']} err" if row['errored'] else "")
            + lat_str
            + gen_str
        )
        hover_lat = f"<br>avg latency: {row['avg_latency']:.1f}s" if row['avg_latency'] is not None else ""
        hover_gen = f"<br>gen throughput: {row['gen_tok_s']:.0f} tok/s" if row.get('gen_tok_s') is not None else ""
        hover_prefill = f"<br>prefill: {row['prompt_tok_s']:.0f} tok/s" if row.get('prompt_tok_s') is not None else ""
        hover_cache = f"<br>avg cache: {row['avg_cache_n']:.0f} tok" if row.get('avg_cache_n') is not None else ""
        hover = (
            f"<b>{row['model']}</b><br>"
            f"file: {row['stem']}<br>"
            f"matched: {row['matched']} / {row['total']}<br>"
            f"pass: {row['passed']} / {row['queries']}<br>"
            f"hallucinated: {row['halluc']}<br>"
            f"errored: {row['errored']}"
            + hover_lat + hover_gen + hover_prefill + hover_cache
        )
        fig.add_trace(go.Bar(
            x=[row["matched"]],
            y=[row["stem"]],
            orientation="h",
            name=row["model"],
            legendgroup=row["model"],
            text=[annotation],
            textposition="outside",
            marker_color=colors[row["model"]],
            marker_line_color="#fff",
            marker_line_width=1,
            hovertext=[hover],
            hoverinfo="text",
        ))

    fig.update_layout(
        title="Leaderboard · total primary lines matched (of possible)",
        xaxis=dict(title="lines matched", range=[0, max_total * 1.4]),
        yaxis=dict(autorange="reversed", automargin=True),
        height=_chart_height(content_rows=len(rows), n_legend_entries=len(rows)),
        margin=dict(l=20, r=40, t=70, b=60),
        legend=_legend_kwargs(),
        bargap=0.25,
    )
    return fig


def per_function_bars(runs: list[Run], colors: dict[str, str]):
    """Grouped bars: one bar per (function × run). Dashed line at pass threshold."""
    import plotly.graph_objects as go

    all_fns: set[str] = set()
    for r in runs:
        for x in r.data["results"]:
            all_fns.add(x["function"])

    def mean_score(fn: str) -> float:
        xs = []
        for r in runs:
            x = next((y for y in r.data["results"] if y["function"] == fn), None)
            if x and x.get("primary_total"):
                xs.append(x["primary_matched"] / x["primary_total"])
        return sum(xs) / len(xs) if xs else 0.0

    fns = sorted(all_fns, key=mean_score, reverse=True)
    if not fns:
        return None

    fig = go.Figure()
    total_max = 20
    for r in runs:
        y = []
        for fn in fns:
            x = next((z for z in r.data["results"] if z["function"] == fn), None)
            if x is None or x.get("error"):
                y.append(None)
            else:
                y.append(x.get("primary_matched", 0))
                total_max = max(total_max, x.get("primary_total", 20))
        fig.add_bar(
            x=fns, y=y,
            name=r.model,
            legendgroup=r.model,
            marker_color=colors[r.model],
            customdata=[r.path.stem] * len(fns),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "model: " + r.model + "<br>"
                "run: %{customdata}<br>"
                "matched: %{y}<extra></extra>"
            ),
        )

    fig.add_hline(
        y=PASS_THRESHOLD, line_dash="dash", line_color="#888",
        annotation_text=f"pass threshold ({PASS_THRESHOLD})",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Per-function score · bars above the dashed line passed",
        xaxis=dict(title="function (sorted by average difficulty)", tickangle=-40,
                   automargin=True),
        yaxis=dict(title="primary lines matched", range=[0, total_max + 2]),
        barmode="group",
        bargap=0.15,
        bargroupgap=0.05,
        height=_chart_height(content_rows=len(fns), n_legend_entries=len(runs)),
        margin=dict(l=70, r=40, t=70, b=160),
        legend=_legend_kwargs(),
    )
    return fig


def recall_vs_depth(runs: list[Run], colors: dict[str, str], positions: dict[str, int]):
    """Scatter + line: X = function start line in source, Y = % matched."""
    import plotly.graph_objects as go

    fig = go.Figure()
    any_data = False
    max_line = 0
    for r in runs:
        pts = []
        for x in r.data["results"]:
            if x.get("error"):
                continue
            fn = x["function"]
            if fn not in positions:
                continue
            total = x.get("primary_total") or 20
            pct = x.get("primary_matched", 0) / total * 100
            pts.append((positions[fn], pct, fn, x.get("primary_matched", 0), total))
        if not pts:
            continue
        any_data = True
        pts.sort(key=lambda t: t[0])
        xs = [p[0] for p in pts]
        max_line = max(max_line, max(xs))
        ys = [p[1] for p in pts]
        hover = [
            f"<b>{p[2]}</b><br>line {p[0]:,}<br>"
            f"{p[3]}/{p[4]} matched ({p[1]:.0f}%)"
            f"<br>model: {r.model}<br>run: {r.path.stem}"
            for p in pts
        ]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines+markers",
            name=r.model,
            legendgroup=r.model,
            line=dict(color=colors[r.model], width=2),
            marker=dict(size=10, color=colors[r.model], line=dict(color="#fff", width=1)),
            hovertext=hover, hoverinfo="text",
        ))

    if not any_data:
        return None

    fig.add_hline(
        y=PASS_THRESHOLD / 20 * 100, line_dash="dash", line_color="#888",
        annotation_text=f"pass threshold ({PASS_THRESHOLD}/20 = {PASS_THRESHOLD / 20 * 100:.0f}%)",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title="Recall vs. position in file · left = near top, right = deep",
        xaxis=dict(title="function start line (deeper in file →)",
                   range=[0, max_line * 1.05]),
        yaxis=dict(title="% primary lines matched", range=[-5, 108]),
        height=_chart_height(content_rows=8, n_legend_entries=len(runs), base=520),
        margin=dict(l=70, r=40, t=70, b=70),
        legend=_legend_kwargs(),
    )
    return fig


def response_time(runs: list[Run], colors: dict[str, str]):
    """Grouped bars: per-function response latency (seconds) per model."""
    import plotly.graph_objects as go

    all_fns: set[str] = set()
    for r in runs:
        for x in r.data["results"]:
            all_fns.add(x["function"])

    # Sort functions by average latency across models (slowest first → top of chart)
    def mean_latency(fn: str) -> float:
        xs = []
        for r in runs:
            x = next((y for y in r.data["results"] if y["function"] == fn), None)
            if x and x.get("latency_s") is not None and not x.get("error"):
                xs.append(x["latency_s"])
        return sum(xs) / len(xs) if xs else 0.0

    fns = sorted(all_fns, key=mean_latency, reverse=True)
    if not fns:
        return None

    fig = go.Figure()
    max_lat = 0
    for r in runs:
        y = []
        custom = []
        for fn in fns:
            x = next((z for z in r.data["results"] if z["function"] == fn), None)
            if x is None or x.get("error") or x.get("latency_s") is None:
                y.append(None)
                custom.append("")
            else:
                lat = x["latency_s"]
                y.append(round(lat, 2))
                max_lat = max(max_lat, lat)
                total = x.get("primary_total", 20)
                matched = x.get("primary_matched", 0)
                parts = [f"matched {matched}/{total}"]
                timings = x.get("timings") or {}
                if timings.get("predicted_per_second") is not None:
                    parts.append(f"{timings['predicted_per_second']:.0f} gen tok/s")
                if timings.get("prompt_per_second") is not None:
                    parts.append(f"{timings['prompt_per_second']:.0f} prefill tok/s")
                if timings.get("cache_n") is not None:
                    parts.append(f"{timings['cache_n']} cached tok")
                custom.append("<br>".join(parts))
        fig.add_bar(
            x=fns, y=y,
            name=r.model,
            legendgroup=r.model,
            marker_color=colors[r.model],
            customdata=custom,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "model: " + r.model + "<br>"
                "run: " + r.path.stem + "<br>"
                "latency: %{y:.1f}s<br>"
                "%{customdata}<extra></extra>"
            ),
        )

    fig.update_layout(
        title="Response time per function (seconds)",
        xaxis=dict(title="function (sorted by average latency, slowest → fastest)",
                   tickangle=-40, automargin=True),
        yaxis=dict(title="latency (seconds)", range=[0, max_lat * 1.15]),
        barmode="group",
        bargap=0.15,
        bargroupgap=0.05,
        height=_chart_height(content_rows=len(fns), n_legend_entries=len(runs)),
        margin=dict(l=70, r=40, t=70, b=160),
        legend=_legend_kwargs(),
    )
    return fig


def speed_vs_quality(runs: list[Run], colors: dict[str, str]):
    """Scatter: X = avg latency, Y = recall %. One marker per run. Pareto frontier overlay."""
    import plotly.graph_objects as go

    points = []
    for r in runs:
        results = r.data["results"]
        non_err = [x for x in results if not x.get("error")]
        if not non_err:
            continue
        total_matched = sum(x.get("primary_matched", 0) for x in non_err)
        total_possible = sum(x.get("primary_total", 0) for x in non_err)
        recall_pct = (total_matched / total_possible * 100) if total_possible else 0
        lats = [x["latency_s"] for x in non_err if x.get("latency_s") is not None]
        avg_lat = sum(lats) / len(lats) if lats else None
        if avg_lat is None:
            continue
        passed = sum(1 for x in non_err if x.get("passed"))
        queries = len(results)
        halluc = sum(x.get("hallucinated", 0) for x in non_err)
        # Throughput info
        tp = _throughput_stats(results)
        gen_tok_s = tp.get("gen_tok_s")
        prompt_tok_s = tp.get("prompt_tok_s")
        avg_cache_n = tp.get("avg_cache_n")
        points.append({
            "model": r.model,
            "stem": r.path.stem,
            "avg_lat": avg_lat,
            "recall_pct": recall_pct,
            "matched": total_matched,
            "total": total_possible,
            "passed": passed,
            "queries": queries,
            "halluc": halluc,
            "gen_tok_s": gen_tok_s,
            "prompt_tok_s": prompt_tok_s,
            "avg_cache_n": avg_cache_n,
        })

    if len(points) < 1:
        return None

    fig = go.Figure()

    # Pareto frontier: points where nothing else is both faster (lower x) and more accurate (higher y).
    # Standard approach: sort by x ascending, walk forward tracking max y so far.
    sorted_pts = sorted(points, key=lambda p: (p["avg_lat"], -p["recall_pct"]))
    frontier: list[dict] = []
    best_recall_so_far = -1
    for p in sorted_pts:
        if p["recall_pct"] > best_recall_so_far:
            frontier.append(p)
            best_recall_so_far = p["recall_pct"]

    # Plot Pareto frontier as a dashed line behind the scatter points.
    if len(frontier) >= 2:
        fig.add_trace(go.Scatter(
            x=[p["avg_lat"] for p in frontier],
            y=[p["recall_pct"] for p in frontier],
            mode="lines",
            name="Pareto frontier",
            line=dict(dash="dash", width=2, color="#888"),
            hoverinfo="skip",
            showlegend=True,
        ))

    # One scatter trace per run (each toggleable from legend).
    for p in points:
        fig.add_trace(go.Scatter(
            x=[p["avg_lat"]],
            y=[p["recall_pct"]],
            mode="markers+text",
            name=p["stem"],
            legendgroup=p["model"],
            marker=dict(
                size=14,
                color=colors[p["model"]],
                line=dict(color="#fff", width=2),
            ),
            text=[
                f"{p['recall_pct']:.0f}%"
                + (f" · {p['gen_tok_s']:.0f} tok/s" if p.get('gen_tok_s') is not None else "")
            ],
            textposition="top center",
            textfont=dict(size=11),
            hovertext=[
                f"<b>{p['stem']}</b><br>"
                f"model: {p['model']}<br>"
                f"recall: {p['matched']}/{p['total']} ({p['recall_pct']:.1f}%)<br>"
                f"pass: {p['passed']}/{p['queries']}<br>"
                f"hallucinated: {p['halluc']}<br>"
                f"avg latency: {p['avg_lat']:.1f}s"
                + (f"<br>gen: {p['gen_tok_s']:.0f} tok/s" if p.get('gen_tok_s') is not None else "")
                + (f"<br>prefill: {p['prompt_tok_s']:.0f} tok/s" if p.get('prompt_tok_s') is not None else "")
                + (f"<br>avg cache: {p['avg_cache_n']:.0f} tok" if p.get('avg_cache_n') is not None else "")
            ],
            hoverinfo="text",
        ))

    # Axis ranges with padding
    x_vals = [p["avg_lat"] for p in points]
    y_vals = [p["recall_pct"] for p in points]
    x_min, x_max = min(x_vals), max(x_vals)
    x_pad = max((x_max - x_min) * 0.1, 1)
    y_min, y_max = min(y_vals), max(y_vals)
    y_pad = max((y_max - y_min) * 0.1, 5)

    # Pass threshold reference line
    pass_pct = PASS_THRESHOLD / 20 * 100

    fig.add_hline(
        y=pass_pct, line_dash="dot", line_color="#aaa",
        annotation_text=f"pass threshold ({pass_pct:.0f}%)",
        annotation_position="bottom left",
    )
    fig.update_layout(
        title="Speed vs. quality · top-left is best (fast + accurate)",
        xaxis=dict(title="average response latency (seconds) →",
                   range=[max(0, x_min - x_pad), x_max + x_pad]),
        yaxis=dict(title="% primary lines matched ↑",
                   range=[max(0, y_min - y_pad), min(105, y_max + y_pad)]),
        height=_chart_height(content_rows=1, n_legend_entries=len(points) + 1, base=520),
        margin=dict(l=70, r=40, t=70, b=70),
        legend=_legend_kwargs(),
    )
    return fig


# --- HTML assembly --------------------------------------------------------


PAGE_CSS = """
  *{box-sizing:border-box;}
  body{font-family:system-ui,-apple-system,sans-serif;margin:0;padding:1.5rem 1.25rem;color:#222;
       background:#fafafa;min-height:100vh;}
  .wrap{max-width:1500px;margin:0 auto;}
  header{font-size:.9rem;color:#666;margin-bottom:.5rem;}
  header a{color:#4c78a8;text-decoration:none;}
  header a:hover{text-decoration:underline;}
  header .corpus{font-weight:600;color:#222;}
  nav{margin:.25rem 0 1.5rem 0;font-size:.95rem;border-bottom:1px solid #e5e5e5;padding-bottom:.5rem;}
  nav a{color:#4c78a8;text-decoration:none;margin-right:1rem;padding:.25rem 0;display:inline-block;}
  nav a.active{color:#222;font-weight:600;border-bottom:2px solid #4c78a8;}
  nav a:hover{text-decoration:underline;}
  h1{margin:.25rem 0;font-size:1.5rem;}
  p.caption{color:#555;margin:.25rem 0 1rem 0;font-size:.95rem;line-height:1.5;}
  .chart{background:#fff;border:1px solid #e5e5e5;border-radius:8px;padding:.5rem;
         box-shadow:0 1px 3px rgba(0,0,0,.04);overflow-x:auto;}
  ul{padding-left:1.25rem;}
  li{margin:.4rem 0;}
  small{color:#888;}
"""


CHART_PAGES = [
    # (slug, title, caption, chart_fn_key)
    ("leaderboard", "Leaderboard",
     "Total primary lines matched across all tested functions, sorted so the top bar is the best run. "
     "Each model has its own legend entry — click to hide/show, double-click to isolate. "
     "`halluc` = lines the model emitted that don't match the expected window.",
     "leaderboard"),
    ("per-function", "Per-function score",
     "One bar per model for each function, sorted left-to-right easiest → hardest. "
     "Bars above the dashed line passed (≥ 8 of 20 primary lines matched). "
     "Toggle a model in the legend to remove it from every cluster.",
     "per_function"),
    ("recall-vs-position", "Recall vs. position in file",
     "Each marker is a function placed at its line number in the source. "
     "If recall falls off as x increases, the model is losing context as depth grows — the "
     "core finding for sliding-window models. Hover any marker for details.",
     "recall_vs_position"),
    ("response-time", "Response time",
     "Per-function response latency (seconds) for each model. Shorter bars mean faster responses. "
     "Functions are sorted by average latency across models. Hover a bar for exact timing details.",
     "response_time"),
    ("speed-vs-quality", "Speed vs. quality",
     "Each marker is one run, placed by its average response latency (x) and recall (y). "
     "The Pareto frontier (dashed line) connects runs where nothing else is both faster and more accurate. "
     "Points below the frontier are dominated — a better tradeoff exists. "
     "Hover any marker for full details.",
     "speed_vs_quality"),
]


def write_chart_page(out_path: Path, group: str, slug: str, title: str, caption: str,
                     fig, all_pages: list[tuple[str, str]]) -> None:
    import plotly.io as pio

    # Nav between charts of this corpus.
    nav_links = " ".join(
        f'<a href="{s}.html" class="{"active" if s == slug else ""}">{t}</a>'
        for s, t in all_pages
    )

    chart_html = pio.to_html(
        fig,
        include_plotlyjs="cdn",
        full_html=False,
        config={"responsive": True, "displaylogo": False},
    )

    body = (
        f'<div class="wrap">'
        f'<header><a href="../index.html">← all corpora</a> · '
        f'<span class="corpus">{group}</span></header>'
        f'<nav>{nav_links}</nav>'
        f'<h1>{title}</h1>'
        f'<p class="caption">{caption}</p>'
        f'<div class="chart">{chart_html}</div>'
        f'</div>'
    )
    out_path.write_text(
        f'<!doctype html><html><head><meta charset="utf-8">'
        f'<title>{group} · {title}</title>'
        f'<style>{PAGE_CSS}</style></head><body>{body}</body></html>'
    )


def write_corpus_index(out_path: Path, group: str, runs: list[Run],
                       generated_pages: list[tuple[str, str]]) -> None:
    models = sorted({r.model for r in runs})
    queries = sum(len(r.data["results"]) for r in runs)
    # Compute overall average latency across all runs
    all_lats = []
    for r in runs:
        for x in r.data["results"]:
            if x.get("latency_s") is not None and not x.get("error"):
                all_lats.append(x["latency_s"])
    avg_lat_str = f" · avg latency: {sum(all_lats) / len(all_lats):.1f}s" if all_lats else ""
    items = "".join(
        f'<li><a href="{slug}.html">{title}</a></li>'
        for slug, title in generated_pages
    )
    body = (
        f'<div class="wrap">'
        f'<header><a href="../index.html">← all corpora</a></header>'
        f'<h1>{group}</h1>'
        f'<p class="caption">{len(runs)} run(s) · {queries} queries{avg_lat_str} · '
        f'{len(models)} unique model(s): {", ".join(models)}</p>'
        f'<ul>{items}</ul>'
        f'</div>'
    )
    out_path.write_text(
        f'<!doctype html><html><head><meta charset="utf-8">'
        f'<title>{group} · charts</title>'
        f'<style>{PAGE_CSS}</style></head><body>{body}</body></html>'
    )


def write_dashboard(group: str, runs: list[Run], out_dir: Path) -> list[tuple[str, str]]:
    """Write all chart pages for one corpus. Returns list of (slug, title) actually generated."""
    colors = assign_colors(runs)
    positions = resolve_line_positions(runs)

    figs = {
        "leaderboard": leaderboard(runs, colors),
        "per_function": per_function_bars(runs, colors),
        "recall_vs_position": recall_vs_depth(runs, colors, positions),
        "response_time": response_time(runs, colors),
        "speed_vs_quality": speed_vs_quality(runs, colors),
    }

    chart_dir = out_dir / group
    chart_dir.mkdir(parents=True, exist_ok=True)

    generated: list[tuple[str, str]] = []
    nav_pages: list[tuple[str, str]] = []
    for slug, title, _caption, fig_key in CHART_PAGES:
        if figs.get(fig_key) is not None:
            nav_pages.append((slug, title))

    for slug, title, caption, fig_key in CHART_PAGES:
        fig = figs.get(fig_key)
        if fig is None:
            continue
        page_path = chart_dir / f"{slug}.html"
        write_chart_page(page_path, group, slug, title, caption, fig, nav_pages)
        generated.append((slug, title))

    write_corpus_index(chart_dir / "index.html", group, runs, generated)
    return generated


def write_top_index(groups: dict[str, list[Run]], out_dir: Path) -> Path:
    idx = out_dir / "index.html"
    items = []
    for name in sorted(groups):
        runs = groups[name]
        models = sorted({r.model for r in runs})
        items.append(
            f'<li><a href="{name}/index.html">{name}</a> '
            f'<small>— {len(runs)} run(s), {len(models)} model(s): {", ".join(models)}</small></li>'
        )
    body = (
        f'<div class="wrap">'
        f'<h1>codeneedle · benchmark dashboards</h1>'
        f'<ul>{"".join(items)}</ul>'
        f'</div>'
    )
    idx.write_text(
        f'<!doctype html><html><head><meta charset="utf-8">'
        f'<title>codeneedle dashboards</title>'
        f'<style>{PAGE_CSS}</style></head><body>{body}</body></html>'
    )
    return idx


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="default: analysis/charts/")
    args = ap.parse_args(argv)

    out_dir = args.output_dir or (REPO_ROOT / "analysis" / "charts")
    groups = load_runs(args.results_dir)
    if not groups:
        print(f"no usable result JSON files in {args.results_dir}")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    total_runs = sum(len(r) for r in groups.values())
    print(f"Loaded {total_runs} run(s) in {len(groups)} group(s)")
    for name, runs in sorted(groups.items()):
        generated = write_dashboard(name, runs, out_dir)
        slugs = ", ".join(s for s, _ in generated)
        print(f"  {name}: {len(runs)} run(s) → {out_dir / name}/{{ {slugs} }}.html")

    idx = write_top_index(groups, out_dir)
    print(f"\nopen {idx}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
