"""Hierarchy visualisation generator.

Builds interactive Plotly HTML (DAG, Treemap, Sunburst) from the
community hierarchy stored in the SQLite database.

Can be used as:
  - A library:  ``from archrag.visualization import generate_visualization``
  - A CLI:      ``python -m archrag.visualization``
"""

from __future__ import annotations

import json
import logging
import sqlite3
import textwrap
from pathlib import Path

import plotly.graph_objects as go

log = logging.getLogger(__name__)

# ── Colours per level ───────────────────────────────────────────────────────

LEVEL_COLOURS = [
    "#4C78A8",  # level 0 – steel blue
    "#F58518",  # level 1 – orange
    "#E45756",  # level 2 – coral
    "#72B7B2",  # level 3 – teal
    "#54A24B",  # level 4 – green
    "#EECA3B",  # level 5 – gold
    "#B279A2",  # level 6 – mauve
    "#FF9DA6",  # level 7 – pink
]


def _colour(level: int) -> str:
    return LEVEL_COLOURS[level % len(LEVEL_COLOURS)]


# ── Data loading ────────────────────────────────────────────────────────────


def _load_hierarchy(db_path: str):
    """Return (communities_by_id, hierarchy_level_ids, entities_by_id)."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    meta_row = cur.execute(
        "SELECT value FROM meta WHERE key='hierarchy_structure'"
    ).fetchone()
    if not meta_row:
        conn.close()
        return {}, [], {}

    structure = json.loads(meta_row[0])
    level_ids: list[list[str]] = structure["level_ids"]

    hierarchy_cids: set[str] = set()
    for ids in level_ids:
        hierarchy_cids.update(ids)

    comms: dict[str, dict] = {}
    for row in cur.execute(
        "SELECT id, level, member_ids, summary FROM communities"
    ).fetchall():
        cid, level, members_json, summary = row
        if cid in hierarchy_cids:
            comms[cid] = {
                "id": cid,
                "level": level,
                "member_ids": json.loads(members_json),
                "summary": summary or "(no summary)",
            }

    entities: dict[str, dict] = {}
    for row in cur.execute(
        "SELECT id, name, description, entity_type FROM entities"
    ).fetchall():
        entities[row[0]] = {
            "id": row[0],
            "name": row[1],
            "description": row[2],
            "entity_type": row[3],
        }

    conn.close()
    return comms, level_ids, entities


# ── DAG figure ──────────────────────────────────────────────────────────────


def _make_dag_figure(comms, level_ids):
    """Create a Plotly DAG of community hierarchy."""
    import networkx as nx

    G = nx.DiGraph()
    for cid, c in comms.items():
        short = textwrap.shorten(c["summary"], width=90, placeholder="…")
        G.add_node(cid, level=c["level"], summary=c["summary"], label=short,
                   member_ids=c["member_ids"])

    for level_idx in range(1, len(level_ids)):
        for parent_id in level_ids[level_idx]:
            if parent_id not in comms:
                continue
            for child_id in comms[parent_id]["member_ids"]:
                if child_id in comms:
                    G.add_edge(child_id, parent_id)

    # Layout
    pos: dict[str, tuple[float, float]] = {}
    for level_idx, ids in enumerate(level_ids):
        n = len(ids)
        for i, cid in enumerate(ids):
            if cid in G:
                pos[cid] = ((i - (n - 1) / 2) * 2.0, level_idx * 3.0)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y,
                            line=dict(width=1.2, color="#888"),
                            hoverinfo="none", mode="lines", name="edges")

    node_traces = []
    for level_idx, ids in enumerate(level_ids):
        xs, ys, texts, hovers, sizes = [], [], [], [], []
        for cid in ids:
            if cid not in pos:
                continue
            x, y = pos[cid]
            data = G.nodes[cid]
            xs.append(x); ys.append(y)
            texts.append(textwrap.shorten(data.get("label", cid[:8]), 40, placeholder="…"))
            hovers.append(
                f"<b>Level {level_idx}</b><br>"
                f"<b>ID:</b> {cid[:10]}<br>"
                f"<b>Members:</b> {len(data.get('member_ids', []))}<br><br>"
                f"<i>{textwrap.fill(data.get('summary', ''), 60)}</i>"
            )
            sizes.append(max(18, min(50, 12 + len(data.get("member_ids", [])) * 3)))

        node_traces.append(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            marker=dict(size=sizes, color=_colour(level_idx),
                        line=dict(width=1.5, color="white"), opacity=0.92),
            text=texts, textposition="top center",
            textfont=dict(size=9, color="#333"),
            hovertext=hovers, hoverinfo="text",
            name=f"Level {level_idx}",
        ))

    fig = go.Figure(data=[edge_trace] + node_traces)
    fig.update_layout(
        title=dict(text="Community Hierarchy — DAG View<br>"
                   "<sup>Each node is a community; edges show containment</sup>",
                   font=dict(size=18)),
        showlegend=True,
        legend=dict(title="Hierarchy Level", bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#ccc", borderwidth=1),
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="Level ↑"),
        plot_bgcolor="white", margin=dict(l=20, r=20, t=80, b=20), height=800,
    )
    return fig


# ── Treemap ─────────────────────────────────────────────────────────────────


def _make_treemap(comms, level_ids, entities):
    ids_list = ["ROOT"]
    labels = ["Knowledge Graph"]
    parents = [""]
    values = [0]
    colours = ["#f0f0f0"]
    hovertexts = ["Root of the community hierarchy"]

    child_to_parent: dict[str, str] = {}
    for level_idx in range(len(level_ids) - 1, 0, -1):
        for parent_id in level_ids[level_idx]:
            if parent_id not in comms:
                continue
            for child_id in comms[parent_id]["member_ids"]:
                if child_id in comms and child_id not in child_to_parent:
                    child_to_parent[child_id] = parent_id

    for level_idx in range(len(level_ids) - 1, -1, -1):
        for cid in level_ids[level_idx]:
            if cid not in comms:
                continue
            c = comms[cid]
            parent = child_to_parent.get(cid, "ROOT")
            short_summary = textwrap.shorten(c["summary"], width=60, placeholder="…")
            ids_list.append(cid)
            labels.append(f"L{c['level']}: {short_summary}")
            parents.append(parent)
            colours.append(_colour(c["level"]))
            hovertexts.append(
                f"<b>Level {c['level']}</b> | ID: {cid[:10]}<br>"
                f"Members: {len(c['member_ids'])}<br><br>"
                f"{textwrap.fill(c['summary'], 70)}"
            )
            values.append(0)

    for cid in level_ids[0]:
        if cid not in comms:
            continue
        for eid in comms[cid]["member_ids"]:
            ent = entities.get(eid)
            if ent:
                ids_list.append(f"entity_{eid}")
                labels.append(ent["name"])
                parents.append(cid)
                values.append(1)
                colours.append("#d9e8f7")
                hovertexts.append(
                    f"<b>{ent['name']}</b>"
                    f"{' [' + ent['entity_type'] + ']' if ent['entity_type'] else ''}<br>"
                    f"{ent['description'][:200]}"
                )

    fig = go.Figure(go.Treemap(
        ids=ids_list, labels=labels, parents=parents, values=values,
        branchvalues="total",
        marker=dict(colors=colours, line=dict(width=1.5, color="white")),
        hovertext=hovertexts, hoverinfo="text", textinfo="label",
        textfont=dict(size=11),
        pathbar=dict(visible=True, textfont=dict(size=12)),
        maxdepth=3,
    ))
    fig.update_layout(
        title=dict(text="Community Hierarchy — Encapsulatory Treemap<br>"
                   "<sup>Click a tile to drill into its sub-communities</sup>",
                   font=dict(size=18)),
        margin=dict(l=10, r=10, t=80, b=10), height=750,
    )
    return fig


# ── Sunburst ────────────────────────────────────────────────────────────────


def _make_sunburst(comms, level_ids, entities):
    ids_list = ["ROOT"]
    labels = ["Knowledge Graph"]
    parents = [""]
    values = [0]
    colours = ["#f0f0f0"]
    hovertexts = ["Root"]

    child_to_parent: dict[str, str] = {}
    for level_idx in range(len(level_ids) - 1, 0, -1):
        for parent_id in level_ids[level_idx]:
            if parent_id not in comms:
                continue
            for child_id in comms[parent_id]["member_ids"]:
                if child_id in comms and child_id not in child_to_parent:
                    child_to_parent[child_id] = parent_id

    for level_idx in range(len(level_ids) - 1, -1, -1):
        for cid in level_ids[level_idx]:
            if cid not in comms:
                continue
            c = comms[cid]
            parent = child_to_parent.get(cid, "ROOT")
            short = textwrap.shorten(c["summary"], 50, placeholder="…")
            ids_list.append(cid)
            labels.append(f"L{c['level']}: {short}")
            parents.append(parent)
            values.append(0)
            colours.append(_colour(c["level"]))
            hovertexts.append(
                f"<b>Level {c['level']}</b><br>{textwrap.fill(c['summary'], 60)}"
            )

    for cid in level_ids[0]:
        if cid not in comms:
            continue
        for eid in comms[cid]["member_ids"]:
            ent = entities.get(eid)
            if ent:
                ids_list.append(f"entity_{eid}")
                labels.append(ent["name"])
                parents.append(cid)
                values.append(1)
                colours.append("#d9e8f7")
                hovertexts.append(f"<b>{ent['name']}</b><br>{ent['description'][:150]}")

    fig = go.Figure(go.Sunburst(
        ids=ids_list, labels=labels, parents=parents, values=values,
        branchvalues="total",
        marker=dict(colors=colours, line=dict(width=1, color="white")),
        hovertext=hovertexts, hoverinfo="text", textfont=dict(size=10),
        maxdepth=3,
    ))
    fig.update_layout(
        title=dict(text="Community Hierarchy — Sunburst<br>"
                   "<sup>Concentric rings = hierarchy levels. Click to drill in.</sup>",
                   font=dict(size=18)),
        margin=dict(l=10, r=10, t=80, b=10), height=750,
    )
    return fig


# ── Public API ──────────────────────────────────────────────────────────────

_VIZ_PATH = Path("data/hierarchy_viz.html")


def generate_visualization(
    db_path: str = "data/archrag.db",
    out_path: str | Path | None = None,
) -> str:
    """Generate the combined HTML visualization and return the HTML string.

    Args:
        db_path: Path to the SQLite database.
        out_path: Where to write the HTML file.  Defaults to ``data/hierarchy_viz.html``.

    Returns:
        The full HTML string.
    """
    if out_path is None:
        out_path = _VIZ_PATH
    out_path = Path(out_path)

    comms, level_ids, entities = _load_hierarchy(db_path)

    if not comms:
        html = _empty_html()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")
        log.info("Visualization generated (empty DB) → %s", out_path)
        return html

    dag_fig = _make_dag_figure(comms, level_ids)
    treemap_fig = _make_treemap(comms, level_ids, entities)
    sunburst_fig = _make_sunburst(comms, level_ids, entities)

    dag_html = dag_fig.to_html(full_html=False, include_plotlyjs=False)
    treemap_html = treemap_fig.to_html(full_html=False, include_plotlyjs=False)
    sunburst_html = sunburst_fig.to_html(full_html=False, include_plotlyjs=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>ArchRAG Community Hierarchy</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; background: #fafafa; }}
  .tabs {{ display: flex; gap: 0; background: #fff; border-bottom: 2px solid #e0e0e0;
           padding: 0 20px; position: sticky; top: 0; z-index: 10; }}
  .tab {{ padding: 12px 28px; cursor: pointer; border-bottom: 3px solid transparent;
          font-size: 15px; color: #555; transition: all 0.2s; user-select: none; }}
  .tab:hover {{ color: #222; background: #f5f5f5; }}
  .tab.active {{ color: #4C78A8; border-bottom-color: #4C78A8; font-weight: 600; }}
  .panel {{ display: none; padding: 10px 20px; }}
  .panel.active {{ display: block; }}
  h2 {{ margin: 18px 20px 4px; color: #333; }}
  .desc {{ margin: 4px 20px 10px; color: #777; font-size: 13px; }}
</style>
</head>
<body>
<h2>ArchRAG — Community Hierarchy Visualisation</h2>
<p class="desc">
  Interactive views of the multi-level attributed community hierarchy stored in
  the knowledge graph. Communities at each level encapsulate entities or
  sub-communities from the level below. Summaries are LLM-generated
  reverse-encodings that describe the community's contents.
</p>
<div class="tabs">
  <div class="tab active" onclick="show('dag')">🌳 DAG View</div>
  <div class="tab" onclick="show('treemap')">📦 Treemap</div>
  <div class="tab" onclick="show('sunburst')">🎯 Sunburst</div>
</div>
<div id="dag" class="panel active">{dag_html}</div>
<div id="treemap" class="panel">{treemap_html}</div>
<div id="sunburst" class="panel">{sunburst_html}</div>
<script>
function show(id) {{
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  event.target.classList.add('active');
  setTimeout(() => window.dispatchEvent(new Event('resize')), 50);
}}
</script>
</body>
</html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    log.info(
        "Visualization generated: %d communities, %d entities → %s",
        len(comms), len(entities), out_path,
    )
    return html


def _empty_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"/><title>ArchRAG — No Data</title>
<style>body{font-family:'Segoe UI',system-ui,sans-serif;display:flex;
align-items:center;justify-content:center;height:100vh;margin:0;
background:#fafafa;color:#555;}
.box{text-align:center;}</style></head>
<body><div class="box"><h2>ArchRAG Community Hierarchy</h2>
<p>No hierarchy data found. Index a corpus first.</p></div></body></html>"""


# ── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Generate hierarchy visualization.")
    parser.add_argument("--db", default="data/archrag.db")
    parser.add_argument("-o", "--output", default="data/hierarchy_viz.html")
    args = parser.parse_args()

    generate_visualization(db_path=args.db, out_path=args.output)
    print(f"✅  Written to {Path(args.output).resolve()}")
