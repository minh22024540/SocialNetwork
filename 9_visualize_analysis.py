#!/usr/bin/env python3
"""
Generate quick visualizations for previously computed analysis outputs.

This script consumes JSON outputs from:
- 7_analyze_small_world.py  -> data_analysis/small_world_analysis.json
- 7_analyze_pagerank.py     -> data_analysis/pagerank_scores.json
- 7_analyze_communities.py  -> data_analysis/communities.json

It produces PNG charts in data_analysis/plots/:
- small_world_metrics.png        : bar chart of avg path length vs clustering
- pagerank_top_persons.png       : top-N persons by PageRank
- pagerank_top_events.png        : top-N events by PageRank
- community_size_distribution.png: histogram of community sizes
- community_top_sizes.png        : bar chart of largest communities

Dependencies: matplotlib (and optional seaborn for nicer style).
Install if needed: `pip install matplotlib seaborn`
"""

from __future__ import annotations

import json
import random
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict

import networkx as nx

import matplotlib.pyplot as plt

try:
    import seaborn as sns

    sns.set_theme(style="whitegrid")
except ImportError:
    sns = None  # graceful fallback to plain matplotlib

# Optional Plotly for interactive, non-overlapping bubbles
try:
    import plotly.express as px
    import plotly.io as pio
    import plotly.graph_objects as go
except ImportError:
    px = None
    pio = None
    go = None

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ANALYSIS = PROJECT_ROOT / "data_analysis"
PLOTS_DIR = DATA_ANALYSIS / "plots"


def load_json(path: Path) -> Dict:
    """Load JSON file if it exists, else return empty dict."""
    if not path.exists():
        print(f"[warn] Missing file: {path}")
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_output_dir() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_graph_subset(jsonl_path: Path, node_filter: set) -> nx.Graph:
    """Load only nodes/edges where both endpoints are in node_filter."""
    G = nx.Graph()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            wp = obj.get("wikipedia", {})
            src = wp.get("id")
            if not isinstance(src, int) or src not in node_filter:
                continue
            category = obj.get("category")
            title = wp.get("title", "")
            G.add_node(src, category=category, title=title)
            for link in obj.get("link_weights", []):
                tgt = link.get("target_id")
                w = link.get("weight", 1.0)
                if isinstance(tgt, int) and tgt in node_filter:
                    if not G.has_edge(src, tgt):
                        G.add_edge(src, tgt, weight=w)
    return G


def load_full_graph(jsonl_path: Path) -> nx.Graph:
    """Load the full graph (persons + events) from JSONL."""
    G = nx.Graph()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            wp = obj.get("wikipedia", {})
            src = wp.get("id")
            category = obj.get("category")
            title = wp.get("title", "")
            if not isinstance(src, int):
                continue
            G.add_node(src, category=category, title=title)
            for link in obj.get("link_weights", []):
                tgt = link.get("target_id")
                w = link.get("weight", 1.0)
                if isinstance(tgt, int):
                    if not G.has_edge(src, tgt):
                        G.add_edge(src, tgt, weight=w)
    return G


def plot_small_world() -> None:
    data = load_json(DATA_ANALYSIS / "small_world_analysis.json")
    metrics = data.get("small_world_metrics", {})
    path_stats = metrics.get("path_statistics", {})

    if not metrics:
        print("[info] Skip small-world plot (no data).")
        return

    avg_path = metrics.get("average_shortest_path_length")
    clustering = metrics.get("clustering_coefficient")
    connected = path_stats.get("connected_pairs", 0)
    disconnected = path_stats.get("disconnected_pairs", 0)

    # Bar view for quick takeaway
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Avg shortest path", "Clustering coeff"], [avg_path, clustering], color=["#4C72B0", "#55A868"])
    ax.set_title("Small-world metrics (people-only endpoints)")
    ax.set_ylabel("Value")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
    for i, val in enumerate([avg_path, clustering]):
        ax.text(i, val + 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax.text(
        0.5,
        ax.get_ylim()[1] * 0.85,
        f"Pairs: {connected} connected / {disconnected} disconnected",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout()
    out_path = PLOTS_DIR / "small_world_metrics.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[ok] Saved {out_path}")

    # Scatter view vs classical small-world thresholds (dot = your network)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(avg_path, clustering, s=160, color="#4C72B0", edgecolor="black", zorder=3)
    ax.axvline(6.0, color="red", linestyle="--", alpha=0.5, label="Typical small-world path ≈ 6")
    ax.axhline(0.1, color="orange", linestyle="--", alpha=0.6, label="Baseline clustering 0.1")
    ax.set_xlabel("Average shortest path length (persons)")
    ax.set_ylabel("Clustering coefficient (persons)")
    ax.set_title("Small-world check (your network as a dot)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.text(avg_path, clustering + 0.02, f"{avg_path:.2f}, {clustering:.2f}", ha="center", fontsize=9)

    fig.tight_layout()
    scatter_path = PLOTS_DIR / "small_world_scatter.png"
    fig.savefig(scatter_path, dpi=200)
    plt.close(fig)
    print(f"[ok] Saved {scatter_path}")


def plot_small_world_sample_subgraph(sample_persons: int = 60, sample_events: int = 40) -> None:
    """Draw a tiny force-directed sketch of the network (for intuition)."""
    data_raw = Path(__file__).resolve().parent / "data_raw"
    project_root = Path(__file__).resolve().parents[1]
    input_file = data_raw / "wiki_entities_with_weights.jsonl"
    if not input_file.exists():
        input_file = data_raw / "wiki_entities_with_links.jsonl"
        if not input_file.exists():
            input_file = project_root / "old_code" / "wiki_entities_with_links.jsonl"

    if not input_file.exists():
        print("[warn] No input graph file found; skipping small-world sketch.")
        return

    # First pass: collect candidates
    persons = []
    events = []
    with input_file.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            wp = obj.get("wikipedia", {})
            node_id = wp.get("id")
            cat = obj.get("category")
            if not isinstance(node_id, int) or cat not in ("person", "event"):
                continue
            if cat == "person":
                persons.append(node_id)
            else:
                events.append(node_id)

    random.shuffle(persons)
    random.shuffle(events)
    sel = set(persons[:sample_persons] + events[:sample_events])
    G = load_graph_subset(input_file, sel)
    if G.number_of_nodes() == 0:
        print("[info] Empty sampled subgraph; skipping small-world sketch.")
        return

    pos = nx.spring_layout(G, seed=7, k=0.45, iterations=80)
    colors = ["#4C72B0" if G.nodes[n].get("category") == "person" else "#55A868" for n in G.nodes]

    fig, ax = plt.subplots(figsize=(9, 7))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.35, width=1.0, edge_color="#888888")
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=55, ax=ax, linewidths=0.3, edgecolors="black")
    ax.set_title(
        f"Small-world sketch (sampled {G.number_of_nodes()} nodes, {G.number_of_edges()} edges)\n"
        "Blue=persons, Green=events"
    )
    ax.axis("off")
    fig.tight_layout()
    out_path = PLOTS_DIR / "small_world_force_sample.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[ok] Saved {out_path}")


def plot_small_world_force_component(max_nodes: int = 1500, fallback_sample: int = 1200) -> None:
    """Force layout of the largest connected component (or top-degree sample)."""
    data_raw = Path(__file__).resolve().parent / "data_raw"
    project_root = Path(__file__).resolve().parents[1]
    input_file = data_raw / "wiki_entities_with_weights.jsonl"
    if not input_file.exists():
        input_file = data_raw / "wiki_entities_with_links.jsonl"
        if not input_file.exists():
            input_file = project_root / "old_code" / "wiki_entities_with_links.jsonl"

    if not input_file.exists():
        print("[warn] No input graph file found; skipping full-component sketch.")
        return

    G = load_full_graph(input_file)
    if G.number_of_nodes() == 0:
        print("[info] Empty graph; skipping full-component sketch.")
        return

    # Largest connected component
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    if not components:
        print("[info] No connected components found.")
        return
    lcc_nodes = components[0]

    # If too large, keep top-degree nodes to maintain readability
    if len(lcc_nodes) > max_nodes:
        subG = G.subgraph(lcc_nodes).copy()
        degrees = sorted(subG.degree, key=lambda x: x[1], reverse=True)
        keep = {n for n, _ in degrees[:fallback_sample]}
        subG = subG.subgraph(keep).copy()
    else:
        subG = G.subgraph(lcc_nodes).copy()

    pos = nx.spring_layout(subG, seed=11, k=0.4, iterations=90)
    colors = ["#4C72B0" if subG.nodes[n].get("category") == "person" else "#55A868" for n in subG.nodes]

    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_edges(subG, pos, ax=ax, alpha=0.18, width=0.7, edge_color="#888888")
    nx.draw_networkx_nodes(subG, pos, node_color=colors, node_size=28, ax=ax, linewidths=0.2, edgecolors="black")
    ax.set_title(
        f"Largest connected component (persons=blue, events=green)\n"
        f"Nodes shown: {subG.number_of_nodes()} | Edges: {subG.number_of_edges()}"
    )
    ax.axis("off")
    fig.tight_layout()
    out_path = PLOTS_DIR / "small_world_force_component.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[ok] Saved {out_path}")


def plot_small_world_circle_component(max_nodes: int = None) -> None:
    """Circular layout sketch of the largest connected component (a la classic small-world diagrams).
    max_nodes=None => show all nodes in the largest component (can be dense).
    """
    data_raw = Path(__file__).resolve().parent / "data_raw"
    project_root = Path(__file__).resolve().parents[1]
    input_file = data_raw / "wiki_entities_with_weights.jsonl"
    if not input_file.exists():
        input_file = data_raw / "wiki_entities_with_links.jsonl"
        if not input_file.exists():
            input_file = project_root / "old_code" / "wiki_entities_with_links.jsonl"

    if not input_file.exists():
        print("[warn] No input graph file found; skipping circular layout.")
        return

    G = load_full_graph(input_file)
    if G.number_of_nodes() == 0:
        print("[info] Empty graph; skipping circular layout.")
        return

    components = sorted(nx.connected_components(G), key=len, reverse=True)
    if not components:
        print("[info] No connected components found.")
        return
    lcc_nodes = components[0]
    subG = G.subgraph(lcc_nodes).copy()

    # If caller sets a max cap, keep top-degree nodes; otherwise show all
    if max_nodes is not None and subG.number_of_nodes() > max_nodes:
        degrees = sorted(subG.degree, key=lambda x: x[1], reverse=True)
        keep = {n for n, _ in degrees[:max_nodes]}
        subG = subG.subgraph(keep).copy()

    pos = nx.circular_layout(subG)
    colors = ["#4C72B0" if subG.nodes[n].get("category") == "person" else "#55A868" for n in subG.nodes]

    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw_networkx_edges(subG, pos, ax=ax, alpha=0.08, width=0.5, edge_color="#999999")
    nx.draw_networkx_nodes(subG, pos, node_color=colors, node_size=28, ax=ax, linewidths=0.2, edgecolors="black")

    # Annotate a few highest-degree hubs (likely the dense spots)
    deg_sorted = sorted(subG.degree, key=lambda x: x[1], reverse=True)
    for node_id, _ in deg_sorted[:8]:
        x, y = pos[node_id]
        title = subG.nodes[node_id].get("title") or f"ID {node_id}"
        ax.text(
            x * 1.08,
            y * 1.08,
            title[:30],
            fontsize=8,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7),
        )
    ax.set_title(
        f"Circular view of largest component (persons=blue, events=green)\n"
        f"Nodes shown: {subG.number_of_nodes()} | Edges: {subG.number_of_edges()}"
    )
    ax.axis("off")
    fig.tight_layout()
    out_path = PLOTS_DIR / "small_world_circle_component.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[ok] Saved {out_path}")

    # Interactive HTML with Plotly for zoom/pan and toggle person-event vs person-projected edges
    if px and pio and go:
        coords = {n: pos[n] for n in subG.nodes}

        def edge_trace(graph, coord_map, color="#AAAAAA", name="Edges"):
            ex, ey = [], []
            for u, v in graph.edges():
                x0, y0 = coord_map[u]
                x1, y1 = coord_map[v]
                ex += [x0, x1, None]
                ey += [y0, y1, None]
            return go.Scatter(
                x=ex,
                y=ey,
                mode="lines",
                line=dict(color=color, width=0.5),
                hoverinfo="none",
                name=name,
                showlegend=False,
            )

        def node_trace(graph, coord_map, name="Nodes"):
            cats = []
            titles = []
            xs = []
            ys = []
            for n in graph.nodes():
                xs.append(coord_map[n][0])
                ys.append(coord_map[n][1])
                cats.append("Person" if graph.nodes[n].get("category") == "person" else "Event")
                titles.append(graph.nodes[n].get("title", ""))
            return go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(
                    size=6,
                    color=["#4C72B0" if c == "Person" else "#55A868" for c in cats],
                    line=dict(color="black", width=0.4),
                ),
                text=titles,
                hoverinfo="text",
                name=name,
                showlegend=False,
            )

        # Person-only projection: connect persons that share an event
        persons = [n for n in subG.nodes if subG.nodes[n].get("category") == "person"]
        events = [n for n in subG.nodes if subG.nodes[n].get("category") == "event"]
        proj_edges = set()
        for ev in events:
            neigh = [p for p in subG.neighbors(ev) if subG.nodes[p].get("category") == "person"]
            if len(neigh) >= 2:
                for i in range(len(neigh)):
                    for j in range(i + 1, len(neigh)):
                        a, b = neigh[i], neigh[j]
                        if a != b:
                            proj_edges.add((min(a, b), max(a, b)))
        G_person = nx.Graph()
        G_person.add_nodes_from(persons)
        G_person.add_edges_from(proj_edges)

        # Use same circular coordinates, but only for persons in projected graph
        person_coords = {n: coords[n] for n in G_person.nodes if n in coords}

        traces_original = [
            edge_trace(subG, coords, color="#CCCCCC", name="Edges"),
            node_trace(subG, coords, name="Nodes"),
        ]
        traces_person = [
            edge_trace(G_person, person_coords, color="#FF7F0E", name="Person-Person"),
            node_trace(G_person, person_coords, name="Persons"),
        ]

        fig_plotly = go.Figure(data=traces_original + traces_person)
        # Hide person-projected traces by default
        vis = [True, True] + [False, False]
        fig_plotly.update_traces(visible=False)
        for i, v in enumerate(vis):
            fig_plotly.data[i].visible = v

        fig_plotly.update_layout(
            title="Circular view (toggle modes)",
            width=900,
            height=900,
            xaxis_visible=False,
            yaxis_visible=False,
            margin=dict(l=0, r=0, t=50, b=0),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=0.5,
                    y=1.08,
                    buttons=[
                        dict(
                            label="Person-Event",
                            method="update",
                            args=[{"visible": [True, True, False, False]}],
                        ),
                        dict(
                            label="Person-Projection",
                            method="update",
                            args=[{"visible": [False, False, True, True]}],
                        ),
                    ],
                )
            ],
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        html_out = PLOTS_DIR / "small_world_circle_component.html"
        pio.write_html(fig_plotly, file=html_out, include_plotlyjs="cdn")
        print(f"[ok] Saved {html_out}")
    else:
        print("[info] Plotly not installed; skipping circular HTML (install plotly to enable).")


def plot_small_world_circle_person_projection(max_edges_per_node: int = 4000) -> None:
    """Person-person circular layout using projection (share-an-event)."""
    data_raw = Path(__file__).resolve().parent / "data_raw"
    project_root = Path(__file__).resolve().parents[1]
    input_file = data_raw / "wiki_entities_with_weights.jsonl"
    if not input_file.exists():
        input_file = data_raw / "wiki_entities_with_links.jsonl"
        if not input_file.exists():
            input_file = project_root / "old_code" / "wiki_entities_with_links.jsonl"

    if not input_file.exists():
        print("[warn] No input graph file found; skipping person-projection circle.")
        return

    # Build bipartite to derive projection
    persons = set()
    events = set()
    person_to_events = {}
    with input_file.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            wp = obj.get("wikipedia", {})
            node_id = wp.get("id")
            cat = obj.get("category")
            if not isinstance(node_id, int):
                continue
            if cat == "person":
                persons.add(node_id)
                links = obj.get("link_weights", [])
                evs = []
                for lk in links:
                    tgt = lk.get("target_id")
                    if isinstance(tgt, int):
                        evs.append(tgt)
                        events.add(tgt)
                person_to_events[node_id] = evs

    # Build weighted projection: persons connected if they share an event
    edge_weights = Counter()
    for p, evs in person_to_events.items():
        evs_unique = list(set(evs))
        if len(evs_unique) < 1:
            continue
        for i in range(len(evs_unique)):
            for j in range(i + 1, len(evs_unique)):
                pass  # just dedup list; weight handled below

    # Efficient event->persons map to accumulate edges
    event_to_persons = defaultdict(list)
    for p, evs in person_to_events.items():
        for ev in evs:
            event_to_persons[ev].append(p)

    for ev, plist in event_to_persons.items():
        if len(plist) < 2:
            continue
        uniq = list(set(plist))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                a, b = uniq[i], uniq[j]
                if a == b:
                    continue
                edge = (min(a, b), max(a, b))
                edge_weights[edge] += 1

    # Build projection graph
    Gp = nx.Graph()
    for p in persons:
        Gp.add_node(p, category="person")
    for (a, b), w in edge_weights.items():
        Gp.add_edge(a, b, weight=w)

    # Keep largest component of projection
    comps = sorted(nx.connected_components(Gp), key=len, reverse=True)
    if not comps:
        print("[info] No projection component; skipping.")
        return
    lcc = comps[0]
    Gp = Gp.subgraph(lcc).copy()

    # Optionally limit edges per node to avoid hairball
    if max_edges_per_node is not None:
        trimmed = nx.Graph()
        trimmed.add_nodes_from(Gp.nodes(data=True))
        for n in Gp.nodes():
            nbrs = sorted(Gp[n].items(), key=lambda x: x[1].get("weight", 1), reverse=True)[:max_edges_per_node]
            for m, attr in nbrs:
                if not trimmed.has_edge(n, m):
                    trimmed.add_edge(n, m, **attr)
        Gp = trimmed

    pos = nx.circular_layout(Gp)
    colors = ["#4C72B0" for _ in Gp.nodes]

    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw_networkx_edges(Gp, pos, ax=ax, alpha=0.08, width=0.4, edge_color="#999999")
    nx.draw_networkx_nodes(Gp, pos, node_color=colors, node_size=10, ax=ax, linewidths=0, edgecolors="none")
    ax.set_title(
        f"Person–person projection (circular) — nodes: {Gp.number_of_nodes()}, edges: {Gp.number_of_edges()}"
    )
    ax.axis("off")
    fig.tight_layout()
    out_path = PLOTS_DIR / "small_world_circle_person_projection.png"
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    print(f"[ok] Saved {out_path}")


def plot_pagerank(top_n: int = 20) -> None:
    data = load_json(DATA_ANALYSIS / "pagerank_scores.json")
    top_persons: List[Dict] = data.get("top_persons", [])[:top_n]
    top_events: List[Dict] = data.get("top_events", [])[:top_n]

    if not top_persons and not top_events:
        print("[info] Skip PageRank plots (no data).")
        return

    def _barplot(items: List[Dict], title: str, filename: str) -> None:
        if not items:
            print(f"[info] Skip {title} (no items).")
            return
        names = [i.get("title", f"id {i.get('id')}") or f"id {i.get('id')}" for i in items]
        scores = [i.get("pagerank", 0) for i in items]

        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(items))))
        ax.barh(range(len(items)), scores, color="#4C72B0")
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("PageRank score")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        out_path = PLOTS_DIR / filename
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[ok] Saved {out_path}")

    _barplot(top_persons, f"Top {len(top_persons)} persons by PageRank", "pagerank_top_persons.png")
    _barplot(top_events, f"Top {len(top_events)} events by PageRank", "pagerank_top_events.png")


def _representative_label(comp: Dict, max_persons: int = 2) -> Tuple[str, str]:
    """Return (short_label, long_label) using event/person titles."""
    sample_nodes = comp.get("sample_nodes") or []
    persons = [n.get("title", "") for n in sample_nodes if n.get("category") == "person" and n.get("title")]
    events = [n.get("title", "") for n in sample_nodes if n.get("category") == "event" and n.get("title")]

    main_event = events[0] if events else ""
    main_persons = persons[:max_persons]

    if main_event and main_persons:
        short = f"{main_event}: " + ", ".join(main_persons)
    elif main_event:
        short = f"{main_event}"
    elif main_persons:
        short = ", ".join(main_persons)
    else:
        short = "Mixed community"

    long_label = short
    return short, long_label


def plot_communities(top_n: int = 10) -> None:
    data = load_json(DATA_ANALYSIS / "communities.json")
    communities = data.get("communities", {})
    partition = data.get("partition", {})
    if not communities:
        print("[info] Skip community plots (no data).")
        return

    def _community_label(comm_id: str, comp: Dict) -> str:
        """Human-friendly label using event anchor + top persons."""
        short, _ = _representative_label(comp)
        person_pct = comp.get("person_ratio", 0)
        return f"{short} ({person_pct:.0%} persons)"

    enriched = []
    label_by_cid = {}
    for cid, comp in communities.items():
        short, long_label = _representative_label(comp)
        label_by_cid[cid] = short
        enriched.append(
            {
                "cid": cid,
                "size": comp.get("size", 0),
                "persons": comp.get("persons", 0),
                "events": comp.get("events", 0),
                "person_ratio": comp.get("person_ratio", 0.0),
                "label_short": short,
                "label_long": long_label,
                "comp": comp,
            }
        )

    enriched = sorted(enriched, key=lambda x: x["size"], reverse=True)
    sizes = [comp.get("size", 0) for comp in communities.values()]
    top_items = enriched[:top_n]

    # Histogram of all community sizes
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sizes, bins=min(30, max(5, len(sizes) // 5)), color="#55A868", edgecolor="white")
    ax.set_title("Community size distribution")
    ax.set_xlabel("Community size")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_hist = PLOTS_DIR / "community_size_distribution.png"
    fig.savefig(out_hist, dpi=200)
    plt.close(fig)
    print(f"[ok] Saved {out_hist}")

    # Bar chart of top communities
    labels = [_community_label(item["cid"], item["comp"]) for item in top_items]
    top_sizes = [item["size"] for item in top_items]
    fig, ax = plt.subplots(figsize=(10, max(5, 0.45 * len(top_items))))
    ax.barh(range(len(top_items)), top_sizes, color="#C44E52")
    ax.set_yticks(range(len(top_items)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Size (#nodes)")
    ax.set_title(f"Largest communities (top {len(top_items)})")
    ax.grid(axis="x", alpha=0.3)
    # Annotate counts inside bars for students to read easily.
    for idx, item in enumerate(top_items):
        size = item["size"]
        persons = item.get("persons", 0)
        events = item.get("events", 0)
        ax.text(
            size * 0.98,
            idx,
            f"{size} nodes | {persons} persons, {events} events",
            ha="right",
            va="center",
            fontsize=9,
            color="white",
            weight="bold",
        )
    fig.tight_layout()
    out_top = PLOTS_DIR / "community_top_sizes.png"
    fig.savefig(out_top, dpi=200)
    plt.close(fig)
    print(f"[ok] Saved {out_top}")

    # Bubble plot: each community as a dot (size = community size, color = person %)
    comm_points = top_items

    fig, ax = plt.subplots(figsize=(11, 8))
    sizes_scaled = [p["size"] for p in comm_points]
    max_size = max(sizes_scaled) if sizes_scaled else 1
    bubble_sizes = [140 + 520 * (s / max_size) for s in sizes_scaled]
    colors = [p["person_ratio"] for p in comm_points]
    palette = plt.get_cmap("plasma")
    scatter = ax.scatter(
        [p["persons"] for p in comm_points],
        [p["events"] for p in comm_points],
        s=bubble_sizes,
        c=colors,
        cmap=palette,
        alpha=0.85,
        edgecolor="black",
    )
    ax.set_xlabel("# Persons in community")
    ax.set_ylabel("# Events in community")
    ax.set_title("Communities as bubbles (size = nodes, color = person ratio)")
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(scatter, ax=ax, label="Person ratio")

    for i, p in enumerate(comm_points):
        jitter_x = 0.12 * ((i % 4) - 1.5)
        jitter_y = 0.12 * (((i // 4) % 4) - 1.5)
        label_wrapped = textwrap.fill(p["label_short"], width=18)
        ax.text(
            p["persons"] + jitter_x,
            p["events"] + jitter_y,
            label_wrapped,
            ha="center",
            va="center",
            fontsize=8.5,
            color="black",
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.8),
        )

    fig.tight_layout()
    bubble_out = PLOTS_DIR / "community_bubbles.png"
    fig.savefig(bubble_out, dpi=200)
    plt.close(fig)
    print(f"[ok] Saved {bubble_out}")

    # Interactive Plotly bubble chart to avoid label overlap (saved as HTML)
    if px and pio:
        df = []
        for item in top_items:
            df.append(
                {
                    "community": item["label_short"],
                    "persons": item["persons"],
                    "events": item["events"],
                    "size": item["size"],
                    "person_ratio": item["person_ratio"],
                }
            )
        fig_plotly = px.scatter(
            df,
            x="persons",
            y="events",
            size="size",
            color="person_ratio",
            hover_name="community",
            color_continuous_scale="Plasma",
            size_max=80,
            title="Communities (interactive bubble chart)",
            labels={"persons": "# Persons", "events": "# Events", "person_ratio": "Person ratio"},
        )
        fig_plotly.update_layout(width=900, height=650)
        html_out = PLOTS_DIR / "community_bubbles_interactive.html"
        pio.write_html(fig_plotly, file=html_out, include_plotlyjs="cdn")
        print(f"[ok] Saved {html_out}")
    else:
        print("[info] Plotly not installed; skipping interactive bubble chart. Install with: pip install plotly")

    # Export a small legend/summary for the top communities to help students.
    summary = [
        {
            "community_id": item["cid"],
            "label": item["label_short"],
            "size": item["size"],
            "persons": item["persons"],
            "events": item["events"],
            "person_ratio": item["person_ratio"],
            "sample_nodes": item["comp"].get("sample_nodes", [])[:6],
        }
        for item in top_items
    ]
    summary_path = PLOTS_DIR / "community_top_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[ok] Saved {summary_path}")

    # Export a small legend/summary for the top communities to help students.
    summary = [
        {
            "community_id": item["cid"],
            "label": item["label_short"],
            "size": item["size"],
            "persons": item["persons"],
            "events": item["events"],
            "person_ratio": item["person_ratio"],
            "sample_nodes": item["comp"].get("sample_nodes", [])[:6],
        }
        for item in top_items
    ]
    summary_path = PLOTS_DIR / "community_top_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[ok] Saved {summary_path}")

    # Node-level strip plot: show how nodes are assigned to top communities (dots by color)
    if partition:
        # Limit nodes per community for readability
        max_nodes_per_comm = 150
        # Map community id -> list of node ids
        comm_to_nodes = {}
        for node_id_str, cid in partition.items():
            comm_to_nodes.setdefault(str(cid), []).append(int(node_id_str))

        # Prepare points for top communities only
        points = []
        for rank, item in enumerate(top_items):
            cid = item["cid"]
            nodes = comm_to_nodes.get(cid, [])
            random.shuffle(nodes)
            nodes = nodes[:max_nodes_per_comm]
            for n in nodes:
                points.append((rank, n, cid))

        if points:
            # Simple jittered strip per community
            fig, ax = plt.subplots(figsize=(11, 5))
            xs = []
            ys = []
            colors = []
            for x_base, _, cid in points:
                xs.append(x_base + random.uniform(-0.25, 0.25))
                ys.append(random.uniform(0, 1))
                colors.append(cid)

            scatter2 = ax.scatter(xs, ys, c=[hash(c) % 20 for c in colors], cmap="tab20", alpha=0.6, edgecolor="none", s=18)
            ax.set_yticks([])
            ax.set_xlabel("Communities (top by size)")
            ax.set_title("Louvain assignments (dots = nodes, color = community)")
            ax.set_xlim(-0.6, len(top_items) - 0.4)
            ax.set_xticks(range(len(top_items)))
            ax.set_xticklabels([label_by_cid[item["cid"]][:20] for item in top_items], rotation=15, ha="right")
            fig.tight_layout()
            strip_out = PLOTS_DIR / "community_node_strip.png"
            fig.savefig(strip_out, dpi=200)
            plt.close(fig)
            print(f"[ok] Saved {strip_out}")


def plot_force_layout_top_community(max_nodes: int = 120) -> None:
    """Force-directed sketch of the largest community (students-friendly)."""
    data = load_json(DATA_ANALYSIS / "communities.json")
    communities = data.get("communities", {})
    partition = data.get("partition", {})
    if not communities or not partition:
        print("[info] Skip force layout (no communities or partition).")
        return

    # Pick largest community
    largest_cid, largest_comp = max(communities.items(), key=lambda kv: kv[1].get("size", 0))
    node_ids = [int(nid) for nid, cid in partition.items() if str(cid) == largest_cid]
    if not node_ids:
        print("[info] No nodes for largest community.")
        return

    # Sample nodes to keep the plot readable
    random.shuffle(node_ids)
    node_ids = node_ids[:max_nodes]
    node_set = set(node_ids)

    # Locate input graph
    data_raw = Path(__file__).resolve().parent / "data_raw"
    project_root = Path(__file__).resolve().parents[1]
    input_file = data_raw / "wiki_entities_with_weights.jsonl"
    if not input_file.exists():
        input_file = data_raw / "wiki_entities_with_links.jsonl"
        if not input_file.exists():
            input_file = project_root / "old_code" / "wiki_entities_with_links.jsonl"

    if not input_file.exists():
        print("[warn] No input graph file found; skipping force layout.")
        return

    G = load_graph_subset(input_file, node_set)
    if G.number_of_nodes() == 0:
        print("[info] Subgraph empty; skipping force layout.")
        return

    pos = nx.spring_layout(G, seed=42, k=0.35)
    colors = ["#4C72B0" if G.nodes[n].get("category") == "person" else "#55A868" for n in G.nodes]

    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.12, width=0.6, edge_color="#999999")
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=52, ax=ax, linewidths=0.2, edgecolors="black")
    ax.set_title(
        f"Largest community sketch (C{largest_cid}) — persons=blue, events=green\n"
        f"Sampled {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )
    ax.axis("off")
    fig.tight_layout()
    out_path = PLOTS_DIR / "community_force_layout.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[ok] Saved {out_path}")


def main() -> None:
    ensure_output_dir()
    plot_small_world()
    plot_small_world_sample_subgraph()
    plot_small_world_force_component()
    plot_small_world_circle_component()
    plot_small_world_circle_person_projection()
    plot_pagerank()
    plot_communities()
    print("\nDone. Charts are in:", PLOTS_DIR)


if __name__ == "__main__":
    main()

