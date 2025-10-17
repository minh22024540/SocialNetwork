import requests
import json
import time
import networkx as nx

# === Wikipedia API setup ===
WIKI_API = "https://vi.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "MyWikiGraphBot/1.0 (contact: your_email@example.com)"}

# === Keyword filters ===
NODE_KEYWORDS = [
    "Lãnh tụ", "anh hùng dân tộc", "bộ trưởng", "thủ tướng", "Ủy viên",
    "Đảng viên", "Đại biểu quốc hội", "Vua", "Kháng chiến", "Võ Tướng",
    "nhà văn việt nam", "Nguyên thủ quốc gia", "Việt Nam"
]
EDGE_KEYWORDS = ["Chiến tranh", "Sự kiện", "Tổ chức"]

# === Helper: Fetch by pageid safely ===
def get_wiki_page_by_id(page_id: int):
    """Fetch metadata, categories, and links for a given Wikipedia page ID."""
    base_params = {
        "action": "query",
        "pageids": page_id,
        "prop": "info|extracts|categories|links",
        "inprop": "url",
        "explaintext": True,
        "cllimit": "max",
        "pllimit": "max",
        "format": "json"
    }

    categories, links = [], []
    cont = {}

    while True:
        params = base_params.copy()
        params.update(cont)
        try:
            r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"❌ Error fetching {page_id}: {e}")
            break

        pages = data.get("query", {}).get("pages", {})
        if not pages:
            break

        page_data = next(iter(pages.values()))
        if "missing" in page_data:
            break

        # Collect categories
        for c in page_data.get("categories", []):
            t = c.get("title")
            if t:
                categories.append(t.replace("Thể loại:", "").strip())

        # Collect links
        for l in page_data.get("links", []):
            if l.get("ns") == 0 and "title" in l:
                links.append(l["title"])

        if "continue" in data:
            cont = data["continue"]
            time.sleep(0.2)
        else:
            break

    return {
        "pageid": page_id,
        "title": page_data.get("title", ""),
        "url": page_data.get("fullurl", ""),
        "extract": page_data.get("extract", ""),
        "categories": list(set(categories)),
        "links": list(set(links))
    }


def get_pageid_by_title(title):
    """Find page ID for a given title."""
    params = {"action": "query", "titles": title, "format": "json"}
    try:
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        page_data = next(iter(pages.values()))
        if "missing" in page_data:
            return None
        return page_data.get("pageid")
    except:
        return None


def should_include_node(page):
    """Check if node qualifies based on categories."""
    cats = set(page.get("categories", []))
    return any(any(k.lower() in c.lower() for k in NODE_KEYWORDS) for c in cats)


def should_include_edge(page):
    """Check if the link qualifies as an event or organization."""
    cats = set(page.get("categories", []))
    return any(any(k.lower() in c.lower() for k in EDGE_KEYWORDS) for c in cats)


# === Graph expansion ===
def expand_graph(G, max_eligible=10):
    visited = set()

    for node, attrs in list(G.nodes(data=True)):
        pid = attrs.get("pageid")
        if not pid or pid in visited:
            continue
        visited.add(pid)

        print(f"🔍 Fetching node: {node} (pageid={pid})")
        page = get_wiki_page_by_id(pid)
        if not page:
            continue

        G.nodes[node].update({
            "categories": page["categories"],
            "url": page["url"],
            "extract": page["extract"][:500]
        })

        eligible_count = 0
        for link_title in page["links"]:
            if eligible_count >= max_eligible:
                print(f"✅ Reached {max_eligible} eligible links, stop expanding {node}.")
                break

            target_pid = get_pageid_by_title(link_title)
            if not target_pid:
                continue

            linked_page = get_wiki_page_by_id(target_pid)
            if not linked_page:
                continue

            if should_include_node(linked_page) or should_include_edge(linked_page):
                eligible_count += 1

                if link_title not in G:
                    G.add_node(
                        link_title,
                        pageid=target_pid,
                        title=linked_page["title"],
                        url=linked_page["url"],
                        categories=linked_page["categories"]
                    )

                rel = "wiki_link"
                if should_include_edge(linked_page):
                    rel = "liên quan sự kiện/chiến tranh/tổ chức"

                if not G.has_edge(node, link_title):
                    G.add_edge(node, link_title, relation=rel)
                    print(f" ➕ Edge: {node} → {link_title} [{rel}]")

            time.sleep(0.2)

    return G


# === Seed graph ===
G = nx.DiGraph()
seed_nodes = {
    "Hồ Chí Minh": 375318,
    "Lê Lợi": 41069,
    "Trần Hưng Đạo": 746,
    "Vua Hùng": 57358,
    "Đinh Bộ Lĩnh": 42184,
}

for name, pid in seed_nodes.items():
    G.add_node(name, pageid=pid, title=f"{name} – Wikipedia tiếng Việt")

# === Expand and save ===
print("🚀 Expanding the Wikipedia graph (1 layer, up to 10 eligible links per node)...\n")
G = expand_graph(G, max_eligible=10)

data = {
    "nodes": [{"id": n, **attrs} for n, attrs in G.nodes(data=True)],
    "edges": [{"source": u, "target": v, **attrs} for u, v, attrs in G.edges(data=True)]
}

with open("wiki_graph_vi_limited.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"\n💾 Saved {len(G.nodes())} nodes, {len(G.edges())} edges → wiki_graph_vi_limited.json")
