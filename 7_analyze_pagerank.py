#!/usr/bin/env python3
"""
Calculate PageRank scores for all nodes in the graph.

This module computes PageRank to identify the most important nodes
(persons and events) in the network.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import networkx as nx
from tqdm import tqdm


def load_graph_from_jsonl(jsonl_path: Path) -> nx.Graph:
    """Load graph from JSONL file.

    Reads entity data from JSONL and constructs an undirected weighted graph
    where nodes are entities and edges represent links between them.

    Args:
        jsonl_path: Path to wiki_entities_with_weights.jsonl file.

    Returns:
        NetworkX undirected weighted graph with:
        - Nodes: entity IDs with attributes (category, title)
        - Edges: weighted links between entities from link_weights field
    """
    G = nx.Graph()
    
    print("Loading graph from JSONL...")
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading graph", unit="entity"):
            line = line.strip()
            if not line:
                continue
            
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            wp = obj.get("wikipedia", {})
            entity_id = wp.get("id")
            category = obj.get("category")
            title = wp.get("title", "")
            
            if not isinstance(entity_id, int):
                continue
            
            # Add node with category and title
            G.add_node(entity_id, category=category, title=title)
            
            # Add edges with weights
            link_weights = obj.get("link_weights", [])
            for link in link_weights:
                target_id = link.get("target_id")
                weight = link.get("weight", 1.0)
                
                if isinstance(target_id, int):
                    # Add edge with weight (undirected)
                    if not G.has_edge(entity_id, target_id):
                        G.add_edge(entity_id, target_id, weight=weight)
    
    return G


def calculate_pagerank(
    G: nx.Graph, alpha: float = 0.85, max_iter: int = 100
) -> Dict[int, float]:
    """Calculate PageRank scores for all nodes.

    Computes weighted PageRank on a directed version of the input graph
    (converted from undirected by making edges bidirectional).

    Args:
        G: NetworkX undirected weighted graph.
        alpha: Damping parameter. Defaults to 0.85.
        max_iter: Maximum iterations for convergence. Defaults to 100.

    Returns:
        Dictionary mapping node ID (int) to PageRank score (float).
        Scores sum to approximately 1.0 across all nodes.
    """
    print("Calculating PageRank...")
    
    # Convert to directed graph for PageRank (make bidirectional)
    G_directed = G.to_directed()
    
    # Calculate weighted PageRank
    pagerank_scores = nx.pagerank(
        G_directed,
        alpha=alpha,
        max_iter=max_iter,
        weight="weight"
    )
    
    return pagerank_scores


def get_top_nodes(
    pagerank_scores: Dict[int, float],
    G: nx.Graph,
    top_n: int = 50
) -> List[Dict]:
    """Get top-ranked nodes with metadata.

    Sorts nodes by PageRank score and returns top N with their attributes.

    Args:
        pagerank_scores: Dictionary mapping node ID to PageRank score.
        G: NetworkX graph containing node attributes.
        top_n: Number of top nodes to return. Defaults to 50.

    Returns:
        List of dictionaries, each containing:
        - id: Node ID
        - title: Entity title
        - category: Entity category (person, event, etc.)
        - pagerank: PageRank score
        - degree: Node degree in the graph
    """
    # Sort by score
    sorted_nodes = sorted(
        pagerank_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    
    top_nodes = []
    for node_id, score in sorted_nodes:
        node_data = G.nodes[node_id]
        top_nodes.append({
            "id": node_id,
            "title": node_data.get("title", ""),
            "category": node_data.get("category", "unknown"),
            "pagerank": score,
            "degree": G.degree(node_id)
        })
    
    return top_nodes


def main() -> None:
    """Main function to calculate PageRank.

    Loads graph from JSONL, calculates PageRank scores, identifies top nodes,
    and saves results to pagerank_scores.json.
    """
    # Try main directory first, then old_code, using project-relative paths
    project_root = Path(__file__).resolve().parents[2]
    data_raw = Path(__file__).resolve().parent / "data_raw"
    data_analysis = Path(__file__).resolve().parent / "data_analysis"

    input_file = data_raw / "wiki_entities_with_weights.jsonl"
    if not input_file.exists():
        # Fallback to original file if weights file doesn't exist
        input_file = data_raw / "wiki_entities_with_links.jsonl"
        if not input_file.exists():
            input_file = project_root / "old_code" / "wiki_entities_with_links.jsonl"
    output_file = data_analysis / "pagerank_scores.json"
    
    print("=" * 80)
    print("PageRank Analysis")
    print("=" * 80)
    
    try:
        # Load graph
        G = load_graph_from_jsonl(input_file)
        
        print(f"\nGraph loaded:")
        print(f"  Total nodes: {G.number_of_nodes():,}")
        print(f"  Total edges: {G.number_of_edges():,}")
        
        if G.number_of_nodes() == 0:
            print("Error: Empty graph")
            return
        
        # Calculate PageRank
        pagerank_scores = calculate_pagerank(G)
        
        # Get top nodes
        top_persons = get_top_nodes(
            {n: s for n, s in pagerank_scores.items() 
             if G.nodes[n].get("category") == "person"},
            G,
            top_n=50
        )
        
        top_events = get_top_nodes(
            {n: s for n, s in pagerank_scores.items() 
             if G.nodes[n].get("category") == "event"},
            G,
            top_n=50
        )
        
        # Compile results
        results = {
            "graph_statistics": {
                "total_nodes": G.number_of_nodes(),
                "total_edges": G.number_of_edges(),
                "nodes_with_pagerank": len(pagerank_scores)
            },
            "pagerank_statistics": {
                "min_score": min(pagerank_scores.values()) if pagerank_scores else 0,
                "max_score": max(pagerank_scores.values()) if pagerank_scores else 0,
                "mean_score": sum(pagerank_scores.values()) / len(pagerank_scores) if pagerank_scores else 0
            },
            "top_persons": top_persons,
            "top_events": top_events,
            "all_scores": {
                str(node_id): score
                for node_id, score in sorted(
                    pagerank_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            }
        }
        
        # Save results
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 80)
        print("Results")
        print("=" * 80)
        print(f"PageRank scores calculated for {len(pagerank_scores):,} nodes")
        print(f"Score range: {results['pagerank_statistics']['min_score']:.6f} - {results['pagerank_statistics']['max_score']:.6f}")
        print(f"Mean score: {results['pagerank_statistics']['mean_score']:.6f}")
        
        print(f"\nTop 10 Persons by PageRank:")
        for i, person in enumerate(top_persons[:10], 1):
            print(f"  {i}. {person['title']} (ID: {person['id']}, Score: {person['pagerank']:.6f}, Degree: {person['degree']})")
        
        print(f"\nTop 10 Events by PageRank:")
        for i, event in enumerate(top_events[:10], 1):
            print(f"  {i}. {event['title']} (ID: {event['id']}, Score: {event['pagerank']:.6f}, Degree: {event['degree']})")
        
        print(f"\nResults saved to: {output_file}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error calculating PageRank: {e}")
        raise


if __name__ == "__main__":
    main()

