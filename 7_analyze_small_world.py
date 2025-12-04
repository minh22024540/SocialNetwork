#!/usr/bin/env python3
"""
Analyze small world properties of the person-event network.

This module calculates small world metrics BETWEEN PEOPLE ONLY, where events
act as intermediate nodes in paths. The graph is undirected and weighted.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import networkx as nx
from tqdm import tqdm


def load_graph_from_jsonl(jsonl_path: Path) -> nx.Graph:
    """Load graph from JSONL file.

    Reads entity data from JSONL and constructs an undirected weighted graph.

    Args:
        jsonl_path: Path to wiki_entities_with_weights.jsonl file.

    Returns:
        NetworkX undirected weighted graph with nodes (entity IDs) and
        weighted edges from link_weights field.
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
            
            if not isinstance(entity_id, int):
                continue
            
            # Add node with category
            G.add_node(entity_id, category=category)
            
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


def calculate_person_to_person_paths(
    G: nx.Graph,
    person_nodes: List[int],
    max_paths: int = 10000
) -> Tuple[float, Dict]:
    """Calculate average shortest path length between person nodes.

    Computes shortest paths between person nodes, allowing paths to traverse
    through event nodes as intermediates. Uses weighted shortest paths.

    Args:
        G: NetworkX undirected weighted graph.
        person_nodes: List of person node IDs to analyze.
        max_paths: Maximum number of path pairs to sample. Defaults to 10000.

    Returns:
        Tuple containing:
        - average_path_length: Mean shortest path length between person pairs
        - statistics_dict: Dictionary with path length distribution stats
    """
    print(f"Calculating person-to-person shortest paths...")
    print(f"  Person nodes: {len(person_nodes):,}")
    print(f"  Total nodes: {G.number_of_nodes():,}")
    
    # Filter to person nodes only for endpoints
    person_set = set(person_nodes)
    
    # Calculate all pairs shortest paths (weighted)
    # Use Dijkstra's algorithm with weights
    path_lengths = []
    path_count = 0
    max_pairs = min(max_paths, len(person_nodes) * (len(person_nodes) - 1) // 2)
    
    print(f"  Sampling up to {max_pairs:,} person pairs...")
    
    # Sample pairs if too many
    import random
    if len(person_nodes) > 1:
        if max_pairs < len(person_nodes) * (len(person_nodes) - 1) // 2:
            # Sample random pairs
            pairs = []
            attempts = 0
            while len(pairs) < max_pairs and attempts < max_pairs * 10:
                p1 = random.choice(person_nodes)
                p2 = random.choice(person_nodes)
                if p1 != p2 and (p1, p2) not in pairs and (p2, p1) not in pairs:
                    pairs.append((p1, p2))
                attempts += 1
        else:
            # Use all pairs
            pairs = [(p1, p2) for i, p1 in enumerate(person_nodes) 
                    for p2 in person_nodes[i+1:]]
    else:
        pairs = []
    
    # Calculate shortest paths
    for p1, p2 in tqdm(pairs, desc="Computing paths", unit="pair"):
        try:
            # Use weighted shortest path (events can be intermediate nodes)
            path_length = nx.shortest_path_length(
                G, p1, p2, weight="weight", method="dijkstra"
            )
            path_lengths.append(path_length)
            path_count += 1
        except nx.NetworkXNoPath:
            # No path exists between these persons
            continue
        except Exception as e:
            print(f"Error computing path ({p1}, {p2}): {e}")
            continue
    
    if not path_lengths:
        return float('inf'), {
            "average_path_length": float('inf'),
            "path_count": 0,
            "connected_pairs": 0,
            "disconnected_pairs": len(pairs)
        }
    
    avg_length = sum(path_lengths) / len(path_lengths)
    
    return avg_length, {
        "average_path_length": avg_length,
        "path_count": path_count,
        "connected_pairs": len(path_lengths),
        "disconnected_pairs": len(pairs) - len(path_lengths),
        "min_path_length": min(path_lengths),
        "max_path_length": max(path_lengths),
        "median_path_length": sorted(path_lengths)[len(path_lengths) // 2]
    }


def calculate_clustering_coefficient(G: nx.Graph, person_nodes: List[int]) -> float:
    """Calculate clustering coefficient for person nodes.
    
    Events are considered as intermediaries, so we calculate clustering
    based on how many person neighbors are connected to each other.
    
    Args:
        G: NetworkX graph (undirected, weighted).
        person_nodes: List of person node IDs.
        
    Returns:
        Average clustering coefficient.
    """
    print("Calculating clustering coefficient for person nodes...")
    
    person_set = set(person_nodes)
    clustering_coeffs = []
    
    for person_id in tqdm(person_nodes, desc="Computing clustering", unit="person"):
        # Get all neighbors (can be persons or events)
        neighbors = list(G.neighbors(person_id))
        
        if len(neighbors) < 2:
            # Need at least 2 neighbors for clustering
            continue
        
        # Count connections between neighbors
        # Only count if both neighbors are persons (person-person clustering)
        connections = 0
        possible_connections = 0
        
        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i+1:]:
                # Only count person-person connections
                if n1 in person_set and n2 in person_set:
                    possible_connections += 1
                    if G.has_edge(n1, n2):
                        connections += 1
        
        if possible_connections > 0:
            coeff = connections / possible_connections
            clustering_coeffs.append(coeff)
    
    if not clustering_coeffs:
        return 0.0
    
    return sum(clustering_coeffs) / len(clustering_coeffs)


def main():
    """Main function to analyze small world properties."""
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
    output_file = data_analysis / "small_world_analysis.json"
    
    print("=" * 80)
    print("Small World Analysis")
    print("=" * 80)
    print("Note: Calculating metrics BETWEEN PEOPLE ONLY")
    print("Events act as intermediate nodes in paths")
    print()
    
    try:
        # Load graph
        G = load_graph_from_jsonl(input_file)
        
        print(f"\nGraph loaded:")
        print(f"  Total nodes: {G.number_of_nodes():,}")
        print(f"  Total edges: {G.number_of_edges():,}")
        
        # Identify person nodes
        person_nodes = [
            node for node, data in G.nodes(data=True)
            if data.get("category") == "person"
        ]
        
        event_nodes = [
            node for node, data in G.nodes(data=True)
            if data.get("category") == "event"
        ]
        
        print(f"  Person nodes: {len(person_nodes):,}")
        print(f"  Event nodes: {len(event_nodes):,}")
        
        if len(person_nodes) < 2:
            print("Error: Need at least 2 person nodes for small world analysis")
            return
        
        # Calculate average shortest path length (person-to-person)
        avg_path_length, path_stats = calculate_person_to_person_paths(G, person_nodes)
        
        # Calculate clustering coefficient
        clustering = calculate_clustering_coefficient(G, person_nodes)
        
        # Compile results
        results = {
            "graph_statistics": {
                "total_nodes": G.number_of_nodes(),
                "total_edges": G.number_of_edges(),
                "person_nodes": len(person_nodes),
                "event_nodes": len(event_nodes)
            },
            "small_world_metrics": {
                "average_shortest_path_length": avg_path_length,
                "characteristic_path_length": avg_path_length,
                "clustering_coefficient": clustering,
                "path_statistics": path_stats
            },
            "small_world_property": {
                "is_small_world": avg_path_length <= 6.0 and clustering > 0.1,
                "average_path_length": avg_path_length,
                "clustering_coefficient": clustering,
                "interpretation": (
                    "Small world property holds" if avg_path_length <= 6.0 and clustering > 0.1
                    else "Small world property may not hold"
                )
            }
        }
        
        # Save results
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 80)
        print("Results")
        print("=" * 80)
        print(f"Average shortest path length (person-to-person): {avg_path_length:.4f}")
        print(f"Clustering coefficient: {clustering:.4f}")
        print(f"Connected person pairs: {path_stats['connected_pairs']:,}")
        print(f"Disconnected person pairs: {path_stats['disconnected_pairs']:,}")
        print(f"\nSmall world property: {results['small_world_property']['interpretation']}")
        print(f"  (Average path length <= 6.0: {avg_path_length <= 6.0})")
        print(f"  (Clustering > 0.1: {clustering > 0.1})")
        print(f"\nResults saved to: {output_file}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error analyzing small world: {e}")
        raise


if __name__ == "__main__":
    main()

