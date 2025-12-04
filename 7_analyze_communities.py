#!/usr/bin/env python3
"""
Detect communities in the person-event network using Louvain algorithm.

This module identifies communities (clusters) of closely connected nodes
and analyzes their composition.
"""

import json
from pathlib import Path
from typing import Dict, List, Set
import networkx as nx
from tqdm import tqdm

try:
    import community.community_louvain as community_louvain
except ImportError:
    print("Warning: python-louvain not installed. Install with: pip install python-louvain")
    print("Falling back to NetworkX greedy modularity communities")
    community_louvain = None


def load_graph_from_jsonl(jsonl_path: Path) -> nx.Graph:
    """Load graph from JSONL file.

    Reads entity data from JSONL and constructs an undirected weighted graph.

    Args:
        jsonl_path: Path to wiki_entities_with_weights.jsonl file.

    Returns:
        NetworkX undirected weighted graph with nodes (entity IDs) containing
        category and title attributes, and weighted edges.
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


def detect_communities_louvain(G: nx.Graph) -> Dict[int, int]:
    """Detect communities using Louvain algorithm.

    Uses the Louvain method for community detection, which optimizes modularity.
    Falls back to NetworkX greedy modularity if python-louvain not available.

    Args:
        G: NetworkX undirected weighted graph.

    Returns:
        Dictionary mapping node ID (int) to community ID (int).
        Nodes in the same community share the same community ID.
    """
    print("Detecting communities using Louvain algorithm...")
    
    if community_louvain is not None:
        # Use python-louvain library
        partition = community_louvain.best_partition(G, weight="weight")
        return partition
    else:
        # Fallback to NetworkX greedy modularity communities
        print("Using NetworkX greedy modularity communities (fallback)...")
        communities_generator = nx.community.greedy_modularity_communities(
            G, weight="weight"
        )
        
        # Convert to node -> community mapping
        partition = {}
        for comm_id, community in enumerate(communities_generator):
            for node in community:
                partition[node] = comm_id
        
        return partition


def analyze_community_composition(
    G: nx.Graph,
    partition: Dict[int, int]
) -> Dict[int, Dict]:
    """Analyze composition of each community.
    
    Args:
        G: NetworkX graph.
        partition: Dictionary mapping node ID to community ID.
        
    Returns:
        Dictionary mapping community ID to composition statistics.
    """
    print("Analyzing community composition...")
    
    communities: Dict[int, List[int]] = {}
    for node_id, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(node_id)
    
    composition = {}
    for comm_id, nodes in communities.items():
        persons = [n for n in nodes if G.nodes[n].get("category") == "person"]
        events = [n for n in nodes if G.nodes[n].get("category") == "event"]
        
        composition[comm_id] = {
            "size": len(nodes),
            "persons": len(persons),
            "events": len(events),
            "person_ratio": len(persons) / len(nodes) if nodes else 0,
            "event_ratio": len(events) / len(nodes) if nodes else 0,
            "node_ids": nodes[:20]  # Store first 20 node IDs as sample
        }
    
    return composition


def calculate_modularity(G: nx.Graph, partition: Dict[int, int]) -> float:
    """Calculate modularity of the partition.
    
    Args:
        G: NetworkX graph.
        partition: Dictionary mapping node ID to community ID.
        
    Returns:
        Modularity score.
    """
    # Use NetworkX modularity (more reliable)
    communities_dict = {}
    for node, comm_id in partition.items():
        if comm_id not in communities_dict:
            communities_dict[comm_id] = set()
        communities_dict[comm_id].add(node)
    
    communities_list = list(communities_dict.values())
    return nx.community.modularity(G, communities_list, weight="weight")


def main():
    """Main function to detect communities."""
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
    output_file = data_analysis / "communities.json"
    
    print("=" * 80)
    print("Community Detection")
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
        
        # Detect communities
        partition = detect_communities_louvain(G)
        
        # Calculate modularity
        modularity = calculate_modularity(G, partition)
        
        # Analyze composition
        composition = analyze_community_composition(G, partition)
        
        # Get top nodes per community (for visualization)
        communities_nodes: Dict[int, List[int]] = {}
        for node_id, comm_id in partition.items():
            if comm_id not in communities_nodes:
                communities_nodes[comm_id] = []
            communities_nodes[comm_id].append(node_id)
        
        # Compile results
        results = {
            "graph_statistics": {
                "total_nodes": G.number_of_nodes(),
                "total_edges": G.number_of_edges(),
                "number_of_communities": len(set(partition.values()))
            },
            "community_statistics": {
                "modularity": modularity,
                "average_community_size": sum(len(nodes) for nodes in communities_nodes.values()) / len(communities_nodes) if communities_nodes else 0,
                "min_community_size": min(len(nodes) for nodes in communities_nodes.values()) if communities_nodes else 0,
                "max_community_size": max(len(nodes) for nodes in communities_nodes.values()) if communities_nodes else 0
            },
            "communities": {
                str(comm_id): {
                    "size": comp["size"],
                    "persons": comp["persons"],
                    "events": comp["events"],
                    "person_ratio": comp["person_ratio"],
                    "event_ratio": comp["event_ratio"],
                    "sample_nodes": [
                        {
                            "id": node_id,
                            "title": G.nodes[node_id].get("title", ""),
                            "category": G.nodes[node_id].get("category", "unknown")
                        }
                        for node_id in comp["node_ids"]
                    ]
                }
                for comm_id, comp in sorted(
                    composition.items(),
                    key=lambda x: x[1]["size"],
                    reverse=True
                )
            },
            "partition": {
                str(node_id): comm_id
                for node_id, comm_id in partition.items()
            }
        }
        
        # Save results
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 80)
        print("Results")
        print("=" * 80)
        print(f"Number of communities detected: {len(set(partition.values())):,}")
        print(f"Modularity score: {modularity:.4f}")
        print(f"Average community size: {results['community_statistics']['average_community_size']:.1f}")
        print(f"Community size range: {results['community_statistics']['min_community_size']} - {results['community_statistics']['max_community_size']}")
        
        print(f"\nTop 10 Largest Communities:")
        sorted_communities = sorted(
            composition.items(),
            key=lambda x: x[1]["size"],
            reverse=True
        )[:10]
        
        for i, (comm_id, comp) in enumerate(sorted_communities, 1):
            print(f"  {i}. Community {comm_id}: {comp['size']} nodes "
                  f"({comp['persons']} persons, {comp['events']} events, "
                  f"person ratio: {comp['person_ratio']:.2%})")
        
        print(f"\nResults saved to: {output_file}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error detecting communities: {e}")
        raise


if __name__ == "__main__":
    main()

