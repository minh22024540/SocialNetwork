#!/usr/bin/env python3
"""
Analyze small world properties of the person-event network.

This module implements the standard small-world property analysis following:
Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of 'small-world' networks.
Nature, 393(6684), 440-442. https://doi.org/10.1038/30918

Small-world networks are characterized by:
1. High clustering coefficient (C_actual >> C_random)
2. Short average path length (L_actual ≈ L_random)

We use the quantitative criteria from Humphries & Gurney (2008):
- L_actual / L_random ≤ 1.5 (path length similar to random)
- C_actual / C_random ≥ 2.0 (clustering much higher than random)

Reference:
Humphries, M. D., & Gurney, K. (2008). Network 'small-world-ness': a quantitative
method for determining canonical network equivalence. PloS one, 3(4), e0002051.

Algorithm:
1. Converts bipartite person-event graph to person-person projection
2. Calculates L_actual and C_actual on projection
3. Generates random graphs (Erdős–Rényi G(n,m)) with same nodes/edges
4. Calculates L_random and C_random on random graphs
5. Compares ratios to determine small-world property
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import networkx as nx
from tqdm import tqdm
from collections import defaultdict


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
            
            # Add node with category and date fields for temporal filtering
            node_attrs = {
                "category": category,
                "date_of_birth": obj.get("date_of_birth"),
                "date_of_death": obj.get("date_of_death"),
                "start_time": obj.get("start_time"),  # For events
                "end_time": obj.get("end_time")  # For events
            }
            G.add_node(entity_id, **node_attrs)
            
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


def extract_year_from_date(date_str: Optional[str]) -> Optional[int]:
    """Extract year from date string (YYYY-MM-DD format).
    
    Args:
        date_str: Date string in YYYY-MM-DD format or None.
        
    Returns:
        Year as integer, or None if date_str is None or invalid.
    """
    if not date_str:
        return None
    try:
        return int(date_str.split('-')[0])
    except (ValueError, AttributeError, IndexError):
        return None


def filter_nodes_by_time_range(
    G: nx.Graph,
    person_nodes: List[int],
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
) -> Tuple[List[int], List[int]]:
    """Filter persons and events by time range.
    
    Persons: included if they were alive during the time range
    (date_of_birth <= end_year AND date_of_death >= start_year)
    
    Events: included if they occurred during the time range
    (start_time <= end_year AND end_time >= start_year)
    
    Args:
        G: Graph with node attributes including date_of_birth, date_of_death, start_time, end_time.
        person_nodes: List of person node IDs.
        start_year: Start year of time range (inclusive). If None, no lower bound.
        end_year: End year of time range (inclusive). If None, no upper bound.
        
    Returns:
        Tuple containing (filtered_person_nodes, filtered_event_nodes).
    """
    person_set = set(person_nodes)
    filtered_persons = []
    filtered_events = []
    
    for node in G.nodes():
        node_data = G.nodes[node]
        category = node_data.get('category', '')
        
        if category == 'person' and node in person_set:
            birth_year = extract_year_from_date(node_data.get('date_of_birth'))
            death_year = extract_year_from_date(node_data.get('date_of_death'))
            
            # Person is included if:
            # - Was born before/at end_year (or no end_year specified)
            # - Died after/at start_year (or no start_year specified, or still alive)
            include = True
            if start_year is not None:
                if death_year is not None and death_year < start_year:
                    include = False  # Died before range starts
            if end_year is not None:
                if birth_year is not None and birth_year > end_year:
                    include = False  # Born after range ends
            
            if include:
                filtered_persons.append(node)
        
        elif category == 'event':
            event_start = extract_year_from_date(node_data.get('start_time'))
            event_end = extract_year_from_date(node_data.get('end_time'))
            
            # Event is included if it overlaps with time range
            include = True
            if start_year is not None:
                if event_end is not None and event_end < start_year:
                    include = False  # Event ended before range starts
            if end_year is not None:
                if event_start is not None and event_start > end_year:
                    include = False  # Event started after range ends
            
            if include:
                filtered_events.append(node)
    
    return filtered_persons, filtered_events


def create_person_person_projection(
    G: nx.Graph,
    person_nodes: List[int],
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
) -> nx.Graph:
    """Create person-person projection from bipartite person-event graph.
    
    Two persons are connected if they share at least one event AND were both
    alive when the event occurred (temporal filtering).
    Edge weight = number of shared events (or sum of original edge weights).
    
    Args:
        G: Bipartite graph with person and event nodes.
        person_nodes: List of person node IDs.
        start_year: Start year for temporal filtering (optional).
        end_year: End year for temporal filtering (optional).
        
    Returns:
        Person-person projection graph (undirected, weighted).
    """
    if start_year is not None or end_year is not None:
        print(f"Creating person-person projection with temporal filter: {start_year}-{end_year}...")
    else:
        print("Creating person-person projection...")
    
    person_set = set(person_nodes)
    
    # Build projection: for each event, connect all persons linked to it
    G_proj = nx.Graph()
    
    # Add all person nodes
    for p in person_nodes:
        G_proj.add_node(p)
    
    # For each event, find all connected persons and create edges
    for node in G.nodes():
        if node not in person_set:  # This is an event
            node_data = G.nodes[node]
            
            # Temporal filtering: check if event overlaps with time range
            if start_year is not None or end_year is not None:
                event_start = extract_year_from_date(node_data.get('start_time'))
                event_end = extract_year_from_date(node_data.get('end_time'))
                
                # Use event year for filtering (prefer start_time, fallback to end_time)
                event_year = event_start if event_start else event_end
                
                if event_year is None:
                    continue  # Skip events without dates if filtering is enabled
                
                if start_year is not None and event_end is not None and event_end < start_year:
                    continue  # Event ended before range
                if end_year is not None and event_start is not None and event_start > end_year:
                    continue  # Event started after range
            
            neighbors = list(G.neighbors(node))
            person_neighbors = [n for n in neighbors if n in person_set]
            
            if len(person_neighbors) >= 2:
                # Get event year for temporal person filtering
                event_start = extract_year_from_date(node_data.get('start_time'))
                event_end = extract_year_from_date(node_data.get('end_time'))
                event_year = event_start if event_start else event_end
                
                # Connect all pairs of persons who share this event
                for i, p1 in enumerate(person_neighbors):
                    p1_data = G.nodes[p1]
                    p1_birth = extract_year_from_date(p1_data.get('date_of_birth'))
                    p1_death = extract_year_from_date(p1_data.get('date_of_death'))
                    
                    for p2 in person_neighbors[i+1:]:
                        p2_data = G.nodes[p2]
                        p2_birth = extract_year_from_date(p2_data.get('date_of_birth'))
                        p2_death = extract_year_from_date(p2_data.get('date_of_death'))
                        
                        # Temporal filtering: both persons must be alive when event occurred
                        if event_year is not None:
                            # Check if both persons were alive during event
                            p1_alive = True
                            p2_alive = True
                            
                            if p1_birth and p1_birth > event_year:
                                p1_alive = False  # Not born yet
                            if p1_death and p1_death < event_year:
                                p1_alive = False  # Already dead
                            
                            if p2_birth and p2_birth > event_year:
                                p2_alive = False  # Not born yet
                            if p2_death and p2_death < event_year:
                                p2_alive = False  # Already dead
                            
                            if not (p1_alive and p2_alive):
                                continue  # Skip if either person wasn't alive during event
                        
                        # Get original edge weights
                        w1 = G.get_edge_data(p1, node, {}).get('weight', 1.0)
                        w2 = G.get_edge_data(p2, node, {}).get('weight', 1.0)
                        # Use average or sum of weights
                        shared_weight = (w1 + w2) / 2.0
                        
                        if G_proj.has_edge(p1, p2):
                            # Increment weight (multiple shared events)
                            G_proj[p1][p2]['weight'] += shared_weight
                        else:
                            G_proj.add_edge(p1, p2, weight=shared_weight)
    
    print(f"  Projection: {G_proj.number_of_nodes()} persons, {G_proj.number_of_edges()} edges")
    return G_proj


def calculate_small_world_metrics(G: nx.Graph) -> Tuple[float, float, Dict]:
    """Calculate standard small-world metrics using optimized NetworkX functions.
    
    Computes average shortest path length and clustering coefficient.
    Uses NetworkX's optimized algorithms instead of manual loops.
    
    Args:
        G: NetworkX graph (person-person projection).
        
    Returns:
        Tuple containing:
        - L_actual: Average shortest path length
        - C_actual: Clustering coefficient
        - stats_dict: Dictionary with detailed statistics
    """
    print("Calculating small-world metrics...")
    
    # Get largest connected component
    if not nx.is_connected(G):
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        largest_cc = components[0]
        G = G.subgraph(largest_cc).copy()
        print(f"  Using largest connected component: {G.number_of_nodes()} nodes")
    
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return float('inf'), 0.0, {"error": "Graph too small"}
    
    # Use NetworkX's optimized function for average shortest path length
    # This is MUCH faster than calculating paths one by one
    print("  Computing average shortest path length (optimized)...")
    try:
        # For weighted graphs, use all_pairs_dijkstra_path_length which is faster
        if G.number_of_edges() > 0:
            # Check if graph has weights
            has_weights = any('weight' in G[u][v] for u, v in list(G.edges())[:10])
            
            if has_weights:
                # For weighted graphs, sample if too large
                if G.number_of_nodes() > 1000:
                    # Sample nodes for very large graphs
                    import random
                    sample_nodes = random.sample(nodes, min(500, len(nodes)))
                    path_lengths = []
                    for source in tqdm(sample_nodes, desc="Computing paths", unit="node"):
                        lengths = nx.single_source_dijkstra_path_length(G, source, weight='weight')
                        path_lengths.extend([l for l in lengths.values() if l > 0])
                    L_actual = sum(path_lengths) / len(path_lengths) if path_lengths else float('inf')
                else:
                    # Use all pairs for smaller graphs
                    L_actual = nx.average_shortest_path_length(G, weight='weight')
            else:
                # Unweighted - use faster unweighted algorithm
                L_actual = nx.average_shortest_path_length(G)
        else:
            L_actual = float('inf')
    except Exception as e:
        print(f"  Warning: Could not compute all-pairs shortest paths: {e}")
        print(f"  Falling back to sampling...")
        # Fallback: sample pairs
        import random
        sample_size = min(1000, len(nodes) * (len(nodes) - 1) // 2)
        pairs = []
        attempts = 0
        while len(pairs) < sample_size and attempts < sample_size * 10:
            p1 = random.choice(nodes)
            p2 = random.choice(nodes)
            if p1 != p2 and (p1, p2) not in pairs and (p2, p1) not in pairs:
                pairs.append((p1, p2))
            attempts += 1
        
        path_lengths = []
        for n1, n2 in tqdm(pairs[:100], desc="Sampling paths", unit="pair"):  # Limit to 100 for speed
            try:
                path_length = nx.shortest_path_length(G, n1, n2)
                path_lengths.append(path_length)
            except nx.NetworkXNoPath:
                continue
        
        if path_lengths:
            L_actual = sum(path_lengths) / len(path_lengths)
        else:
            L_actual = float('inf')
    
    # Calculate clustering coefficient (already optimized in NetworkX)
    print("  Computing clustering coefficient...")
    C_actual = nx.average_clustering(G, weight='weight')
    
    stats = {
        "average_path_length": L_actual,
        "clustering_coefficient": C_actual,
        "nodes_in_component": len(nodes),
        "edges_in_component": G.number_of_edges()
    }
    
    return L_actual, C_actual, stats


def calculate_random_baseline(G: nx.Graph, num_samples: int = 3) -> Tuple[float, float]:
    """Calculate random graph baseline for small-world comparison.
    
    Generates Erdős–Rényi random graphs G(n,m) with same number of nodes and edges
    as the original graph, following the standard methodology from Watts & Strogatz (1998).
    
    Uses NetworkX's gnm_random_graph() which implements the Erdős–Rényi model:
    - n nodes, m edges chosen uniformly at random from all possible edges
    - This is the standard baseline for small-world comparison
    
    Args:
        G: Original graph.
        num_samples: Number of random graphs to generate (reduced for speed).
        
    Returns:
        Tuple containing (L_random, C_random) averages across random graphs.
    """
    print(f"Generating {num_samples} random graphs for baseline...")
    
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    # For very large graphs, reduce samples and use sampling
    if n > 500:
        num_samples = 2
        print(f"  Large graph detected ({n} nodes), using {num_samples} samples")
    
    L_random_list = []
    C_random_list = []
    
    for i in tqdm(range(num_samples), desc="Random graphs"):
        # Generate random graph with same nodes and edges
        G_random = nx.gnm_random_graph(n, m)
        
        if not nx.is_connected(G_random):
            # Use largest component
            components = sorted(nx.connected_components(G_random), key=len, reverse=True)
            G_random = G_random.subgraph(components[0]).copy()
        
        if G_random.number_of_nodes() < 2:
            continue
        
        # Calculate metrics using optimized functions
        try:
            # For large graphs, use sampling
            if G_random.number_of_nodes() > 500:
                # Sample nodes for path length
                import random
                sample_nodes = random.sample(list(G_random.nodes()), min(200, G_random.number_of_nodes()))
                path_lengths = []
                for source in sample_nodes[:50]:  # Limit sources
                    lengths = nx.single_source_shortest_path_length(G_random, source)
                    path_lengths.extend([l for l in lengths.values() if l > 0])
                L_r = sum(path_lengths) / len(path_lengths) if path_lengths else None
            else:
                L_r = nx.average_shortest_path_length(G_random)
            
            C_r = nx.average_clustering(G_random)
            
            if L_r is not None:
                L_random_list.append(L_r)
            C_random_list.append(C_r)
        except Exception as e:
            print(f"  Warning: Error computing random graph {i+1}: {e}")
            continue
    
    if not L_random_list or not C_random_list:
        return None, None
    
    L_random = sum(L_random_list) / len(L_random_list)
    C_random = sum(C_random_list) / len(C_random_list)
    
    return L_random, C_random


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
    """Main function to analyze small world properties.
    
    Supports temporal filtering via command-line arguments:
    --start-year: Start year for filtering (inclusive)
    --end-year: End year for filtering (inclusive)
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze small-world properties with optional temporal filtering')
    parser.add_argument('--start-year', type=int, default=None, help='Start year for temporal filtering (inclusive)')
    parser.add_argument('--end-year', type=int, default=None, help='End year for temporal filtering (inclusive)')
    args = parser.parse_args()
    
    start_year = args.start_year
    end_year = args.end_year
    
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
    
    # Add time range to output filename if filtering
    if start_year is not None or end_year is not None:
        year_str = f"_{start_year or 'start'}_{end_year or 'end'}"
        output_file = data_analysis / f"small_world_analysis{year_str}.json"
    else:
        output_file = data_analysis / "small_world_analysis.json"
    
    print("=" * 80)
    print("Small World Analysis (Standard Algorithm)")
    print("=" * 80)
    if start_year is not None or end_year is not None:
        print(f"Temporal filtering: {start_year or 'start'} - {end_year or 'end'}")
    print("Step 1: Convert bipartite graph to person-person projection")
    print("Step 2: Calculate small-world metrics on projection")
    print("Step 3: Compare to random graph baseline")
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
        
        # Step 1: Filter nodes by time range (if specified)
        if start_year is not None or end_year is not None:
            print(f"\nFiltering nodes by time range: {start_year or 'start'} - {end_year or 'end'}")
            filtered_persons, filtered_events = filter_nodes_by_time_range(
                G, person_nodes, start_year, end_year
            )
            print(f"  Filtered persons: {len(filtered_persons):,} / {len(person_nodes):,}")
            print(f"  Filtered events: {len(filtered_events):,}")
            person_nodes = filtered_persons
        
        # Step 2: Create person-person projection (with temporal filtering)
        G_proj = create_person_person_projection(G, person_nodes, start_year, end_year)
        
        # Step 3: Calculate actual small-world metrics on projection
        L_actual, C_actual, path_stats = calculate_small_world_metrics(G_proj)
        
        # Step 4: Calculate random graph baseline
        L_random, C_random = calculate_random_baseline(G_proj)
        
        # Step 5: Determine if small-world property holds
        # Following Watts-Strogatz (1998) definition with quantitative criteria from Humphries & Gurney (2008)
        is_small_world = False
        interpretation = "Small world property may not hold"
        sigma = None  # Small-world coefficient
        
        if L_random is not None and C_random is not None:
            # Calculate ratios
            L_ratio = L_actual / L_random if L_random > 0 else float('inf')
            C_ratio = C_actual / C_random if C_random > 0 else 0
            
            # Calculate small-world coefficient σ (Humphries & Gurney, 2008)
            # σ = (C/C_random) / (L/L_random)
            # σ > 1 indicates small-world property
            if L_ratio > 0:
                sigma = C_ratio / L_ratio
            
            # Quantitative criteria (Humphries & Gurney, 2008):
            # - L_actual / L_random ≤ 1.5 (path length similar to random)
            # - C_actual / C_random ≥ 2.0 (clustering much higher than random)
            # Alternative: σ > 1 (small-world coefficient)
            is_small_world = (L_ratio <= 1.5 and C_ratio >= 2.0) or (sigma is not None and sigma > 1.0)
            
            interpretation = (
                "Small world property holds (L_actual/L_random ≈ {:.2f}, C_actual/C_random ≈ {:.2f}, σ ≈ {:.2f})"
                .format(L_ratio, C_ratio, sigma if sigma else 0)
            ) if is_small_world else (
                "Small world property may not hold (L_actual/L_random ≈ {:.2f}, C_actual/C_random ≈ {:.2f}, σ ≈ {:.2f})"
                .format(L_ratio, C_ratio, sigma if sigma else 0)
            )
        else:
            # Fallback to heuristic if random baseline fails
            is_small_world = L_actual <= 6.0 and C_actual > 0.1
            interpretation = (
                "Small world property holds (heuristic: L ≤ 6.0, C > 0.1)"
                if is_small_world
                else "Small world property may not hold (heuristic check failed)"
            )
        
        # Compile results
        results = {
            "temporal_filtering": {
                "start_year": start_year,
                "end_year": end_year,
                "enabled": start_year is not None or end_year is not None
            },
            "graph_statistics": {
                "total_nodes": G.number_of_nodes(),
                "total_edges": G.number_of_edges(),
                "person_nodes": len(person_nodes),
                "event_nodes": len(event_nodes),
                "projection_nodes": G_proj.number_of_nodes(),
                "projection_edges": G_proj.number_of_edges()
            },
            "small_world_metrics": {
                "average_shortest_path_length": L_actual,
                "clustering_coefficient": C_actual,
                "path_statistics": path_stats
            },
            "random_baseline": {
                "average_shortest_path_length": L_random,
                "clustering_coefficient": C_random,
                "L_ratio": L_actual / L_random if L_random else None,
                "C_ratio": C_actual / C_random if C_random else None,
                "small_world_coefficient_sigma": sigma
            },
            "small_world_property": {
                "is_small_world": is_small_world,
                "average_path_length": L_actual,
                "clustering_coefficient": C_actual,
                "interpretation": interpretation
            }
        }
        
        # Save results
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 80)
        print("Results")
        print("=" * 80)
        print(f"Person-person projection: {G_proj.number_of_nodes()} nodes, {G_proj.number_of_edges()} edges")
        print(f"\nActual metrics:")
        print(f"  Average shortest path length (L_actual): {L_actual:.4f}")
        print(f"  Clustering coefficient (C_actual): {C_actual:.4f}")
        print(f"  Connected pairs: {path_stats.get('connected_pairs', 0):,}")
        
        if L_random is not None and C_random is not None:
            L_ratio = L_actual / L_random
            C_ratio = C_actual / C_random
            sigma = C_ratio / L_ratio if L_ratio > 0 else None
            print(f"\nRandom baseline:")
            print(f"  Average shortest path length (L_random): {L_random:.4f}")
            print(f"  Clustering coefficient (C_random): {C_random:.4f}")
            print(f"\nRatios (Watts-Strogatz comparison):")
            print(f"  L_actual / L_random = {L_ratio:.4f}")
            print(f"  C_actual / C_random = {C_ratio:.4f}")
            if sigma:
                print(f"  Small-world coefficient σ = C_ratio / L_ratio = {sigma:.4f}")
            print(f"\nSmall world property: {results['small_world_property']['interpretation']}")
            print(f"  Criteria (Humphries & Gurney, 2008):")
            print(f"    L_actual/L_random ≤ 1.5: {L_ratio <= 1.5}")
            print(f"    C_actual/C_random ≥ 2.0: {C_ratio >= 2.0}")
            if sigma:
                print(f"    σ > 1.0: {sigma > 1.0}")
        else:
            print(f"\nSmall world property: {results['small_world_property']['interpretation']}")
            print(f"  (Using heuristic thresholds: L ≤ 6.0, C > 0.1)")
        
        print(f"\nResults saved to: {output_file}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error analyzing small world: {e}")
        raise


if __name__ == "__main__":
    main()

