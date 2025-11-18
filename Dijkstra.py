from neo4j import GraphDatabase
import networkx as nx


class Neo4jGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def load_graph(self):
        query = """
        MATCH (n:Page)
        OPTIONAL MATCH (n)-[r:BICONNECTS]->(m:Page)
        RETURN n.id AS from, m.id AS to, r.weight AS weight
        """

        G = nx.DiGraph()

        with self.driver.session() as session:
            results = session.run(query)

            for record in results:
                node_from = record["from"]
                node_to = record["to"]
                weight = record["weight"]

                # Add nodes
                if node_from:
                    G.add_node(node_from)
                if node_to:
                    G.add_node(node_to)

                # Add edges
                if node_from and node_to:
                    G.add_edge(node_from, node_to, weight=weight)

        return G


if __name__ == "__main__":
    uri = "neo4j://localhost:7687"
    user = "neo4j"
    password = "neo4jtest12"

    start_node = 779
    end_node = 1742981

    neo = Neo4jGraph(uri, user, password)
    graph = neo.load_graph()
    neo.close()

    # Dijkstra shortest path
    try:
        path = nx.dijkstra_path(graph, start_node, end_node, weight="weight")
        distance = nx.dijkstra_path_length(graph, start_node, end_node, weight="weight")

        print("Đường đi ngắn nhất:", path)
        print("Tổng trọng số:", distance)
    except nx.NetworkXNoPath:
        print("Không có đường đi giữa 2 node.")
