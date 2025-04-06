import heapq


# ---------------- Graph Class ------------------
class Dgraph:
    def __init__(self, nodes):
        self.graph = {i: [] for i in range(nodes)}
        self.weight = {}

    def add_edge(self, u, v, w):
        self.graph[u].append(v)
        self.weight[(u, v)] = w

    def neighbors_nodes(self, node):
        return self.graph[node]

    def number_of_nodes(self):
        return len(self.graph)
    


# Check if graph contains any negative edge weight
def has_negative_edge(graph):
    for _, weight in graph.weight.items():
        if weight < 0:
            return True
    return False

# Step 2: Dijkstra 
def dijkstra(graph, source):
    num_nodes = graph.number_of_nodes()
    distances = {node: float('inf') for node in range(num_nodes)}
    predecessors = {node: None for node in range(num_nodes)}
    distances[source] = 0

    # priority queue as min-heap
    heap = [(0, source)]

    while heap:
        current_distance, u = heapq.heappop(heap)

        for v in graph.neighbors_nodes(u):
            weight = graph.weight[(u, v)]
            if distances[v] > current_distance + weight:
                distances[v] = current_distance + weight
                predecessors[v] = u
                heapq.heappush(heap, (distances[v], v))

    return distances, predecessors

# Step 3: Bellman-Ford 
def bellman_ford(graph, source):
    num_nodes = graph.number_of_nodes()
    distances = {node: float('inf') for node in range(num_nodes)}
    predecessors = {node: None for node in range(num_nodes)}
    distances[source] = 0

    # Relax all edges V-1 times
    for _ in range(num_nodes - 1):
        for (u, v), weight in graph.weight.items():
            if distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                predecessors[v] = u

    return distances, predecessors

# Step 4: Choosing which algorithm to use
def all_pairs_shortest_paths(graph):
    if has_negative_edge(graph):
        print("Using Bellman-Ford")
        return bellman_ford_all_pairs(graph)
    else:
        print("Using Dijkstra")
        return dijkstra_all_pairs(graph)

# Step 5: Run Dijkstra from every node to get all-pairs shortest paths
def dijkstra_all_pairs(graph):
    all_pairs_distances = {}
    all_pairs_predecessors = {}
    for node in range(graph.number_of_nodes()):
        distances, predecessors = dijkstra(graph, node)
        all_pairs_distances[node] = distances
        all_pairs_predecessors[node] = predecessors
    return all_pairs_distances, all_pairs_predecessors

# Step 6: Run Bellman-Ford from every node to get all-pairs shortest paths
def bellman_ford_all_pairs(graph):
    all_pairs_distances = {}
    all_pairs_predecessors = {}
    for node in range(graph.number_of_nodes()):
        distances, predecessors = bellman_ford(graph, node)
        all_pairs_distances[node] = distances
        all_pairs_predecessors[node] = predecessors
    return all_pairs_distances, all_pairs_predecessors

if __name__ == "__main__":
    print("=== Test Case 1: Non-negative edges (Dijkstra) ===")
    graph1 = Dgraph(5)
    graph1.add_edge(0, 1, 2)
    graph1.add_edge(0, 2, 4)
    graph1.add_edge(1, 2, 1)
    graph1.add_edge(1, 3, 7)
    graph1.add_edge(2, 4, 3)
    graph1.add_edge(3, 4, 1)

    distances, predecessors = all_pairs_shortest_paths(graph1)

    print("\nAll-Pairs Shortest Distances (Dijkstra):")
    for src in distances:
        print(f"From node {src}: {distances[src]}")

    print("\nPredecessors:")
    for src in predecessors:
        print(f"From node {src}: {predecessors[src]}")

    print("\n=== Test Case 2: Graph with negative edge (Bellman-Ford) ===")
    graph2 = Dgraph(4)
    graph2.add_edge(0, 1, 1)
    graph2.add_edge(1, 2, -2)  # Negative edge
    graph2.add_edge(2, 3, 2)
    graph2.add_edge(3, 1, 1)

    distances, predecessors = all_pairs_shortest_paths(graph2)

    print("\nAll-Pairs Shortest Distances (Bellman-Ford):")
    for src in distances:
        print(f"From node {src}: {distances[src]}")

    print("\nPredecessors:")
    for src in predecessors:
        print(f"From node {src}: {predecessors[src]}")




