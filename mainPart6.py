import csv
import heapq
import math
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, deque
from abc import ABC, abstractmethod

# =============================================================================
# Graph Classes
# =============================================================================
class WeightedGraph:
    """
    Represents an undirected weighted graph.
    """
    def __init__(self, num_nodes):
        # Initialize an adjacency list and a dictionary of edge weights.
        self.adj = {i: [] for i in range(num_nodes)}
        self.weights = {}  # key: (node1, node2), value: weight

    def add_edge(self, node1, node2, weight):
        if node2 not in self.adj[node1]:
            self.adj[node1].append(node2)
            self.weights[(node1, node2)] = weight
            # Undirected: add both ways.
            self.adj[node2].append(node1)
            self.weights[(node2, node1)] = weight

    def connected_nodes(self, node):
        return self.adj[node]

    def get_edge_weight(self, node1, node2):
        return self.weights.get((node1, node2), float('inf'))

    def number_of_nodes(self):
        return len(self.adj)

# =============================================================================
# PathFinder Interface and Implementations
# =============================================================================
class PathFinder(ABC):
    """
    Abstract base class for shortest path algorithms.
    """
    @abstractmethod
    def find_shortest_path(self, graph: WeightedGraph, source: int, destination: int) -> list:
        pass

class DijkstraPathFinder(PathFinder):
    """
    Implements Dijkstra's algorithm.
    """
    def find_shortest_path(self, graph: WeightedGraph, source: int, destination: int) -> list:
        open_set = []
        heapq.heappush(open_set, (0, source))
        came_from = {}
        g_score = {node: float('inf') for node in graph.adj}
        g_score[source] = 0

        while open_set:
            current_cost, current = heapq.heappop(open_set)
            if current == destination:
                return self._reconstruct_path(came_from, source, destination)

            for neighbor in graph.connected_nodes(current):
                edge_weight = graph.get_edge_weight(current, neighbor)
                tentative_score = current_cost + edge_weight
                if tentative_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_score
                    heapq.heappush(open_set, (tentative_score, neighbor))
        return []

    def _reconstruct_path(self, came_from: dict, source: int, destination: int) -> list:
        path = []
        current = destination
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(source)
        path.reverse()
        return path

class AStarAdapter(PathFinder):
    """
    Adapter for the A* algorithm. It wraps your existing A* implementation.
    """
    def __init__(self, station_positions: dict, destination: int):
        # Precompute a heuristic based on Euclidean distances.
        self.heuristic = self._build_heuristic(station_positions, destination)

    def _build_heuristic(self, station_positions: dict, destination: int) -> dict:
        dest_pos = station_positions[destination]
        heuristic = {}
        for station, pos in station_positions.items():
            heuristic[station] = math.sqrt((pos[0] - dest_pos[0])**2 + (pos[1] - dest_pos[1])**2)
        return heuristic

    def find_shortest_path(self, graph: WeightedGraph, source: int, destination: int) -> list:
        return a_star(graph, source, destination, self.heuristic)

class BellmanFordPathFinder(PathFinder):
    def __init__(self, k: int):
        self.k = k

    def find_shortest_path(self, graph: WeightedGraph, source: int, destination: int) -> list:
        distances, paths = self.bellman_ford(graph, source, self.k)
        return paths[destination] if distances[destination] != float('infinity') else []

    def bellman_ford(self, graph: WeightedGraph, source: int, k: int):
        distances = {node: float('infinity') for node in graph.adj}
        distances[source] = 0
        paths = {node: [] for node in graph.adj}
        paths[source] = [source]
        relaxation_count = {node: 0 for node in graph.adj}

        edges = []
        for u in graph.adj:
            for v in graph.adj[u]:
                w = graph.get_edge_weight(u, v)
                edges.append((u, v, w))

        for _ in range(len(graph.adj) - 1):
            no_changes = True
            for u, v, w in edges:
                if relaxation_count[v] >= k:
                    continue
                if distances[u] + w < distances[v]:
                    distances[v] = distances[u] + w
                    paths[v] = paths[u] + [v]
                    relaxation_count[v] += 1
                    no_changes = False
            if no_changes:
                break

        return distances, paths

# =============================================================================
# ShortestPathFinder Context (Strategy Pattern)
# =============================================================================
class ShortestPathFinder:
    """
    Uses a specified pathfinding algorithm to compute shortest paths.
    """
    def __init__(self, graph: WeightedGraph, algorithm: PathFinder):
        self.graph = graph
        self.algorithm = algorithm

    def find_path(self, source: int, destination: int) -> list:
        return self.algorithm.find_shortest_path(self.graph, source, destination)

# =============================================================================
# Utility Functions (Loading, Distance, etc.)
# =============================================================================
def euclidean_distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def load_stations(file_name):
    station_positions = {}
    with open(file_name, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            station_id = int(row['id'])
            lat = float(row['latitude'])
            lon = float(row['longitude'])
            station_positions[station_id] = (lat, lon)
    return station_positions

def load_connections(file_name):
    connections = []
    with open(file_name, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            station1 = int(row['station1'])
            station2 = int(row['station2'])
            line = int(row['line'])
            connections.append((station1, station2, line))
    return connections

def build_graph(stations_file, connections_file):
    station_positions = load_stations(stations_file)
    station_connections = load_connections(connections_file)
    num_nodes = max(station_positions.keys()) + 1
    graph = WeightedGraph(num_nodes)
    for station1, station2, _ in station_connections:
        weight = euclidean_distance(station_positions[station1], station_positions[station2])
        graph.add_edge(station1, station2, weight)
    return graph, station_positions, station_connections

def build_connection_lookup(station_connections):
    lookup = {}
    for station1, station2, line in station_connections:
        lookup[(station1, station2)] = line
        lookup[(station2, station1)] = line
    return lookup

def count_transfers(path, connection_lookup):
    if not path or len(path) < 2:
        return 0
    current_line = connection_lookup.get((path[0], path[1]))
    transfers = 0
    for i in range(1, len(path)-1):
        segment_line = connection_lookup.get((path[i], path[i+1]))
        if segment_line != current_line:
            transfers += 1
            current_line = segment_line
    return transfers

# =============================================================================
# Utility Functions For Part 2:
# =============================================================================
def generate_random_graph(n, density=0.5):
    """
    Generate a random graph with n nodes and approximately density*n*(n-1) edges.
        
    Returns:
        graph: Dictionary representation of the graph
    """
    graph = {i: {} for i in range(n)}
    
    # Number of possible edges in a directed graph is n*(n-1)
    possible_edges = n * (n - 1)
    target_edges = int(possible_edges * density)
    
    edges_added = 0
    while edges_added < target_edges:
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        
        # Skip self-loops
        if u == v or v in graph[u]:
            continue
            
        # Add edge with random weight (1-10)
        weight = random.randint(1, 10)
        graph[u][v] = weight
        edges_added += 1
    
    return graph

def measure_performance(graph, source, k, algorithm):
    """
    Measure the performance of the specified algorithm.

    Returns:
        execution_time: Time taken to execute the algorithm (in seconds)
        accuracy: Percentage of nodes that have a valid path
    """
    start_time = time.time()
    distances, paths = algorithm(graph, source, k)
    end_time = time.time()
    
    # Measure accuracy (percentage of nodes with valid paths)
    reachable_nodes = sum(1 for d in distances.values() if d != float('infinity'))
    accuracy = reachable_nodes / len(graph) * 100
    
    return end_time - start_time, accuracy

def draw_plot(x_values, dijkstra_values, bellman_values, x_label, y_label, title):
    """
    Draw a plot comparing Dijkstra and Bellman-Ford performance.

    """
    x = np.array(x_values)
    
    # Create the figure with specified size
    fig = plt.figure(figsize=(20, 8))
    
    # Plot Dijkstra's algorithm results
    plt.plot(x, dijkstra_values, 'b-o', label="Dijkstra's Algorithm")
    
    # Plot Bellman-Ford algorithm results
    plt.plot(x, bellman_values, 'g-s', label="Bellman-Ford Algorithm")
    
    # Calculate and show mean lines
    dijkstra_mean = np.mean(dijkstra_values)
    bellman_mean = np.mean(bellman_values)
    
    plt.axhline(dijkstra_mean, color="blue", linestyle="--", label="Dijkstra Avg")
    plt.axhline(bellman_mean, color="green", linestyle="--", label="Bellman-Ford Avg")
    
    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    
    # Display the plot
    plt.show()

# =============================================================================
# A* Implementation (Existing code used via adapter)
# =============================================================================
def a_star(graph: WeightedGraph, start, goal, heuristic):
    open_set = []
    heapq.heappush(open_set, (heuristic.get(start, 0), start))
    came_from = {}
    g_score = {node: float('inf') for node in graph.adj.keys()}
    g_score[start] = 0
    f_score = {node: float('inf') for node in graph.adj.keys()}
    f_score[start] = heuristic.get(start, 0)

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            return _reconstruct_path(came_from, start, goal)
        for neighbor in graph.connected_nodes(current):
            edge_weight = graph.get_edge_weight(current, neighbor)
            tentative_g = g_score[current] + edge_weight
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic.get(neighbor, 1000)
                if neighbor not in [n for _, n in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return []

def _reconstruct_path(came_from, start, goal):
    path = []
    current = goal
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path

# =============================================================================
# Experiment Function for Part 2.3
# =============================================================================
def run_experiment():
    """
    Run the performance analysis experiment comparing class-based Dijkstra and Bellman-Ford algorithms.
    """
    graph_sizes = [10, 50, 100, 200]
    densities = [0.1, 0.3, 0.5, 0.7]
    k_values = [1, 3, 5, 10]

    results = {
        'dijkstra': defaultdict(list),
        'bellman_ford': defaultdict(list)
    }

    def convert_to_weighted_graph(graph_dict):
        wg = WeightedGraph(len(graph_dict))
        for u in graph_dict:
            for v, w in graph_dict[u].items():
                wg.add_edge(u, v, w)
        return wg

    print("Experiment 1: Varying Graph Size")
    for size in graph_sizes:
        raw_graph = generate_random_graph(size, 0.5)
        wg = convert_to_weighted_graph(raw_graph)
        source = 0
        k = 3

        # Dijkstra
        dijkstra_algo = ShortestPathFinder(wg, DijkstraPathFinder())
        d_start = time.time()
        dijkstra_paths = {node: dijkstra_algo.find_path(source, node) for node in wg.adj}
        d_time = time.time() - d_start
        d_accuracy = sum(1 for path in dijkstra_paths.values() if path) / size * 100

        # Bellman-Ford
        bellman_algo = ShortestPathFinder(wg, BellmanFordPathFinder(k))
        b_start = time.time()
        bellman_paths = {node: bellman_algo.find_path(source, node) for node in wg.adj}
        b_time = time.time() - b_start
        b_accuracy = sum(1 for path in bellman_paths.values() if path) / size * 100

        results['dijkstra']['size_time'].append(d_time)
        results['dijkstra']['size_acc'].append(d_accuracy)
        results['bellman_ford']['size_time'].append(b_time)
        results['bellman_ford']['size_acc'].append(b_accuracy)

        print(f"Size {size}: Dijkstra {d_time:.5f}s, {d_accuracy:.2f}% | Bellman-Ford {b_time:.5f}s, {b_accuracy:.2f}%")

    print("\nExperiment 2: Varying Graph Density")
    for density in densities:
        raw_graph = generate_random_graph(100, density)
        wg = convert_to_weighted_graph(raw_graph)
        source = 0
        k = 3

        dijkstra_algo = ShortestPathFinder(wg, DijkstraPathFinder())
        d_start = time.time()
        dijkstra_paths = {node: dijkstra_algo.find_path(source, node) for node in wg.adj}
        d_time = time.time() - d_start
        d_accuracy = sum(1 for path in dijkstra_paths.values() if path) / 100 * 100

        bellman_algo = ShortestPathFinder(wg, BellmanFordPathFinder(k))
        b_start = time.time()
        bellman_paths = {node: bellman_algo.find_path(source, node) for node in wg.adj}
        b_time = time.time() - b_start
        b_accuracy = sum(1 for path in bellman_paths.values() if path) / 100 * 100

        results['dijkstra']['density_time'].append(d_time)
        results['dijkstra']['density_acc'].append(d_accuracy)
        results['bellman_ford']['density_time'].append(b_time)
        results['bellman_ford']['density_acc'].append(b_accuracy)

        print(f"Density {density:.1f}: Dijkstra {d_time:.5f}s, {d_accuracy:.2f}% | Bellman-Ford {b_time:.5f}s, {b_accuracy:.2f}%")

    print("\nExperiment 3: Varying k Value")
    for k in k_values:
        raw_graph = generate_random_graph(100, 0.5)
        wg = convert_to_weighted_graph(raw_graph)
        source = 0

        dijkstra_algo = ShortestPathFinder(wg, DijkstraPathFinder())
        d_start = time.time()
        dijkstra_paths = {node: dijkstra_algo.find_path(source, node) for node in wg.adj}
        d_time = time.time() - d_start
        d_accuracy = sum(1 for path in dijkstra_paths.values() if path) / 100 * 100

        bellman_algo = ShortestPathFinder(wg, BellmanFordPathFinder(k))
        b_start = time.time()
        bellman_paths = {node: bellman_algo.find_path(source, node) for node in wg.adj}
        b_time = time.time() - b_start
        b_accuracy = sum(1 for path in bellman_paths.values() if path) / 100 * 100

        results['dijkstra']['k_time'].append(d_time)
        results['dijkstra']['k_acc'].append(d_accuracy)
        results['bellman_ford']['k_time'].append(b_time)
        results['bellman_ford']['k_acc'].append(b_accuracy)

        print(f"k={k}: Dijkstra {d_time:.5f}s, {d_accuracy:.2f}% | Bellman-Ford {b_time:.5f}s, {b_accuracy:.2f}%")

    # Plot 1: Time vs Graph Size
    draw_plot(
        graph_sizes,
        results['dijkstra']['size_time'],
        results['bellman_ford']['size_time'],
        "Graph Size (nodes)",
        "Execution Time (s)",
        "Execution Time vs Graph Size (k=3, density=0.5)"
    )

    # Plot 2: Time vs Graph Density
    draw_plot(
        densities,
        results['dijkstra']['density_time'],
        results['bellman_ford']['density_time'],
        "Graph Density",
        "Execution Time (s)",
        "Execution Time vs Graph Density (size=100, k=3)"
    )

    # Plot 3: Time vs k Value
    draw_plot(
        k_values,
        results['dijkstra']['k_time'],
        results['bellman_ford']['k_time'],
        "k Value",
        "Execution Time (s)",
        "Execution Time vs k Value (size=100, density=0.5)"
    )

    # Plot 4: Accuracy vs k Value
    draw_plot(
        k_values,
        results['dijkstra']['k_acc'],
        results['bellman_ford']['k_acc'],
        "k Value",
        "Accuracy (%)",
        "Path Discovery Accuracy vs k Value (size=100, density=0.5)"
    )


# =============================================================================
# Experiment Function for Part 5 (with transfers)
# =============================================================================
def experiment_part_5():
    stations_file = "london_stations.csv"
    connections_file = "london_connections.csv"
    # Build graph and load station & connection data.
    graph, station_positions, station_connections = build_graph(stations_file, connections_file)
    stations = list(station_positions.keys())
    connection_lookup = build_connection_lookup(station_connections)

    # Containers to accumulate times per transfer category.
    timings = {
        "same_line": {"dijkstra_times": [], "astar_times": []},
        "one_transfer": {"dijkstra_times": [], "astar_times": []},
        "multiple_transfers": {"dijkstra_times": [], "astar_times": []}
    }
    results = []

    # We'll use our DijkstraPathFinder and AStarAdapter as our algorithms.
    dijkstra_finder = DijkstraPathFinder()

    # For each pair, we instantiate an A* adapter with the destination's heuristic.
    for source in stations:
        for destination in stations:
            if source == destination:
                continue

            # Time Dijkstra
            start_time = time.perf_counter()
            dijkstra_path = dijkstra_finder.find_shortest_path(graph, source, destination)
            dijkstra_time = time.perf_counter() - start_time

            # Time A* using the adapter.
            astar_adapter = AStarAdapter(station_positions, destination)
            start_time = time.perf_counter()
            astar_path = astar_adapter.find_shortest_path(graph, source, destination)
            astar_time = time.perf_counter() - start_time

            # Count transfers (using the Dijkstra path as an example)
            transfers = count_transfers(dijkstra_path, connection_lookup)
            if transfers == 0:
                category = "same_line"
            elif transfers == 1:
                category = "one_transfer"
            else:
                category = "multiple_transfers"

            timings[category]["dijkstra_times"].append(dijkstra_time)
            timings[category]["astar_times"].append(astar_time)

            results.append({
                "source": source,
                "destination": destination,
                "transfers": transfers,
                "category": category,
                "dijkstra_time": dijkstra_time,
                "astar_time": astar_time
            })
            print(f"Source: {source}, Destination: {destination} | Transfers: {transfers} ({category}) | "
                  f"Dijkstra: {dijkstra_time:.6f}s, A*: {astar_time:.6f}s")

    # Compute average times per category.
    avg_results = {}
    for cat, times in timings.items():
        avg_dij = sum(times["dijkstra_times"]) / len(times["dijkstra_times"]) if times["dijkstra_times"] else 0
        avg_astar = sum(times["astar_times"]) / len(times["astar_times"]) if times["astar_times"] else 0
        avg_results[cat] = {"avg_dijkstra_time": avg_dij, "avg_astar_time": avg_astar}
    
    print("\nAverage Running Times by Transfer Category:")
    for cat, averages in avg_results.items():
        print(f"{cat.capitalize()}: Dijkstra = {averages['avg_dijkstra_time']:.6f}s, A* = {averages['avg_astar_time']:.6f}s")
    
    return avg_results, results

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    run_experiment()
    experiment_part_5()
