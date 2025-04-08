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

    def find_shortest_path_k(self, graph, source, destination, k):
        distances = {node: float('inf') for node in graph.adj}
        distances[source] = 0
        paths = {node: [] for node in graph.adj}
        paths[source] = [source]
        relaxation_count = {node: 0 for node in graph.adj}
        priority_queue = [(0, source)]

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            if current_distance > distances[current_node]:
                continue

            for neighbor in graph.connected_nodes(current_node):
                if relaxation_count[neighbor] < k:
                    edge_weight = graph.get_edge_weight(current_node, neighbor)
                    distance = current_distance + edge_weight
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        paths[neighbor] = paths[current_node] + [neighbor]
                        heapq.heappush(priority_queue, (distance, neighbor))
                        relaxation_count[neighbor] += 1

        return distances, paths, relaxation_count



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
    """
    Bellman-Ford algorithm with global iteration limit (k).
    """
    def __init__(self, k: int):
        self.k = k

    def find_shortest_path(self, graph: WeightedGraph, source: int, destination: int) -> list:
        distances, paths = self._bellman_ford(graph, source, self.k)
        return paths.get(destination, [])

    def _bellman_ford(self, graph: WeightedGraph, source: int, k: int):
        distances = {node: float('inf') for node in graph.adj}
        distances[source] = 0
        paths = {node: [] for node in graph.adj}
        paths[source] = [source]

        for _ in range(k):
            updated = False
            for u in graph.adj:
                for v in graph.connected_nodes(u):
                    weight = graph.get_edge_weight(u, v)
                    if distances[u] + weight < distances[v]:
                        distances[v] = distances[u] + weight
                        paths[v] = paths[u] + [v]
                        updated = True
            if not updated:
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
    
    def find_path_k(self, source: int, destination: int, k) -> list:
        return self.algorithm.find_shortest_path_k(self.graph, source, destination,k)

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
    # Create a graph with n nodes and approximately density*n*(n-1) directed edges
    graph = {i: {} for i in range(n)}
    possible_edges = n * (n - 1)  # Total possible directed edges excluding self-loops
    target_edges = int(possible_edges * density)  # Target number of edges based on density
    edges_added = 0

    while edges_added < target_edges:
        u, v = random.randint(0, n - 1), random.randint(0, n - 1)
        if u != v and v not in graph[u]:
            graph[u][v] = random.randint(1, 10)
            edges_added += 1
    return graph


def generate_random_undirected_graph(n, density=0.05):
    """
    Generate a sparse undirected graph with approximately density*n*(n-1)/2 edges.
    """
    graph = WeightedGraph(n)
    possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(possible_edges)

    target_edges = int(density * len(possible_edges))
    for i in range(target_edges):
        u, v = possible_edges[i]
        weight = random.randint(1, 10)
        graph.add_edge(u, v, weight) 

    return graph

def generate_strict_chain_graph(n):
    g = WeightedGraph(n)
    for i in range(n - 1):
        g.add_edge(i, i + 1, 1)
    return g


def measure_performance(graph, source, k, algorithm):
    """
    Measure the performance of the specified algorithm.

    Returns:
        execution_time: Time taken to execute the algorithm (in seconds)
        accuracy: Percentage of nodes that have a valid path
    """
    start_time = time.time()
    distances, paths, relax_count = algorithm(graph, source, k, track_relaxations=True)
    end_time = time.time()
    return end_time - start_time, relax_count

def draw_plot(x_values, dijkstra_values, bellman_values, x_label, y_label, title, y_limit=None, scale_factor=1.0):
    x = np.array(x_values)
    fig, ax = plt.subplots(figsize=(14, 6))

    if dijkstra_values:
        d_values = [v * scale_factor for v in dijkstra_values]
        if len(d_values) == len(x):
            ax.plot(x, d_values, 'o-', label="Dijkstra", linewidth=2, markersize=8)
            ax.axhline(np.mean(d_values), color="blue", linestyle=":", linewidth=1.5, label="Dijkstra Avg")

    if bellman_values:
        b_values = [v * scale_factor for v in bellman_values]
        if len(b_values) == len(x):
            ax.plot(x, b_values, 's--', label="Bellman-Ford", linewidth=2, markersize=8)
            ax.axhline(np.mean(b_values), color="green", linestyle=":", linewidth=1.5, label="Bellman Avg")

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14)
    if y_limit:
        ax.set_ylim(*y_limit)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)
    plt.tight_layout()
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
    graph_sizes = [10, 50, 100, 200]
    densities = [0.1, 0.3, 0.5, 0.7, 0.9]
    k_values = [1, 3, 5, 10]
    repeat_count = 1
    accuracy_k_values = [1, 3, 5, 10]
    accuracy_trials = 1
    graph_size = 100
    graph_density = 0.5

    results = {
        'dijkstra': defaultdict(list),
        'bellman_ford': defaultdict(list),
        'accuracy_k_values': accuracy_k_values
    }

    def convert_to_weighted_graph(graph_dict):
        wg = WeightedGraph(len(graph_dict))
        for u in graph_dict:
            for v, w in graph_dict[u].items():
                wg.add_edge(u, v, w)
        return wg

    def generate_random_graph(n, density=0.5):
        graph = {i: {} for i in range(n)}
        possible_edges = n * (n - 1)
        target_edges = int(possible_edges * density)
        edges_added = 0

        while edges_added < target_edges:
            u, v = np.random.randint(0, n), np.random.randint(0, n)
            if u != v and v not in graph[u]:
                graph[u][v] = np.random.randint(1, 11)
                edges_added += 1
        return graph

    print("Experiment 1: Varying Graph Size")
    for size in graph_sizes:
        raw_graph = generate_random_graph(size, 0.5)
        wg = convert_to_weighted_graph(raw_graph)
        source = 0
        k = size

        dijkstra_finder = ShortestPathFinder(wg, DijkstraPathFinder())
        d_start = time.time()
        relax_count = 0
        for dest in wg.adj:
            if dest != source:
                _, _, rel = dijkstra_finder.find_path_k(source, dest, k=k)
                relax_count += sum(rel.values())
        d_time = (time.time() - d_start) * 1000

        bellman_finder = ShortestPathFinder(wg, BellmanFordPathFinder(k))
        b_start = time.time()
        for dest in wg.adj:
            if dest != source:
                bellman_finder.find_path(source, dest)
        b_time = (time.time() - b_start) * 1000

        results['dijkstra']['size_time'].append(d_time)
        results['dijkstra']['size_relax'].append(relax_count)
        results['bellman_ford']['size_time'].append(b_time)

        print(f"Size {size}: Dijkstra {d_time:.2f}ms, {relax_count} relax | Bellman-Ford {b_time:.2f}ms")

    print("\nExperiment 2: Varying Graph Density")
    for density in densities:
        raw_graph = generate_random_graph(100, density)
        wg = convert_to_weighted_graph(raw_graph)
        source = 0
        k = 100

        dijkstra_finder = ShortestPathFinder(wg, DijkstraPathFinder())
        d_start = time.time()
        relax_count = 0
        for dest in wg.adj:
            if dest != source:
                _, _, rel = dijkstra_finder.find_path_k(source, dest, k=k)
                relax_count += sum(rel.values())
        d_time = (time.time() - d_start) * 1000

        bellman_finder = ShortestPathFinder(wg, BellmanFordPathFinder(k))
        b_start = time.time()
        for dest in wg.adj:
            if dest != source:
                bellman_finder.find_path(source, dest)
        b_time = (time.time() - b_start) * 1000

        results['dijkstra']['density_time'].append(d_time)
        results['dijkstra']['density_relax'].append(relax_count)
        results['bellman_ford']['density_time'].append(b_time)

        print(f"Density {density:.1f}: Dijkstra {d_time:.2f}ms, {relax_count} relax | Bellman-Ford {b_time:.2f}ms")

    print("\nExperiment 3: Varying k Value (Averaged over trials)")
    trials = 5
    for k in k_values:
        d_time_total = 0
        b_time_total = 0
        d_relax_total = 0

        for _ in range(trials):
            raw_graph = generate_random_graph(100, 0.5)
            wg = convert_to_weighted_graph(raw_graph)
            source = 0

            dijkstra_finder = ShortestPathFinder(wg, DijkstraPathFinder())
            d_start = time.time()
            relax_count = 0
            for dest in wg.adj:
                if dest != source:
                    _, _, rel = dijkstra_finder.find_path_k(source, dest, k=k)
                    relax_count += sum(rel.values())
            d_time = (time.time() - d_start) * 1000
            d_time_total += d_time
            d_relax_total += relax_count

            bellman_finder = ShortestPathFinder(wg, BellmanFordPathFinder(k))
            b_start = time.time()
            for dest in wg.adj:
                if dest != source:
                    bellman_finder.find_path(source, dest)
            b_time = (time.time() - b_start) * 1000
            b_time_total += b_time

        avg_d_time = d_time_total / trials
        avg_b_time = b_time_total / trials
        avg_d_relax = d_relax_total / trials

        results['dijkstra']['k_time'].append(avg_d_time)
        results['dijkstra']['k_relax'].append(avg_d_relax)
        results['bellman_ford']['k_time'].append(avg_b_time)

        print(f"k={k}: Dijkstra = {avg_d_time:.2f}ms, relax = {avg_d_relax:.1f} | Bellman-Ford = {avg_b_time:.2f}ms")

    print("\nExperiment 4: Accuracy Comparison Between Algorithms")
    accuracy_results = {
        'dijkstra': {k: {'correct': 0, 'total': 0, 'relax': 0} for k in accuracy_k_values},
        'bellman_ford': {k: {'correct': 0, 'total': 0} for k in accuracy_k_values}
    }

    for _ in range(accuracy_trials):
        raw_graph = generate_random_graph(graph_size, graph_density)
        wg = convert_to_weighted_graph(raw_graph)
        source = 0

        gt_finder = ShortestPathFinder(wg, DijkstraPathFinder())
        gt_paths = {}
        for dest in range(graph_size):
            if dest == source:
                continue
            path = gt_finder.find_path(source, dest)
            if path:
                gt_paths[dest] = path

        for k in accuracy_k_values:
            dj_finder = ShortestPathFinder(wg, DijkstraPathFinder())
            _, all_paths, rel = dj_finder.find_path_k(source, None, k=k)
            for dest in gt_paths:
                acc_path = all_paths.get(dest, [])
                if acc_path == gt_paths[dest]:
                    accuracy_results['dijkstra'][k]['correct'] += 1
                accuracy_results['dijkstra'][k]['total'] += 1
                accuracy_results['dijkstra'][k]['relax'] += sum(rel.values())

            bf_finder = ShortestPathFinder(wg, BellmanFordPathFinder(k))
            for dest in gt_paths:
                path = bf_finder.find_path(source, dest)
                if path == gt_paths[dest]:
                    accuracy_results['bellman_ford'][k]['correct'] += 1
                accuracy_results['bellman_ford'][k]['total'] += 1

    d_acc = []
    b_acc = []
    d_relax_list = []
    for k in accuracy_k_values:
        d = accuracy_results['dijkstra'][k]
        b = accuracy_results['bellman_ford'][k]
        d_percent = 100 * d['correct'] / d['total'] if d['total'] else 0
        b_percent = 100 * b['correct'] / b['total'] if b['total'] else 0
        d_acc.append(d_percent)
        b_acc.append(b_percent)
        d_relax_list.append(d['relax'])
        print(f"k={k}: Dijkstra Accuracy = {d_percent:.2f}%, {d['relax']} relax | Bellman-Ford Accuracy = {b_percent:.2f}%")

    results['dijkstra']['accuracy'] = d_acc
    results['dijkstra']['accuracy_relax'] = d_relax_list
    results['bellman_ford']['accuracy'] = b_acc


    draw_plot(graph_sizes, results['dijkstra']['size_time'], results['bellman_ford']['size_time'],
              "Graph Size (nodes)", "Execution Time (ms)", "Execution Time vs Graph Size")

    draw_plot(densities, results['dijkstra']['density_time'], results['bellman_ford']['density_time'],
              "Graph Density", "Execution Time (ms)", "Execution Time vs Graph Density")

    draw_plot(k_values, results['dijkstra']['k_time'], results['bellman_ford']['k_time'],
              "k Value", "Execution Time (ms)", "Execution Time vs Relaxation Limit (k)")

    draw_plot(accuracy_k_values, d_acc, b_acc,
              "k Value", "Accuracy (%)", "Path Finding Accuracy vs k Value (Experiment 4)")

    return results



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
    # experiment_part_5()
