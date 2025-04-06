import csv
import heapq
import math
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class Dgraph():
    def __init__(self, nodes):
        self.graph = {}
        self.weight = {}
        for i in range(nodes):
            self.graph[i] = []

    def are_connected(self, node1, node2):
        for node in self.adj[node1]:
            if node == node2:
                return True
        return False

    def connected_nodes(self, node):
        return self.graph[node]

    def add_node(self,):
        #add a new node number = length of existing node
        self.graph[len(self.graph)] = []

    def add_edge(self, node1, node2, weight):
        if node1 not in self.graph[node2]:
            self.graph[node1].append(node2)
            self.weight[(node1, node2)] = weight

            #since it is undirected
            self.graph[node2].append(node1)
            self.weight[(node2, node1)] = weight

    def number_of_nodes(self,):
        return len(self.graph)

    def has_edge(self, src, dst):
        return dst in self.graph[src] 

    def get_weight(self,):
        total = 0
        for node1 in self.graph:
            for node2 in self.graph[node1]:
                total += self.weight[(node1, node2)]
                
        # because it is undirected
        return total/2
    
#Part 2 starts from here
#Part 2.1:
    
def dijkstra(graph, source, k, track_relaxations=False):
    """
    Implementation of Dijkstra's algorithm variation where each node can be relaxed at most k times.

    Returns:
        distances: Dictionary mapping each node to its shortest distance from source
        paths: Dictionary mapping each node to the path from source
    """
    
    # Initialize distances from source to all nodes as infinity
    distances = {node: float('infinity') for node in graph}
    distances[source] = 0

    # Store the actual paths from the source to each node
    paths = {node: [] for node in graph}
    paths[source] = [source]

    # Count how many times each node has been relaxed
    relaxation_count = {node: 0 for node in graph}
    total_relaxations = 0

    # Min-heap priority queue to determine next closest node to process
    priority_queue = [(0, source)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        # Skip outdated entries in the priority queue
        if current_distance > distances[current_node]:
            continue

        # Explore all neighboring nodes
        for neighbor, weight in graph[current_node].items():
            # Enforce the relaxation limit
            if relaxation_count[neighbor] >= k:
                continue

            # Calculate tentative distance to neighbor
            distance = current_distance + weight

            # If this path is better, update the distance and path
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                paths[neighbor] = paths[current_node] + [neighbor]
                relaxation_count[neighbor] += 1
                total_relaxations += 1
                heapq.heappush(priority_queue, (distance, neighbor))

    if track_relaxations:
        return distances, paths, total_relaxations
    return distances, paths

# Part 2.2:
def bellman_ford(graph, source, k, track_relaxations=False):
    """
    Implementation of Bellman-Ford algorithm variation where each node can be relaxed at most k times.

    Returns:
        distances: Dictionary mapping each node to its shortest distance from source
        paths: Dictionary mapping each node to the path from source
    """
    # Initialize distances and paths
    distances = {node: float('infinity') for node in graph}
    distances[source] = 0
    paths = {node: [] for node in graph}
    paths[source] = [source]

    # Keep track of how many times each node has been relaxed
    relaxation_count = {node: 0 for node in graph}
    total_relaxations = 0

    # Collect all edges in the graph
    edges = [(u, v, w) for u in graph for v, w in graph[u].items()]

    # Bellman-Ford runs up to (V - 1) iterations
    for _ in range(len(graph) - 1):
        no_changes = True  # Flag to detect if we can terminate early

        for u, v, weight in edges:
            # Skip if node 'v' has already been relaxed k times
            if relaxation_count[v] >= k:
                continue

            # Relax edge if it offers a shorter path
            if distances[u] != float('infinity') and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                paths[v] = paths[u] + [v]
                relaxation_count[v] += 1
                total_relaxations += 1
                no_changes = False  # A change occurred this round

        # If no relaxations happened, early termination is possible
        if no_changes:
            break

    if track_relaxations:
        return distances, paths, total_relaxations
    return distances, paths

# Part 2.3:
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

def draw_plot(x_values, dijkstra_values, bellman_values, x_label, y_label, title, y_limit=None):
    x = np.array(x_values)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x, dijkstra_values, 'o-', label="Dijkstra's Algorithm", linewidth=2, markersize=8)
    ax.plot(x, bellman_values, 's--', label="Bellman-Ford Algorithm", linewidth=2, markersize=8)
    ax.axhline(np.mean(dijkstra_values), color="blue", linestyle=":", linewidth=1.5, label="Dijkstra Avg")
    ax.axhline(np.mean(bellman_values), color="green", linestyle=":", linewidth=1.5, label="Bellman Avg")
    for i, (dx, bx) in enumerate(zip(dijkstra_values, bellman_values)):
        ax.annotate(f"{dx:.2f}", (x[i], dx), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=9)
        ax.annotate(f"{bx:.2f}", (x[i], bx), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14)
    if y_limit:
        ax.set_ylim(*y_limit)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.show()

def run_experiment():
    graph_sizes = [10, 50, 100, 200]
    densities = [0.1, 0.3, 0.5, 0.7]
    k_values = [1, 3, 5, 10]

    results = {
        'dijkstra': defaultdict(list),
        'bellman_ford': defaultdict(list)
    }

    # Experiment 1: Vary Graph Size
    print("Experiment 1: Varying Graph Size")
    for size in graph_sizes:
        graph = generate_random_graph(size, 0.5)
        source = 0
        k = 3
        d_time, d_relax = measure_performance(graph, source, k, dijkstra)
        b_time, b_relax = measure_performance(graph, source, k, bellman_ford)
        results['dijkstra']['size_time'].append(d_time)
        results['dijkstra']['size_relax'].append(d_relax)
        results['bellman_ford']['size_time'].append(b_time)
        results['bellman_ford']['size_relax'].append(b_relax)
        print(f"Size {size}: Dijkstra {d_time:.5f}s, {d_relax} relax | Bellman-Ford {b_time:.5f}s, {b_relax} relax")

    # Experiment 2: Vary Graph Density
    print("\nExperiment 2: Varying Graph Density")
    for density in densities:
        graph = generate_random_graph(100, density)
        source = 0
        k = 3
        d_time, d_relax = measure_performance(graph, source, k, dijkstra)
        b_time, b_relax = measure_performance(graph, source, k, bellman_ford)
        results['dijkstra']['density_time'].append(d_time)
        results['dijkstra']['density_relax'].append(d_relax)
        results['bellman_ford']['density_time'].append(b_time)
        results['bellman_ford']['density_relax'].append(b_relax)
        print(f"Density {density:.1f}: Dijkstra {d_time:.5f}s, {d_relax} relax | Bellman-Ford {b_time:.5f}s, {b_relax} relax")

    # Experiment 3: Vary Relaxation Limit k
    print("\nExperiment 3: Varying k Value")
    for k in k_values:
        graph = generate_random_graph(100, 0.5)
        source = 0
        d_time, d_relax = measure_performance(graph, source, k, dijkstra)
        b_time, b_relax = measure_performance(graph, source, k, bellman_ford)
        results['dijkstra']['k_time'].append(d_time)
        results['dijkstra']['k_relax'].append(d_relax)
        results['bellman_ford']['k_time'].append(b_time)
        results['bellman_ford']['k_relax'].append(b_relax)
        print(f"k={k}: Dijkstra {d_time:.5f}s, {d_relax} relax | Bellman-Ford {b_time:.5f}s, {b_relax} relax")

    # Plotting
    draw_plot(graph_sizes, results['dijkstra']['size_time'], results['bellman_ford']['size_time'],
              "Graph Size (nodes)", "Execution Time (s)", "Execution Time vs Graph Size (k=3, density=0.5)")

    draw_plot(densities, results['dijkstra']['density_time'], results['bellman_ford']['density_time'],
              "Graph Density", "Execution Time (s)", "Execution Time vs Graph Density (size=100, k=3)")

    draw_plot(k_values, results['dijkstra']['k_time'], results['bellman_ford']['k_time'],
              "k Value", "Execution Time (s)", "Execution Time vs k Value (size=100, density=0.5)")

    draw_plot(k_values, results['dijkstra']['k_relax'], results['bellman_ford']['k_relax'],
              "k Value", "Relaxation Count", "Total Relaxations vs k Value (size=100, density=0.5)")
    

def a_star(graph: Dgraph, start, goal, heuristic):
    # Min-heap priority queue
    open_set = []
    heapq.heappush(open_set, (heuristic.get(start, 0), start))

    # Stores the best-known path
    came_from = {}

    # gScore: Cost from start node to the current node
    g_score = {node: float('inf') for node in graph.graph.keys()}
    g_score[start] = 0

    # fScore: Estimated cost from start to goal via current node
    f_score = {node: float('inf') for node in graph.graph.keys()}
    f_score[start] = heuristic.get(start, 0)

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # Reverse to get the correct order

        for neighbor in graph.connected_nodes(current):
            edge_weight = graph.weight.get((current, neighbor), float('inf'))
            tentative_g_score = g_score[current] + edge_weight

            if tentative_g_score < g_score[neighbor]:  # Found a better path
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic.get(neighbor, 1000)

                if neighbor not in [node[1] for node in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []

#dijkstra for part5

def dijkstra2(graph: Dgraph, start, goal):
    # Min-heap priority queue
    open_set = []
    heapq.heappush(open_set, (0, start))  # (cost, node)

    # Stores the best-known path
    came_from = {}

    # gScore: Cost from start node to the current node
    g_score = {node: float('inf') for node in graph.graph.keys()}
    g_score[start] = 0

    while open_set:
        current_cost, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # Reverse to get the correct order

        for neighbor in graph.connected_nodes(current):
            edge_weight = graph.weight.get((current, neighbor), float('inf'))
            tentative_g_score = current_cost + edge_weight

            if tentative_g_score < g_score[neighbor]:  # Found a better path
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score

                heapq.heappush(open_set, (tentative_g_score, neighbor))

    return []  # Return empty path if goal was never reached

def euclidean_distance(pos1, pos2):
    # Calculate the straight line distence
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def load_stations(file_name):
    station_positions = {}
    #opens the CSV file with proper handling of newlines and character encoding
    with open(file_name, newline='', encoding='utf-8') as csvfile:
        #create the reader 
        reader = csv.DictReader(csvfile)
        #read each entity and store the core attribute
        for row in reader:
            # Convert station id to int and latitude/longitude to float.
            station_id = int(row['id'])
            lat = float(row['latitude'])
            lon = float(row['longitude'])
            station_positions[station_id] = (lat, lon)
    return station_positions

def load_connections(file_name):
    connections = []
    #opens the CSV file with proper handling of newlines and character encoding
    with open(file_name, newline='', encoding='utf-8') as csvfile:
        #create the reader 
        reader = csv.DictReader(csvfile)
        #read each entity and store the core attribute
        for row in reader:
            station1 = int(row['station1'])
            station2 = int(row['station2'])
            line = int(row['line'])
            connections.append((station1, station2,line))
    return connections


def build_graph(stations_file, connections_file):
    #Read the csv data base
    station_positions = load_stations(stations_file)
    station_connections = load_connections(connections_file)
    
    # Number of nodes is determined by the maximum station id plus one.
    num_nodes = max(station_positions.keys()) + 1
    graph = Dgraph(num_nodes)
    
    # Add edges: compute weight using the Euclidean distance between station positions.
    for station1, station2,_ in station_connections:
        weight = euclidean_distance(station_positions[station1], station_positions[station2]) 
        graph.add_edge(station1, station2, weight)
    
    return graph, station_positions, station_connections


def heuristic_function(station_positions:dict, destination):
    heuristic = {}
    dest_pos = station_positions[destination]
    for station, pos in station_positions.items():
        heuristic[station] = euclidean_distance(pos, dest_pos)
    return heuristic

def build_connection_lookup(station_connections):
    # Build a dictionary mapping (station1, station2) to line number.
    # Assumes the network is undirected.
    lookup = {}
    for station1, station2, line in station_connections:
        lookup[(station1, station2)] = line
        lookup[(station2, station1)] = line
    return lookup

def count_transfers(path, connection_lookup):
    if not path or len(path) < 2:
        return 0  # No journey, no transfers.
    
    # Get the line for the first segment.
    current_line = connection_lookup.get((path[0], path[1]))
    transfers = 0
    # Iterate over segments
    for i in range(1, len(path) - 1):
        segment_line = connection_lookup.get((path[i], path[i+1]))
        if segment_line != current_line:
            transfers += 1
            current_line = segment_line
    return transfers

def experiment_part_5 ():
    stations_file = "london_stations.csv"
    connections_file = "london_connections.csv"
    
    # Build graph, station_positions and connections data
    graph, station_positions, station_connections = build_graph(stations_file, connections_file)
    stations = list(station_positions.keys())
    
    # Build the connection lookup from station_connections for transfer counting.
    connection_lookup = build_connection_lookup(station_connections)
    
    # Containers to accumulate times for each transfer category.
    timings = {
        "same_line": {"dijkstra_times": [], "astar_times": []},
        "one_transfer": {"dijkstra_times": [], "astar_times": []},
        "multiple_transfers": {"dijkstra_times": [], "astar_times": []}
    }
    
    # Store detailed results for optional analysis.
    results = []
    
    for source in stations:
        for destination in stations:
            if source == destination:
                continue
            
            # Time Dijkstra (modified to stop when destination is reached)
            start = time.perf_counter()
            dijkstra_path = dijkstra2(graph, source, destination)
            dijkstra_time = time.perf_counter() - start
            
            # Build heuristic for current destination
            heuristic = heuristic_function(station_positions, destination)
            start = time.perf_counter()
            astar_path = a_star(graph, source, destination, heuristic)
            astar_time = time.perf_counter() - start
            
            # Count the number of transfers on the path (using dijkstra_path, or astar_path if they are equal)
            transfers = count_transfers(dijkstra_path, connection_lookup)
            
            # Categorize based on transfer count:
            if transfers == 0:
                category = "same_line"
            elif transfers == 1:
                category = "one_transfer"
            else:
                category = "multiple_transfers"
            
            # Append times to the corresponding category.
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
    
    # Compute average times for each category.
    avg_results = {}
    for cat, times in timings.items():
        avg_dij = sum(times["dijkstra_times"]) / len(times["dijkstra_times"]) if times["dijkstra_times"] else 0
        avg_astar = sum(times["astar_times"]) / len(times["astar_times"]) if times["astar_times"] else 0
        avg_results[cat] = {"avg_dijkstra_time": avg_dij, "avg_astar_time": avg_astar}
    
    print("\nAverage Running Times by Transfer Category:")
    for cat, averages in avg_results.items():
        print(f"{cat.capitalize()}: Dijkstra = {averages['avg_dijkstra_time']:.6f}s, A* = {averages['avg_astar_time']:.6f}s")
 
    

if __name__ == "__main__":
    # print(load_connections("london_connections.csv"))
    # print(load_stations("london_stations.csv"))
    # print(heuristic_function(load_stations("london_stations.csv"),50))
    experiment_part_5()
    # run_experiment()




