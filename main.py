import csv
import heapq
import math
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, deque

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
def dijkstra(graph, source, k):
    """
    Implementation of Dijkstra's algorithm variation where each node can be relaxed at most k times.

    Returns:
        distances: Dictionary mapping each node to its shortest distance from source
        paths: Dictionary mapping each node to the path from source
    """
    # Initialize distances with infinity for all nodes except the source
    distances = {node: float('infinity') for node in graph}
    distances[source] = 0
    
    # Initialize paths dictionary to store the shortest path to each node
    paths = {node: [] for node in graph}
    paths[source] = [source]
    
    # Initialize a dictionary to track relaxation count for each node
    relaxation_count = {node: 0 for node in graph}
    
    # Initialize priority queue with (distance, node)
    priority_queue = [(0, source)]
    
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        
        # Skip if we've found a better path since this entry was added
        if current_distance > distances[current_node]:
            continue
            
        # Process neighbors
        for neighbor, weight in graph[current_node].items():
            # Skip if node has already been relaxed k times
            if relaxation_count[neighbor] >= k:
                continue
                
            distance = current_distance + weight
            
            # If we found a shorter path, update distance and path
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                paths[neighbor] = paths[current_node] + [neighbor]
                relaxation_count[neighbor] += 1
                heapq.heappush(priority_queue, (distance, neighbor))
    
    return distances, paths

# Part 2.2:
def bellman_ford(graph, source, k):
    """
    Implementation of Bellman-Ford algorithm variation where each node can be relaxed at most k times.

    Returns:
        distances: Dictionary mapping each node to its shortest distance from source
        paths: Dictionary mapping each node to the path from source
    """
    # Initialize distances with infinity for all nodes except the source
    distances = {node: float('infinity') for node in graph}
    distances[source] = 0
    
    # Initialize paths dictionary to store the shortest path to each node
    paths = {node: [] for node in graph}
    paths[source] = [source]
    
    # Initialize a dictionary to track relaxation count for each node
    relaxation_count = {node: 0 for node in graph}
    
    # Get all edges from the graph
    edges = []
    for node in graph:
        for neighbor, weight in graph[node].items():
            edges.append((node, neighbor, weight))
    
    # Relax all edges repeatedly
    for _ in range(len(graph) - 1):  # In the worst case, we need N-1 iterations
        no_changes = True
        for u, v, weight in edges:
            # Skip if node has already been relaxed k times
            if relaxation_count[v] >= k:
                continue
                
            if distances[u] != float('infinity') and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                paths[v] = paths[u] + [v]
                relaxation_count[v] += 1
                no_changes = False
        
        # If no changes were made in this iteration, we can break early
        if no_changes:
            break
    
    return distances, paths

# Part 2.3:
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

def run_experiment():
    """
    Run the performance analysis experiment comparing both algorithms.
    """
    # Parameters to vary
    graph_sizes = [10, 50, 100, 200]
    densities = [0.1, 0.3, 0.5, 0.7]
    k_values = [1, 3, 5, 10]
    
    # Results storage
    results = {
        'dijkstra': defaultdict(list),
        'bellman_ford': defaultdict(list)
    }
    
    # 1. Vary graph size (fixed density=0.5, k=3)
    print("Experiment 1: Varying Graph Size")
    for size in graph_sizes:
        graph = generate_random_graph(size, 0.5)
        source = 0
        k = 3
        
        dijkstra_time, dijkstra_acc = measure_performance(graph, source, k, dijkstra)
        bellman_time, bellman_acc = measure_performance(graph, source, k, bellman_ford)
        
        results['dijkstra']['size_time'].append(dijkstra_time)
        results['dijkstra']['size_acc'].append(dijkstra_acc)
        results['bellman_ford']['size_time'].append(bellman_time)
        results['bellman_ford']['size_acc'].append(bellman_acc)
        
        print(f"Size {size}: Dijkstra {dijkstra_time:.5f}s, {dijkstra_acc:.2f}% | Bellman-Ford {bellman_time:.5f}s, {bellman_acc:.2f}%")
    
    # 2. Vary graph density (fixed size=100, k=3)
    print("\nExperiment 2: Varying Graph Density")
    for density in densities:
        graph = generate_random_graph(100, density)
        source = 0
        k = 3
        
        dijkstra_time, dijkstra_acc = measure_performance(graph, source, k, dijkstra)
        bellman_time, bellman_acc = measure_performance(graph, source, k, bellman_ford)
        
        results['dijkstra']['density_time'].append(dijkstra_time)
        results['dijkstra']['density_acc'].append(dijkstra_acc)
        results['bellman_ford']['density_time'].append(bellman_time)
        results['bellman_ford']['density_acc'].append(bellman_acc)
        
        print(f"Density {density}: Dijkstra {dijkstra_time:.5f}s, {dijkstra_acc:.2f}% | Bellman-Ford {bellman_time:.5f}s, {bellman_acc:.2f}%")
    
    # 3. Vary k value (fixed size=100, density=0.5)
    print("\nExperiment 3: Varying k Value")
    for k in k_values:
        graph = generate_random_graph(100, 0.5)
        source = 0
        
        dijkstra_time, dijkstra_acc = measure_performance(graph, source, k, dijkstra)
        bellman_time, bellman_acc = measure_performance(graph, source, k, bellman_ford)
        
        results['dijkstra']['k_time'].append(dijkstra_time)
        results['dijkstra']['k_acc'].append(dijkstra_acc)
        results['bellman_ford']['k_time'].append(bellman_time)
        results['bellman_ford']['k_acc'].append(bellman_acc)
        
        print(f"k={k}: Dijkstra {dijkstra_time:.5f}s, {dijkstra_acc:.2f}% | Bellman-Ford {bellman_time:.5f}s, {bellman_acc:.2f}%")
    
    # Display four separate plots
    
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




