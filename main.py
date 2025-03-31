import csv
import heapq
import math

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
    
def a_star(graph:Dgraph, source, destination, heuristic:dict):
    # Priority queue: stores tuples (f, g, current_node) 
    # Will sort by the first value in the tuple
    open_set = []
    #Heapify the value with min heap
    heapq.heappush(open_set, (heuristic[source], 0, source))
    
    # cost_so_far holds the best-known cost to reach each node
    cost_so_far = {source: 0}
    # predecessor dictionary to reconstruct the path
    predecessor = {source: None}
    
    while open_set:
        _, current_cost, current_node = heapq.heappop(open_set)
        
        # If we reached the destination, reconstruct and return the path.
        if current_node == destination:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = predecessor[current_node]
            return predecessor, path[::-1]  # Reverse the path to make it left to right
        
        # Iterates through all neighbors of the current node.
        for neighbor in graph.connected_nodes(current_node):
            new_cost = current_cost + graph.weight[(current_node, neighbor)]
            # If neighbor is not visited or a better cost is found, update it.
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                predecessor[neighbor] = current_node
                # updates the f
                # f is used for choose a better possible nodes for possible shortest path that improves the efficiency
                f_new = new_cost + heuristic.get(neighbor, float('inf')) #if exiest key neihbor choose it else return infinity
                heapq.heappush(open_set, (f_new, new_cost, neighbor))
    
    # If destination is unreachable, return the predecessor dictionary and an empty path.
    return predecessor, []

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
            weight = float(row['time'])
            connections.append((station1, station2, weight))
    return connections


def build_graph(stations_file, connections_file):
    #Read the csv data base
    station_positions = load_stations(stations_file)
    station_connections = load_connections(connections_file)
    
    # Number of nodes is determined by the maximum station id plus one.
    num_nodes = max(station_positions.keys()) + 1
    graph = Dgraph(num_nodes)
    
    # Add edges from connections.
    for station1, station2, weight in station_connections:
        graph.add_edge(station1, station2, weight)
    
    return graph, station_positions


def heuristic_function(station_positions:dict, destination):
    heuristic = {}
    dest_pos = station_positions[destination]
    for station, pos in station_positions.items():
        heuristic[station] = euclidean_distance(pos, dest_pos)
    return heuristic

def experiment_part_5 ():
    stations_file = "london_stations.csv"
    connections_file = "london_connections.csv"

if __name__ == "__main__":
    # print(load_connections("london_connections.csv"))
    # print(load_stations("london_stations.csv"))
    print(heuristic_function(load_stations("london_stations.csv"),50))
    experiment_part_5()
