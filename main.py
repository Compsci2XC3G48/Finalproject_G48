import heapq

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

# Example usage:
if __name__ == "__main__":
    # Create a graph with 5 nodes
    g = Dgraph(5)
    
    # Add edges (undirected)
    g.add_edge(0, 1, 2)
    g.add_edge(0, 2, 4)
    g.add_edge(1, 2, 1)
    g.add_edge(1, 3, 7)
    g.add_edge(2, 4, 3)
    g.add_edge(3, 4, 1)
    
    # Define a heuristic function as a dictionary.
    # For this example, the heuristic values are arbitrarily chosen.
    heuristic = {0: 7, 1: 6, 2: 2, 3: 1, 4: 0}
    
    pred, path = a_star(g, 0, 4, heuristic)
    print("Predecessors:", pred)
    print("Shortest path from 0 to 4:", path)