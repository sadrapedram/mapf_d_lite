import osmnx as ox
import networkx as nx
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

class TrafficSimulation:
    def __init__(self, place="Vake, Tbilisi, Georgia", num_cars=50, speed=10.0, replan_interval=10):
        """Initialize the traffic simulation. Speed in meters per frame (100ms)."""
        self.place = place
        self.num_cars = num_cars
        self.base_speed = speed  # Base speed in meters per frame
        self.replan_interval = replan_interval  # Frames between replanning
        self.running = True
        
        # Load the road network graph
        self._load_graph()
        # Extract node positions (longitude, latitude)
        self.positions = {node: (data['x'], data['y']) for node, data in self.city_graph.nodes(data=True)}
        if not self.positions:
            raise ValueError("No position data extracted from graph!")
        
        # Initialize car positions and simulation state
        self.car_positions = [random.choice(list(self.city_graph.nodes)) for _ in range(num_cars)]
        self.traffic_flow = {}  # Tracks number of cars on each edge
        self.car_destinations = [self.get_distant_node(pos) for pos in self.car_positions]
        self.car_planned_paths = [[] for _ in range(num_cars)]
        self.car_path_indices = [0] * num_cars
        self.current_edges = [None] * num_cars
        self.progress = [0.0] * num_cars  # Progress along current edge (0 to 1)
        self.replan_counters = [random.randint(0, self.replan_interval - 1) for _ in range(num_cars)]
        
        # Plan initial paths for all cars
        for i in range(self.num_cars):
            self.plan_path(i)
        
        # Set up the plot
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.mpl_connect('close_event', self._on_close)

    def _load_graph(self):
        """Load the road network graph as a directed graph with length attributes."""
        print(f"Loading {self.place}...")
        try:
            graph = ox.graph_from_place(self.place, network_type="drive")
            graph = ox.distance.add_edge_lengths(graph)
            self.city_graph = nx.DiGraph(graph)
            if not self.city_graph.nodes or not self.city_graph.edges:
                raise ValueError("City graph is empty!")
        except Exception as e:
            print(f"Error loading graph: {e}")
            exit(1)

    def _on_close(self, event):
        """Handle window close event to stop the simulation."""
        self.running = False

    def get_distant_node(self, start, min_depth=10):
        """Select a node at least min_depth nodes away from start using BFS, ensuring it's reachable."""
        queue = deque([(start, 0)])
        visited = set([start])
        distant_nodes = []
        reachable_nodes = []
        while queue:
            node, depth = queue.popleft()
            if node != start:
                reachable_nodes.append(node)
                if depth >= min_depth:
                    distant_nodes.append(node)
            for neighbor in self.city_graph.successors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
        if distant_nodes:
            return random.choice(distant_nodes)
        elif reachable_nodes:
            return random.choice(reachable_nodes)
        else:
            return start  # Stay put if no reachable nodes

    def plan_path(self, i, max_retries=10):
        """Plan a path for car i to its destination using A* with traffic-adjusted costs."""
        current_node = self.car_positions[i]
        destination = self.car_destinations[i]
        
        # If already at destination, pick a new one
        if current_node == destination:
            destination = self.get_distant_node(current_node)
            if destination == current_node:
                self.car_planned_paths[i] = [current_node]
                self.current_edges[i] = None
                return
            self.car_destinations[i] = destination
    
        # Heuristic for A*: Euclidean distance divided by base speed
        def heuristic(u, target):
            pos_u = self.positions[u]
            pos_target = self.positions[target]
            dist = ((pos_u[0] - pos_target[0])**2 + (pos_u[1] - pos_target[1])**2)**0.5
            return dist / self.base_speed

        # Weight function adjusting for traffic
        def weight_func(u, v, d):
            traffic = self.traffic_flow.get((u, v), 0)
            speed_factor = max(0.1, 1.0 / (1.0 + traffic * 0.5))  # Reduce speed with traffic
            adjusted_speed = self.base_speed * speed_factor
            return d['length'] / adjusted_speed

        retries = max_retries
        while retries > 0:
            try:
                path = nx.astar_path(self.city_graph, current_node, destination, 
                                   heuristic=heuristic, weight=weight_func)
                self.car_planned_paths[i] = path
                self.car_path_indices[i] = 0
                if len(path) > 1 and (path[0], path[1]) in self.city_graph.edges:
                    self.current_edges[i] = (path[0], path[1])
                    self.progress[i] = 0.0
                else:
                    self.current_edges[i] = None
                return
            except nx.NetworkXNoPath:
                destination = self.get_distant_node(current_node)
                if destination == current_node:
                    self.car_planned_paths[i] = [current_node]
                    self.current_edges[i] = None
                    return
                self.car_destinations[i] = destination
                retries -= 1
    
        print(f"Warning: Could not find path for car {i} after {max_retries} retries. Staying at {current_node}")
        self.car_planned_paths[i] = [current_node]
        self.current_edges[i] = None

    def plot_base_map(self):
        """Plot the base road network."""
        for edge in self.city_graph.edges:
            x = [self.positions[edge[0]][0], self.positions[edge[1]][0]]
            y = [self.positions[edge[0]][1], self.positions[edge[1]][1]]
            self.ax.plot(x, y, color='gray', linewidth=0.5, zorder=1)

    def update(self, frame):
        """Update the simulation state and visualization for each frame."""
        if not self.running:
            self.anim.event_source.stop()
            return
        
        self.ax.clear()
        self.plot_base_map()
        
        # Replan paths periodically
        for i in range(self.num_cars):
            if self.replan_counters[i] == 0:
                self.plan_path(i)
            self.replan_counters[i] = (self.replan_counters[i] + 1) % self.replan_interval
        
        car_x, car_y = [], []
        for i in range(self.num_cars):
            if self.current_edges[i] is None:
                # Move to next edge if available
                if self.car_path_indices[i] < len(self.car_planned_paths[i]) - 1:
                    self.car_path_indices[i] += 1
                    next_edge = (self.car_planned_paths[i][self.car_path_indices[i]], 
                                 self.car_planned_paths[i][self.car_path_indices[i] + 1])
                    if next_edge in self.city_graph.edges:
                        self.current_edges[i] = next_edge
                        self.progress[i] = 0.0
                        # Update traffic flow
                        if self.car_path_indices[i] > 0:
                            prev_edge = (self.car_planned_paths[i][self.car_path_indices[i] - 1], 
                                         self.car_planned_paths[i][self.car_path_indices[i]])
                            if prev_edge in self.traffic_flow and self.traffic_flow[prev_edge] > 0:
                                self.traffic_flow[prev_edge] -= 1
                        self.traffic_flow[next_edge] = self.traffic_flow.get(next_edge, 0) + 1
                    else:
                        self.plan_path(i)
                else:
                    # Reached end of path, set new destination
                    self.car_destinations[i] = self.get_distant_node(self.car_positions[i])
                    self.plan_path(i)
            else:
                # Move along current edge
                edge = self.current_edges[i]
                length = self.city_graph.edges[edge]['length']
                traffic_flow = self.traffic_flow.get(edge, 0)
                speed_factor = max(0.1, 1.0 / (1.0 + traffic_flow * 0.5))
                adjusted_speed = self.base_speed * speed_factor
                frames_needed = length / adjusted_speed
                self.progress[i] += 1.0 / frames_needed
                
                if self.progress[i] >= 1.0:
                    self.car_positions[i] = edge[1]
                    self.current_edges[i] = None
                    self.progress[i] = 0.0
                else:
                    # Interpolate position
                    start_x, start_y = self.positions[edge[0]]
                    end_x, end_y = self.positions[edge[1]]
                    interp_x = start_x + (end_x - start_x) * self.progress[i]
                    interp_y = start_y + (end_y - start_y) * self.progress[i]
                    car_x.append(interp_x)
                    car_y.append(interp_y)
        
        # Visualize traffic flow
        for edge, flow in self.traffic_flow.items():
            if flow > 0:
                x = [self.positions[edge[0]][0], self.positions[edge[1]][0]]
                y = [self.positions[edge[0]][1], self.positions[edge[1]][1]]
                self.ax.plot(x, y, color=plt.cm.Reds(min(1.0, flow * 0.1)), 
                            linewidth=min(3.0, flow * 0.3), zorder=2)
        
        # Plot cars
        if car_x and car_y:
            self.ax.scatter(car_x, car_y, c='blue', s=20, zorder=5, edgecolors='black')
        
        # Set plot limits
        x_coords = [pos[0] for pos in self.positions.values()]
        y_coords = [pos[1] for pos in self.positions.values()]
        self.ax.set_xlim(min(x_coords) - 0.005, max(x_coords) + 0.005)
        self.ax.set_ylim(min(y_coords) - 0.005, max(y_coords) + 0.005)
        
        self.ax.set_title(f"{self.place} Traffic Simulation - Frame {frame}")
        return self.ax,

    def run(self):
        """Start the simulation."""
        print(f"Starting simulation... Close the window to stop.")
        self.anim = FuncAnimation(self.fig, self.update, frames=None, interval=100, 
                                repeat=False, cache_frame_data=False)
        plt.show()
        print("Simulation stopped.")

if __name__ == "__main__":
    # Create and run the simulation with 500 cars
    sim = TrafficSimulation(place="Vake, Tbilisi, Georgia", num_cars=500, speed=30.0)
    sim.run()