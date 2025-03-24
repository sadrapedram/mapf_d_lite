import osmnx as ox
import networkx as nx
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

class TrafficSimulation:
    def __init__(self, place="Vake, Tbilisi, Georgia", num_cars=50, speed=10.0):
        """Initialize the traffic simulation. Speed in meters per frame (100ms)."""
        self.place = place
        self.num_cars = num_cars
        self.base_speed = speed  # Base speed in meters per frame
        self.running = True
        
        # Load and prepare the directed graph
        self._load_graph()
        
        # Initialize car properties
        self.car_positions = [random.choice(list(self.city_graph.nodes)) for _ in range(num_cars)]
        self.traffic_flow = {}  # Directional traffic flow, initialized dynamically
        self.car_visit_history = [deque(maxlen=4) for _ in range(num_cars)]
        self.previous_positions = self.car_positions.copy()
        self.current_edges = [None] * num_cars
        self.progress = [0.0] * num_cars
        for i, pos in enumerate(self.car_positions):
            self.car_visit_history[i].append(pos)
        
        # Node positions for plotting
        self.positions = {node: (data['x'], data['y']) for node, data in self.city_graph.nodes(data=True)}
        if not self.positions:
            raise ValueError("No position data extracted from graph!")
        
        # Set up the plot
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.mpl_connect('close_event', self._on_close)

    def _load_graph(self):
        """Load the road network graph as a directed graph with length attributes."""
        print(f"Loading {self.place}...")
        try:
            graph = ox.graph_from_place(self.place, network_type="drive")
            graph = ox.distance.add_edge_lengths(graph)
            self.city_graph = nx.DiGraph(graph)  # Use directed graph for separate lanes
            if not self.city_graph.nodes or not self.city_graph.edges:
                raise ValueError("City graph is empty!")
            # Note: No connectivity check for DiGraph, as directions matter
        except Exception as e:
            print(f"Error loading graph: {e}")
            exit(1)

    def _on_close(self, event):
        """Handle window close event to stop the simulation."""
        self.running = False

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
        
        # Update car positions and traffic flow
        car_x, car_y = [], []
        for i in range(self.num_cars):
            current_node = self.car_positions[i]
            previous_node = self.previous_positions[i]
            
            # If not currently traversing an edge, choose a new one
            if self.current_edges[i] is None:
                neighbors = list(self.city_graph.successors(current_node))  # Directed successors
                if neighbors:
                    if len(self.car_visit_history[i]) >= 3:
                        forbidden_nodes = set(self.car_visit_history[i])
                        valid_neighbors = [n for n in neighbors if n not in forbidden_nodes] or neighbors
                    else:
                        valid_neighbors = neighbors
                    
                    next_node = random.choice(valid_neighbors)
                    edge_to = (current_node, next_node)
                    if edge_to not in self.city_graph.edges:
                        continue  # Skip if edge doesn’t exist (shouldn’t happen with successors)
                    self.current_edges[i] = edge_to
                    self.progress[i] = 0.0
                    
                    # Update traffic flow when entering the edge
                    old_flow_to = self.traffic_flow.get(edge_to, 0)
                    self.traffic_flow[edge_to] = old_flow_to + 1
                    
                    # Decrease traffic flow on the edge being left (if not starting frame)
                    edge_from = (previous_node, current_node)
                    old_flow_from = self.traffic_flow.get(edge_from, 0)
                    if frame > 0 and old_flow_from > 0 and edge_from in self.city_graph.edges:
                        self.traffic_flow[edge_from] = old_flow_from - 1
                    
                    self.previous_positions[i] = current_node
                    left_info = (f"Left {edge_from}: {old_flow_from} -> {self.traffic_flow.get(edge_from, 0)}"
                                 if frame > 0 and edge_from in self.city_graph.edges else "Initial move")
                    print(f"Frame {frame} - Car {i}: Starting {edge_to}, Traffic {edge_to}: {old_flow_to} -> {self.traffic_flow[edge_to]}, {left_info}")
            
            # Update progress along the current edge
            if self.current_edges[i]:
                edge = self.current_edges[i]
                length = self.city_graph.edges[edge]['length']
                traffic_flow = self.traffic_flow.get(edge, 0)
                
                # Adjust speed based on directional traffic flow
                speed_factor = max(0.1, 1.0 / (1.0 + traffic_flow * 0.5))  # Slows down with traffic in this direction
                adjusted_speed = self.base_speed * speed_factor
                frames_needed = length / adjusted_speed
                self.progress[i] += 1.0 / frames_needed
                
                # Interpolate position along the edge
                start_x, start_y = self.positions[edge[0]]
                end_x, end_y = self.positions[edge[1]]
                interp_x = start_x + (end_x - start_x) * self.progress[i]
                interp_y = start_y + (end_y - start_y) * self.progress[i]
                car_x.append(interp_x)
                car_y.append(interp_y)
                
                # If progress reaches 1, move to the next node
                if self.progress[i] >= 1.0:
                    self.car_positions[i] = edge[1]
                    self.current_edges[i] = None
                    self.car_visit_history[i].append(edge[1])
                    print(f"Frame {frame} - Car {i}: Arrived at {edge[1]} from {edge}, Traffic: {traffic_flow}")
        
        # Visualize traffic flow with directional consideration
        for edge, flow in self.traffic_flow.items():
            if flow > 0:
                x = [self.positions[edge[0]][0], self.positions[edge[1]][0]]
                y = [self.positions[edge[0]][1], self.positions[edge[1]][1]]
                self.ax.plot(x, y, color=plt.cm.Reds(min(1.0, flow * 0.1)), linewidth=min(3.0, flow * 0.3), zorder=2)
        
        # Plot cars with reduced size and fixed blue color
        if car_x and car_y:
            self.ax.scatter(car_x, car_y, c='blue', s=20, zorder=5, edgecolors='black')
        
        # Set axis limits
        x_coords = [pos[0] for pos in self.positions.values()]
        y_coords = [pos[1] for pos in self.positions.values()]
        self.ax.set_xlim(min(x_coords) - 0.005, max(x_coords) + 0.005)
        self.ax.set_ylim(min(y_coords) - 0.005, max(y_coords) + 0.005)
        
        self.ax.set_title(f"{self.place} Traffic Simulation - Frame {frame}")
        return self.ax,

    def run(self):
        """Start the simulation."""
        print(f"Starting simulation... Close the window to stop.")
        self.anim = FuncAnimation(self.fig, self.update, frames=None, interval=100, repeat=False, cache_frame_data=False)
        plt.show()
        print("Simulation stopped.")

# Run the simulation
if __name__ == "__main__":
    sim = TrafficSimulation(place="Vake, Tbilisi, Georgia", num_cars=500, speed=30.0)
    sim.run()