import osmnx as ox
import networkx as nx
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from heapq import heappush, heappop
import numpy as np

class TrafficSimulation:
    def __init__(self, place="Vake, Tbilisi, Georgia", num_cars=50, speed=10.0):
        """Initialize the traffic simulation. Speed in meters per frame (100ms)."""
        self.place = place
        self.num_cars = num_cars
        self.base_speed = speed
        self.running = True
        
        # Load and prepare the directed graph
        self._load_graph()
        
        # Node positions for plotting
        self.positions = {node: (data['x'], data['y']) for node, data in self.city_graph.nodes(data=True)}
        
        # Traffic tracking
        self.traffic_flow = {}  # Dictionary to track number of cars per edge
        
        # Initialize car properties
        self.car_starts = []
        self.car_ends = []
        self.car_positions = []
        self.car_paths = [[] for _ in range(num_cars)]
        self.car_colors = self._generate_unique_colors(num_cars)
        self.car_active = [True] * num_cars  # Track which cars are still moving
        self._assign_start_end_points()
        
        # Compute initial paths
        self._compute_initial_paths()
        
        self.current_edges = [None] * num_cars
        self.progress = [0.0] * num_cars
        self.path_index = [0] * num_cars
        
        # Set up the plot
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.mpl_connect('close_event', self._on_close)

    def _load_graph(self):
        """Load the road network graph as a directed graph with length attributes."""
        print(f"Loading {self.place}...")
        graph = ox.graph_from_place(self.place, network_type="drive")
        graph = ox.distance.add_edge_lengths(graph)
        self.city_graph = nx.DiGraph(graph)

    def _generate_unique_colors(self, num_colors):
        """Generate a list of unique colors for the given number of cars."""
        base_colors = [plt.cm.tab20(i) for i in range(20)]
        if num_colors <= 20:
            return base_colors[:num_colors]
        else:
            extra_colors = [plt.cm.tab20b(i) for i in range(20)]
            colors = base_colors + extra_colors
            if num_colors <= 40:
                return colors[:num_colors]
            else:
                additional_colors = []
                for _ in range(num_colors - 40):
                    while True:
                        new_color = (random.random(), random.random(), random.random(), 1.0)
                        if all(np.linalg.norm(np.array(new_color) - np.array(c)) > 0.1 for c in colors + additional_colors):
                            additional_colors.append(new_color)
                            break
                return colors + additional_colors

    def _on_close(self, event):
        """Handle window close event to stop the simulation."""
        self.running = False

    def _assign_start_end_points(self):
        """Assign random start and end points that are far apart."""
        nodes = list(self.city_graph.nodes)
        for _ in range(self.num_cars):
            while True:
                start = random.choice(nodes)
                end = random.choice(nodes)
                if start != end:
                    try:
                        length = nx.shortest_path_length(self.city_graph, start, end, weight='length')
                        if length > 1000:
                            self.car_starts.append(start)
                            self.car_ends.append(end)
                            self.car_positions.append(start)
                            break
                    except nx.NetworkXNoPath:
                        continue

    def _d_star(self, start, goal):
        """D* algorithm with traffic weight consideration."""
        open_list = []
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        came_from = {}
        heap = [(f_score[start], start)]
        
        while heap:
            _, current = heappop(heap)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for neighbor in self.city_graph.successors(current):
                edge = (current, neighbor)
                base_cost = self.city_graph.edges[edge]['length']
                traffic_count = self.traffic_flow.get(edge, 0)
                traffic_penalty = traffic_count * 100  # Adjust this multiplier as needed
                total_cost = base_cost + traffic_penalty
                
                tentative_g = g_score[current] + total_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    heappush(heap, (f_score[neighbor], neighbor))
        return []

    def _heuristic(self, node1, node2):
        """Euclidean distance heuristic."""
        x1, y1 = self.positions[node1]
        x2, y2 = self.positions[node2]
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def _compute_initial_paths(self):
        """Compute initial D* paths for all cars."""
        for i in range(self.num_cars):
            path = self._d_star(self.car_starts[i], self.car_ends[i])
            if path:
                self.car_paths[i] = path
            else:
                print(f"Car {i}: No path found from {self.car_starts[i]} to {self.car_ends[i]}")

    def plot_base_map(self):
        """Plot the base road network with traffic visualization."""
        for edge in self.city_graph.edges:
            x = [self.positions[edge[0]][0], self.positions[edge[1]][0]]
            y = [self.positions[edge[0]][1], self.positions[edge[1]][1]]
            flow = self.traffic_flow.get(edge, 0)
            color = plt.cm.Reds(min(1.0, flow * 0.1)) if flow > 0 else 'gray'
            linewidth = min(3.0, flow * 0.3) if flow > 0 else 0.5
            self.ax.plot(x, y, color=color, linewidth=linewidth, zorder=1)
        
        start_x = [self.positions[s][0] for s in self.car_starts]
        start_y = [self.positions[s][1] for s in self.car_starts]
        self.ax.scatter(start_x, start_y, c=self.car_colors, s=100, marker='^', zorder=4, edgecolors='black')
        
        end_x = [self.positions[e][0] for e in self.car_ends]
        end_y = [self.positions[e][1] for e in self.car_ends]
        self.ax.scatter(end_x, end_y, c=self.car_colors, s=100, marker='s', zorder=4, edgecolors='black')

    def update(self, frame):
        """Update the simulation state and visualization for each frame."""
        if not self.running:
            self.anim.event_source.stop()
            return
        
        self.ax.clear()
        self.plot_base_map()
        
        car_x, car_y, active_car_colors = [], [], []
        
        # Update traffic flow counts
        temp_traffic = {}  # Temporary tracking for current frame
        for i in range(self.num_cars):
            if self.current_edges[i] is not None:
                temp_traffic[self.current_edges[i]] = temp_traffic.get(self.current_edges[i], 0) + 1
        
        self.traffic_flow = temp_traffic  # Update traffic flow
        
        for i in range(self.num_cars):
            # Handle cars that are still moving
            if self.car_active[i]:
                if self.current_edges[i] is None and self.path_index[i] < len(self.car_paths[i]) - 1:
                    current_node = self.car_positions[i]
                    next_node = self.car_paths[i][self.path_index[i] + 1]
                    edge = (current_node, next_node)
                    if edge in self.city_graph.edges:
                        self.current_edges[i] = edge
                        self.progress[i] = 0.0
                
                if self.current_edges[i]:
                    edge = self.current_edges[i]
                    length = self.city_graph.edges[edge]['length']
                    traffic_flow = self.traffic_flow.get(edge, 0)
                    speed_factor = max(0.1, 1.0 / (1.0 + traffic_flow * 0.5))
                    adjusted_speed = self.base_speed * speed_factor
                    frames_needed = length / adjusted_speed
                    self.progress[i] += 1.0 / frames_needed
                    
                    start_x, start_y = self.positions[edge[0]]
                    end_x, end_y = self.positions[edge[1]]
                    interp_x = start_x + (end_x - start_x) * self.progress[i]
                    interp_y = start_y + (end_y - start_y) * self.progress[i]
                    
                    if self.progress[i] >= 1.0:
                        self.car_positions[i] = edge[1]
                        self.current_edges[i] = None
                        self.path_index[i] += 1
                        if self.path_index[i] >= len(self.car_paths[i]) - 1:
                            self.car_active[i] = False  # Mark as reached destination
                        else:
                            # Recalculate path if traffic is heavy
                            if traffic_flow > 3:
                                new_path = self._d_star(self.car_positions[i], self.car_ends[i])
                                if new_path:
                                    self.car_paths[i] = new_path
                                    self.path_index[i] = 0
            
            # Add all cars to visualization (moving or at destination)
            if self.car_active[i] and self.current_edges[i]:  # Moving car
                edge = self.current_edges[i]
                start_x, start_y = self.positions[edge[0]]
                end_x, end_y = self.positions[edge[1]]
                interp_x = start_x + (end_x - start_x) * self.progress[i]
                interp_y = start_y + (end_y - start_y) * self.progress[i]
                car_x.append(interp_x)
                car_y.append(interp_y)
            else:  # Stationary car (at start or end)
                pos_x, pos_y = self.positions[self.car_positions[i]]
                car_x.append(pos_x)
                car_y.append(pos_y)
            active_car_colors.append(self.car_colors[i])
        
        if car_x and car_y:
            self.ax.scatter(car_x, car_y, c=active_car_colors, s=50, zorder=5, edgecolors='black')
        
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

if __name__ == "__main__":
    sim = TrafficSimulation(place="Vake, Tbilisi, Georgia", num_cars=100, speed=30.0)
    sim.run()