import osmnx as ox
import networkx as nx
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
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
        
        # Initialize car properties with start and end points
        self.car_starts = []
        self.car_ends = []
        self.car_positions = []
        self.car_paths = [[] for _ in range(num_cars)]
        self.car_colors = self._generate_unique_colors(num_cars)  # Generate unique colors
        self._assign_start_end_points()
        
        # Compute initial D* paths
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
        # Use tab20 colormap (20 distinct colors)
        base_colors = [plt.cm.tab20(i) for i in range(20)]
        
        if num_colors <= 20:
            return base_colors[:num_colors]
        else:
            # Extend with tab20b if more colors are needed (up to 40)
            extra_colors = [plt.cm.tab20b(i) for i in range(20)]
            colors = base_colors + extra_colors
            if num_colors <= 40:
                return colors[:num_colors]
            else:
                # Generate additional unique colors using random RGB
                additional_colors = []
                for _ in range(num_colors - 40):
                    while True:
                        new_color = (random.random(), random.random(), random.random(), 1.0)  # RGBA
                        # Ensure the new color is sufficiently different from existing ones
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
                        if length > 1000:  # Ensure at least 1km apart
                            self.car_starts.append(start)
                            self.car_ends.append(end)
                            self.car_positions.append(start)
                            break
                    except nx.NetworkXNoPath:
                        continue

    def _d_star(self, start, goal):
        """Simplified D* algorithm for a single car."""
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
                tentative_g = g_score[current] + self.city_graph.edges[edge]['length']
                
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
        """Compute initial D* paths for all cars with basic MAPF conflict avoidance."""
        for i in range(self.num_cars):
            path = self._d_star(self.car_starts[i], self.car_ends[i])
            if path:
                self.car_paths[i] = path
            else:
                print(f"Car {i}: No path found from {self.car_starts[i]} to {self.car_ends[i]}")

    def plot_base_map(self):
        """Plot the base road network and start/end points."""
        for edge in self.city_graph.edges:
            x = [self.positions[edge[0]][0], self.positions[edge[1]][0]]
            y = [self.positions[edge[0]][1], self.positions[edge[1]][1]]
            self.ax.plot(x, y, color='gray', linewidth=0.5, zorder=1)
        
        # Plot start points as triangles
        start_x = [self.positions[s][0] for s in self.car_starts]
        start_y = [self.positions[s][1] for s in self.car_starts]
        self.ax.scatter(start_x, start_y, c=self.car_colors, s=100, marker='^', zorder=4, edgecolors='black')
        
        # Plot end points as squares
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
        active_cars = []  # Track which cars are still moving
        for i in range(self.num_cars):
            if self.path_index[i] >= len(self.car_paths[i]) - 1:
                continue  # Car has reached destination
            
            if self.current_edges[i] is None:
                current_node = self.car_positions[i]
                next_node = self.car_paths[i][self.path_index[i] + 1]
                edge = (current_node, next_node)
                if edge in self.city_graph.edges:
                    self.current_edges[i] = edge
                    self.progress[i] = 0.0
            
            if self.current_edges[i]:
                edge = self.current_edges[i]
                length = self.city_graph.edges[edge]['length']
                adjusted_speed = self.base_speed
                frames_needed = length / adjusted_speed
                self.progress[i] += 1.0 / frames_needed
                
                start_x, start_y = self.positions[edge[0]]
                end_x, end_y = self.positions[edge[1]]
                interp_x = start_x + (end_x - start_x) * self.progress[i]
                interp_y = start_y + (end_y - start_y) * self.progress[i]
                car_x.append(interp_x)
                car_y.append(interp_y)
                active_car_colors.append(self.car_colors[i])  # Use the car's specific color
                active_cars.append(i)
                
                if self.progress[i] >= 1.0:
                    self.car_positions[i] = edge[1]
                    self.current_edges[i] = None
                    self.path_index[i] += 1
        
        # Plot cars as circles with their specific colors
        if car_x and car_y:
            self.ax.scatter(car_x, car_y, c=active_car_colors, s=50, zorder=5, edgecolors='black')
        
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

if __name__ == "__main__":
    sim = TrafficSimulation(place="Vake, Tbilisi, Georgia", num_cars=100, speed=30.0)
    sim.run()