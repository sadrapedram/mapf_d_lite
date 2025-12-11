import random
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Reuse graph loading and node positions from the existing simulator
from graph_v5 import TrafficSimulation


Node = int
Path = List[Node]


def euclidean_heuristic(positions: Dict[Node, Tuple[float, float]], a: Node, b: Node) -> float:
    ax, ay = positions[a]
    bx, by = positions[b]
    return float(np.hypot(ax - bx, ay - by))


def astar_path_with_counts(
    G: nx.DiGraph,
    positions: Dict[Node, Tuple[float, float]],
    start: Node,
    goal: Node,
) -> Tuple[Path, int]:
    """
    Standard A* on directed graph with 'length' weights.
    Returns (path, nodes_expanded).
    """
    open_heap: List[Tuple[float, Node]] = []
    import heapq

    g: Dict[Node, float] = {start: 0.0}
    f: Dict[Node, float] = {start: euclidean_heuristic(positions, start, goal)}
    came: Dict[Node, Node] = {}

    heapq.heappush(open_heap, (f[start], start))
    expanded = 0
    closed = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        expanded += 1

        if current == goal:
            # Reconstruct path
            path: Path = [current]
            while current in came:
                current = came[current]
                path.append(current)
            path.reverse()
            return path, expanded

        for nbr in G.successors(current):
            edge_data = G.edges.get((current, nbr))
            if edge_data is None:
                continue
            w = float(edge_data.get("length", 1.0))
            tentative_g = g[current] + w
            if tentative_g < g.get(nbr, float("inf")):
                came[nbr] = current
                g[nbr] = tentative_g
                f[nbr] = tentative_g + euclidean_heuristic(positions, nbr, goal)
                heapq.heappush(open_heap, (f[nbr], nbr))

    return [], expanded


def dfs_path_with_counts(
    G: nx.DiGraph,
    start: Node,
    goal: Node,
    max_expansions: Optional[int] = None,
) -> Tuple[Path, int]:
    """
    Depth-First Search on a directed graph. Returns the first path found and
    the number of nodes expanded. Uses an explicit stack to support large graphs.
    If max_expansions is set and reached, returns ([], expansions).
    """
    # Stack holds (node, iterator over successors)
    stack: List[Tuple[Node, Optional[object]]] = []
    parent: Dict[Node, Node] = {}
    visited: set = set()
    expanded = 0

    stack.append((start, None))
    visited.add(start)

    while stack:
        node, it = stack[-1]
        # Initialize iterator lazily
        if it is None:
            it = iter(G.successors(node))
            stack[-1] = (node, it)
            expanded += 1
            if max_expansions is not None and expanded >= max_expansions:
                return [], expanded

        try:
            nbr = next(it)  # type: ignore[misc]
        except StopIteration:
            stack.pop()
            continue

        if nbr in visited:
            continue
        parent[nbr] = node
        if nbr == goal:
            # Reconstruct path
            path: Path = [nbr]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            return path, expanded

        visited.add(nbr)
        stack.append((nbr, None))

    return [], expanded


def path_length(G: nx.DiGraph, path: Path) -> float:
    if not path or len(path) == 1:
        return 0.0
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        data = G.edges.get((u, v))
        if data is None:
            # If edge data missing, assume unit cost
            total += 1.0
        else:
            total += float(data.get("length", 1.0))
    return total


def pick_far_nodes(
    G: nx.DiGraph, min_length: float = 1000.0, max_tries: int = 5000
) -> Tuple[Node, Node]:
    """Pick start, goal such that both s→t and t→s exist and are long enough."""
    nodes = list(G.nodes)
    best_pair: Optional[Tuple[Node, Node]] = None
    best_min_len = -1.0
    for _ in range(max_tries):
        s = random.choice(nodes)
        t = random.choice(nodes)
        if s == t:
            continue
        try:
            d_st = nx.shortest_path_length(G, s, t, weight="length")
            d_ts = nx.shortest_path_length(G, t, s, weight="length")
            min_bidir = min(d_st, d_ts)
            if min_bidir >= min_length:
                return s, t
            if min_bidir > best_min_len:
                best_min_len = min_bidir
                best_pair = (s, t)
        except nx.NetworkXNoPath:
            continue
    # Fallback: return the best bidirectional pair found even if short
    if best_pair is not None:
        return best_pair
    # Final fallback: any two distinct nodes
    if len(nodes) >= 2:
        return nodes[0], nodes[1]
    return nodes[0], nodes[0]


class AstarVsDFS:
    def __init__(self, place: str = "Vake, Tbilisi, Georgia", seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Reuse TrafficSimulation to load graph + positions, but we won't animate cars
        sim = TrafficSimulation(place=place, num_cars=0, speed=30.0)
        self.G: nx.DiGraph = sim.city_graph
        self.positions: Dict[Node, Tuple[float, float]] = sim.positions
        # Close the simulator's unused figure to avoid extra window
        try:
            import matplotlib.pyplot as _plt
            _plt.close(sim.fig)
        except Exception:
            pass
        self.place = place

    def run_once(self, min_pair_length: float = 1000.0, max_dfs_expansions: Optional[int] = None):
        # Choose a start and end
        s, t = pick_far_nodes(self.G, min_length=min_pair_length)

        # DFS: s -> t
        dfs_path_out, dfs_expanded_out = dfs_path_with_counts(self.G, s, t, max_expansions=max_dfs_expansions)
        # DFS: t -> s (return)
        dfs_path_back, dfs_expanded_back = dfs_path_with_counts(self.G, t, s, max_expansions=max_dfs_expansions)

        # A*: s -> t
        astar_path_out, astar_expanded_out = astar_path_with_counts(self.G, self.positions, s, t)
        # A*: t -> s (return)
        astar_path_back, astar_expanded_back = astar_path_with_counts(self.G, self.positions, t, s)

        # Metrics
        dfs_len_out = path_length(self.G, dfs_path_out)
        dfs_len_back = path_length(self.G, dfs_path_back)
        astar_len_out = path_length(self.G, astar_path_out)
        astar_len_back = path_length(self.G, astar_path_back)

        # Visualization
        self._plot_comparison(
            s,
            t,
            dfs_path_out,
            dfs_path_back,
            astar_path_out,
            astar_path_back,
            dfs_expanded_out + dfs_expanded_back,
            astar_expanded_out + astar_expanded_back,
            dfs_len_out + dfs_len_back,
            astar_len_out + astar_len_back,
        )

    # --- Interactive mode ---
    def run_interactive(
        self,
        min_pair_length: float = 1000.0,
        max_dfs_expansions: Optional[int] = None,
    ) -> None:
        """
        Opens a single window. Press Enter to generate a new random start/end
        and redraw both DFS and A* round trips. Press Esc or close window to stop.
        """
        self._stop = False
        self._min_pair_length = min_pair_length
        self._max_dfs_expansions = max_dfs_expansions

        self._fig, self._ax = plt.subplots(figsize=(10, 10))

        # Initial draw
        self._do_iteration(self._ax)

        # Wire up events
        self._cid_key = self._fig.canvas.mpl_connect('key_press_event', self._on_key)
        self._cid_close = self._fig.canvas.mpl_connect('close_event', self._on_close)

        plt.show()

        # Cleanup (best-effort)
        try:
            self._fig.canvas.mpl_disconnect(self._cid_key)
            self._fig.canvas.mpl_disconnect(self._cid_close)
        except Exception:
            pass

    def _on_key(self, event) -> None:
        if getattr(self, "_stop", False):
            return
        key = getattr(event, 'key', None)
        if key in ("enter", "return"):
            self._do_iteration(self._ax)
        elif key == "escape":
            self._stop = True
            try:
                plt.close(self._fig)
            except Exception:
                pass

    def _on_close(self, event) -> None:
        self._stop = True

    def _do_iteration(self, ax) -> None:
        # Choose a start and end that are viable in both directions
        s, t = pick_far_nodes(self.G, min_length=self._min_pair_length)

        # Compute both algorithms, out and back
        dfs_out, dfs_exp_out = dfs_path_with_counts(self.G, s, t, max_expansions=self._max_dfs_expansions)
        dfs_back, dfs_exp_back = dfs_path_with_counts(self.G, t, s, max_expansions=self._max_dfs_expansions)

        astar_out, astar_exp_out = astar_path_with_counts(self.G, self.positions, s, t)
        astar_back, astar_exp_back = astar_path_with_counts(self.G, self.positions, t, s)

        # Metrics
        dfs_total_len = path_length(self.G, dfs_out) + path_length(self.G, dfs_back)
        astar_total_len = path_length(self.G, astar_out) + path_length(self.G, astar_back)
        dfs_expanded = dfs_exp_out + dfs_exp_back
        astar_expanded = astar_exp_out + astar_exp_back

        # Render
        self._render_iteration(
            ax,
            s,
            t,
            dfs_out,
            dfs_back,
            astar_out,
            astar_back,
            dfs_expanded,
            astar_expanded,
            dfs_total_len,
            astar_total_len,
        )
        try:
            self._fig.canvas.draw_idle()
        except Exception:
            pass

    def _render_iteration(
        self,
        ax,
        start: Node,
        goal: Node,
        dfs_out: Path,
        dfs_back: Path,
        astar_out: Path,
        astar_back: Path,
        dfs_expanded: int,
        astar_expanded: int,
        dfs_total_len: float,
        astar_total_len: float,
    ) -> None:
        ax.clear()

        # Base map (light gray)
        for u, v in self.G.edges:
            x = [self.positions[u][0], self.positions[v][0]]
            y = [self.positions[u][1], self.positions[v][1]]
            ax.plot(x, y, color="#cccccc", linewidth=0.4, zorder=1)

        def draw_path(path: Path, color: str, lw: float = 2.0, style: str = "-", z: int = 5):
            if not path or len(path) < 2:
                return
            xs = [self.positions[n][0] for n in path]
            ys = [self.positions[n][1] for n in path]
            ax.plot(xs, ys, color=color, linewidth=lw, linestyle=style, zorder=z)

        # Draw DFS paths (out and back)
        draw_path(dfs_out, color="#f39c12", lw=2.5, style="-", z=6)      # orange solid
        draw_path(dfs_back, color="#f39c12", lw=2.0, style="--", z=6)    # orange dashed

        # Draw A* paths (out and back)
        draw_path(astar_out, color="#2980b9", lw=2.5, style="-", z=7)    # blue solid
        draw_path(astar_back, color="#2980b9", lw=2.0, style="--", z=7)  # blue dashed

        # Start and Goal markers
        ax.scatter(
            [self.positions[start][0]], [self.positions[start][1]],
            c="#2ecc71", s=100, marker="^", edgecolors="black", zorder=8, label="Start"
        )
        ax.scatter(
            [self.positions[goal][0]], [self.positions[goal][1]],
            c="#e74c3c", s=100, marker="s", edgecolors="black", zorder=8, label="Goal"
        )

        # Limits
        xs = [p[0] for p in self.positions.values()]
        ys = [p[1] for p in self.positions.values()]
        ax.set_xlim(min(xs) - 0.005, max(xs) + 0.005)
        ax.set_ylim(min(ys) - 0.005, max(ys) + 0.005)

        # Title and metrics box
        ax.set_title(f"A* vs DFS (Round Trip) — {self.place}")
        metrics = (
            f"DFS: visited {dfs_expanded} nodes, round-trip length ≈ {dfs_total_len:.0f} m\n"
            f"A*: expanded {astar_expanded} nodes, round-trip length ≈ {astar_total_len:.0f} m"
        )
        ax.text(0.02, 0.98, metrics, transform=ax.transAxes, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#333333", alpha=0.8))

        # Legend for path styles
        from matplotlib.lines import Line2D
        legend_elems = [
            Line2D([0], [0], color="#f39c12", lw=2.5, linestyle="-", label="DFS out"),
            Line2D([0], [0], color="#f39c12", lw=2.0, linestyle="--", label="DFS back"),
            Line2D([0], [0], color="#2980b9", lw=2.5, linestyle="-", label="A* out"),
            Line2D([0], [0], color="#2980b9", lw=2.0, linestyle="--", label="A* back"),
        ]
        ax.legend(handles=legend_elems, loc="lower right")

        # Instructions overlay
        ax.text(0.02, 0.02, "Press Enter for next pair • Press Esc to quit",
                transform=ax.transAxes, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#333333", alpha=0.75))

    def _plot_comparison(
        self,
        start: Node,
        goal: Node,
        dfs_out: Path,
        dfs_back: Path,
        astar_out: Path,
        astar_back: Path,
        dfs_expanded: int,
        astar_expanded: int,
        dfs_total_len: float,
        astar_total_len: float,
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 10))

        # Base map (light gray)
        for u, v in self.G.edges:
            x = [self.positions[u][0], self.positions[v][0]]
            y = [self.positions[u][1], self.positions[v][1]]
            ax.plot(x, y, color="#cccccc", linewidth=0.4, zorder=1)

        def draw_path(path: Path, color: str, lw: float = 2.0, style: str = "-", z: int = 5):
            if not path or len(path) < 2:
                return
            xs = [self.positions[n][0] for n in path]
            ys = [self.positions[n][1] for n in path]
            ax.plot(xs, ys, color=color, linewidth=lw, linestyle=style, zorder=z)

        # Draw DFS paths (out and back)
        draw_path(dfs_out, color="#f39c12", lw=2.5, style="-", z=6)      # orange solid
        draw_path(dfs_back, color="#f39c12", lw=2.0, style="--", z=6)    # orange dashed

        # Draw A* paths (out and back)
        draw_path(astar_out, color="#2980b9", lw=2.5, style="-", z=7)    # blue solid
        draw_path(astar_back, color="#2980b9", lw=2.0, style="--", z=7)  # blue dashed

        # Start and Goal markers
        ax.scatter(
            [self.positions[start][0]], [self.positions[start][1]],
            c="#2ecc71", s=100, marker="^", edgecolors="black", zorder=8, label="Start"
        )
        ax.scatter(
            [self.positions[goal][0]], [self.positions[goal][1]],
            c="#e74c3c", s=100, marker="s", edgecolors="black", zorder=8, label="Goal"
        )

        # Limits
        xs = [p[0] for p in self.positions.values()]
        ys = [p[1] for p in self.positions.values()]
        ax.set_xlim(min(xs) - 0.005, max(xs) + 0.005)
        ax.set_ylim(min(ys) - 0.005, max(ys) + 0.005)

        # Title and metrics box
        ax.set_title(f"A* vs DFS (Round Trip) — {self.place}")
        metrics = (
            f"DFS: visited {dfs_expanded} nodes, round-trip length ≈ {dfs_total_len:.0f} m\n"
            f"A*: expanded {astar_expanded} nodes, round-trip length ≈ {astar_total_len:.0f} m"
        )
        ax.text(0.02, 0.98, metrics, transform=ax.transAxes, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#333333", alpha=0.8))

        # Legend for path styles
        from matplotlib.lines import Line2D
        legend_elems = [
            Line2D([0], [0], color="#f39c12", lw=2.5, linestyle="-", label="DFS out"),
            Line2D([0], [0], color="#f39c12", lw=2.0, linestyle="--", label="DFS back"),
            Line2D([0], [0], color="#2980b9", lw=2.5, linestyle="-", label="A* out"),
            Line2D([0], [0], color="#2980b9", lw=2.0, linestyle="--", label="A* back"),
        ]
        ax.legend(handles=legend_elems, loc="lower right")

        plt.show()


if __name__ == "__main__":
    # Note: Requires internet access for OSMnx to download the place graph.
    # Don't fix the seed so each run gets a new random start/end.
    comp = AstarVsDFS(place="Vake, Tbilisi, Georgia", seed=None)
    # Interactive mode: Enter for next pair; Esc or close to stop
    comp.run_interactive(min_pair_length=1000.0, max_dfs_expansions=None)
