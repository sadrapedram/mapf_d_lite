import csv
import multiprocessing
import time
from typing import Optional

from astar_vs_dfs import (
    AstarVsDFS,
    astar_path_with_counts,
    dfs_path_with_counts,
    path_length,
    pick_far_nodes,
)


_GLOBAL_G = None
_GLOBAL_POSITIONS = None


def _init_worker(G, positions) -> None:
    global _GLOBAL_G, _GLOBAL_POSITIONS
    _GLOBAL_G = G
    _GLOBAL_POSITIONS = positions


def _run_chunk_core(
    num_iterations: int,
    G,
    positions,
    min_pair_length: float,
    max_dfs_expansions: Optional[int],
):
    total_dfs_distance = 0.0
    total_astar_distance = 0.0
    total_dfs_time = 0.0
    total_astar_time = 0.0

    for _ in range(num_iterations):
        start, goal = pick_far_nodes(G, min_length=min_pair_length)

        dfs_start_time = time.perf_counter()
        dfs_out, _ = dfs_path_with_counts(G, start, goal, max_expansions=max_dfs_expansions)
        dfs_back, _ = dfs_path_with_counts(G, goal, start, max_expansions=max_dfs_expansions)
        total_dfs_time += time.perf_counter() - dfs_start_time

        astar_start_time = time.perf_counter()
        astar_out, _ = astar_path_with_counts(G, positions, start, goal)
        astar_back, _ = astar_path_with_counts(G, positions, goal, start)
        total_astar_time += time.perf_counter() - astar_start_time

        dfs_round_trip = path_length(G, dfs_out) + path_length(G, dfs_back)
        astar_round_trip = path_length(G, astar_out) + path_length(G, astar_back)

        total_dfs_distance += dfs_round_trip
        total_astar_distance += astar_round_trip

    return total_dfs_distance, total_astar_distance, total_dfs_time, total_astar_time


def _run_chunk_worker(
    num_iterations: int,
    min_pair_length: float,
    max_dfs_expansions: Optional[int],
):
    return _run_chunk_core(
        num_iterations,
        _GLOBAL_G,
        _GLOBAL_POSITIONS,
        min_pair_length,
        max_dfs_expansions,
    )


def run_experiments(
    num_iterations: int = 1000000,
    min_pair_length: float = 1000.0,
    max_dfs_expansions: Optional[int] = None,
    output_csv: str = "astar_vs_dfs_stats.csv",
) -> None:
    comp = AstarVsDFS(place="Vake, Tbilisi, Georgia", seed=None)
    G = comp.G
    positions = comp.positions

    total_dfs_distance = 0.0
    total_astar_distance = 0.0
    total_dfs_time = 0.0
    total_astar_time = 0.0
    completed_iterations = 0

    def _update_progress(completed: int) -> None:
        if num_iterations <= 0:
            return
        bar_length = 40
        fraction = completed / float(num_iterations)
        fraction = min(max(fraction, 0.0), 1.0)
        filled = int(bar_length * fraction)
        bar = "#" * filled + "-" * (bar_length - filled)
        print(
            f"\rProgress: |{bar}| {completed}/{num_iterations} ({fraction * 100.0:.1f}%)",
            end="",
            flush=True,
        )

    def _make_chunks(total: int, target_chunk_size: int = 1000) -> list[int]:
        chunks: list[int] = []
        remaining = total
        while remaining > 0:
            size = min(target_chunk_size, remaining)
            chunks.append(size)
            remaining -= size
        return chunks

    try:
        num_workers = multiprocessing.cpu_count()
    except NotImplementedError:
        num_workers = 1

    num_workers = max(1, min(num_workers, num_iterations))

    if num_workers == 1:
        print("Running experiments in single-process mode...")
        chunk_sizes = _make_chunks(num_iterations)
        _update_progress(0)
        for chunk_size in chunk_sizes:
            if chunk_size <= 0:
                continue
            dfs_dist, astar_dist, dfs_time, astar_time = _run_chunk_core(
                chunk_size,
                G,
                positions,
                min_pair_length,
                max_dfs_expansions,
            )
            total_dfs_distance += dfs_dist
            total_astar_distance += astar_dist
            total_dfs_time += dfs_time
            total_astar_time += astar_time
            completed_iterations += chunk_size
            _update_progress(completed_iterations)
        print()
    else:
        print(f"Running experiments using {num_workers} worker processes...")
        chunk_sizes = _make_chunks(num_iterations)

        _update_progress(0)

        def _on_result(result, chunk_size_local: int) -> None:
            nonlocal total_dfs_distance, total_astar_distance, total_dfs_time, total_astar_time, completed_iterations
            dfs_dist, astar_dist, dfs_time, astar_time = result
            total_dfs_distance += dfs_dist
            total_astar_distance += astar_dist
            total_dfs_time += dfs_time
            total_astar_time += astar_time
            completed_iterations += chunk_size_local
            _update_progress(completed_iterations)

        with multiprocessing.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(G, positions),
        ) as pool:
            for chunk_size in chunk_sizes:
                if chunk_size <= 0:
                    continue
                pool.apply_async(
                    _run_chunk_worker,
                    args=(chunk_size, min_pair_length, max_dfs_expansions),
                    callback=lambda res, chunk_size_local=chunk_size: _on_result(res, chunk_size_local),
                )
            pool.close()
            pool.join()
        _update_progress(num_iterations)
        print()

    dfs_avg = total_dfs_distance / float(num_iterations)
    astar_avg = total_astar_distance / float(num_iterations)
    dfs_avg_time = total_dfs_time / float(num_iterations)
    astar_avg_time = total_astar_time / float(num_iterations)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "iterations",
                "dfs_total_distance",
                "dfs_avg_distance_per_round",
                "astar_total_distance",
                "astar_avg_distance_per_round",
                "dfs_total_time_seconds",
                "dfs_avg_time_per_round_seconds",
                "astar_total_time_seconds",
                "astar_avg_time_per_round_seconds",
            ]
        )
        writer.writerow(
            [
                num_iterations,
                total_dfs_distance,
                dfs_avg,
                total_astar_distance,
                astar_avg,
                total_dfs_time,
                dfs_avg_time,
                total_astar_time,
                astar_avg_time,
            ]
        )

    print(f"Results written to {output_csv}")


if __name__ == "__main__":
    # Adjust num_iterations if you want a different number of runs.
    run_experiments(num_iterations=1_000_000)
