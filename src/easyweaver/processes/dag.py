"""DAG (Directed Acyclic Graph) utilities for process execution ordering.

Builds a dependency graph from process query bindings, detects cycles,
and produces topological execution waves for parallel scheduling.
"""

from __future__ import annotations

from easyweaver.core.exceptions import ProcessExecutionError


def build_dag(queries: dict[str, dict]) -> dict[str, list[str]]:
    """Build a DAG adjacency list from process query configs.

    Args:
        queries: Mapping of schema_name -> {query_name -> ProcessQueryConfig}.
            Each ProcessQueryConfig may have a ``bindings`` list whose entries
            contain a ``source_dataset`` field (e.g. ``"schema1.orders"``).

    Returns:
        Adjacency list where each key maps to a list of its dependencies.
        Example: ``{"orders.main": [], "customers.main": ["orders.main"]}``
    """
    dag: dict[str, list[str]] = {}

    for schema_name, schema_queries in queries.items():
        for query_name, query_config in schema_queries.items():
            key = f"{schema_name}.{query_name}"
            deps: list[str] = []

            # Extract bindings - handle both model objects and dicts
            bindings = None
            if hasattr(query_config, "bindings"):
                bindings = query_config.bindings
            elif isinstance(query_config, dict):
                bindings = query_config.get("bindings", [])

            if bindings:
                for binding in bindings:
                    if hasattr(binding, "source_dataset"):
                        src = binding.source_dataset
                    elif isinstance(binding, dict):
                        src = binding.get("source_dataset", "")
                    else:
                        continue
                    if src and src not in deps:
                        deps.append(src)

            dag[key] = deps

    return dag


def detect_cycles(dag: dict[str, list[str]]) -> None:
    """Raise ``ProcessExecutionError`` if the DAG contains a cycle.

    Uses iterative DFS with white/gray/black colouring.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = dict.fromkeys(dag, WHITE)

    for start in dag:
        if color[start] != WHITE:
            continue
        # Iterative DFS
        stack: list[tuple[str, int]] = [(start, 0)]
        color[start] = GRAY

        while stack:
            node, idx = stack.pop()
            deps = dag.get(node, [])

            if idx < len(deps):
                # Push current node back with next index
                stack.append((node, idx + 1))
                neighbour = deps[idx]
                if neighbour not in dag:
                    # Reference to a node not in the DAG - skip
                    continue
                if color[neighbour] == GRAY:
                    raise ProcessExecutionError(
                        f"Cycle detected in process dependencies involving '{neighbour}'",
                        details={"node": neighbour},
                    )
                if color[neighbour] == WHITE:
                    color[neighbour] = GRAY
                    stack.append((neighbour, 0))
            else:
                color[node] = BLACK


def topological_sort(dag: dict[str, list[str]]) -> list[list[str]]:
    """Return execution waves via Kahn's algorithm (BFS topological sort).

    Each wave contains nodes whose dependencies are all in earlier waves,
    meaning they can execute in parallel.

    Returns:
        List of waves, e.g. ``[["orders.main", "shipments.main"], ["customers.main"]]``
    """
    if not dag:
        return []

    # Build in-degree map (only counting edges within the DAG)
    in_degree: dict[str, int] = dict.fromkeys(dag, 0)
    for node, deps in dag.items():
        for dep in deps:
            if dep in dag:
                in_degree[node] = in_degree.get(node, 0) + 1

    waves: list[list[str]] = []
    remaining = dict(in_degree)

    while True:
        wave = sorted(node for node, deg in remaining.items() if deg == 0)
        if not wave:
            break
        waves.append(wave)
        for node in wave:
            del remaining[node]
        # Recount in-degrees: only deps still in remaining count
        for node in remaining:
            remaining[node] = sum(1 for dep in dag[node] if dep in remaining)

    return waves


def get_ready_datasets(dag: dict[str, list[str]], completed: set[str]) -> list[str]:
    """Return datasets whose dependencies are fully satisfied.

    A dataset is ready if all its dependencies are in ``completed``.
    Datasets already in ``completed`` are excluded.

    Args:
        dag: The dependency graph.
        completed: Set of already-completed dataset keys.

    Returns:
        Sorted list of dataset keys ready for execution.
    """
    ready = []
    for node, deps in dag.items():
        if node in completed:
            continue
        # All deps must be completed (deps not in dag are considered satisfied)
        if all(dep in completed or dep not in dag for dep in deps):
            ready.append(node)
    return sorted(ready)
