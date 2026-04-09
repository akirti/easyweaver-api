"""Tests for easyweaver.processes.dag — DAG utilities."""

import pytest

from easyweaver.core.exceptions import ProcessExecutionError
from easyweaver.processes.dag import (
    build_dag,
    detect_cycles,
    get_ready_datasets,
    topological_sort,
)


# ── build_dag ─────────────────────────────────────────────────────────

class TestBuildDag:
    def test_no_bindings(self):
        """Queries with no bindings produce an empty dependency list."""
        queries = {
            "orders": {"main": {"source_id": "s1", "table": "orders"}},
            "customers": {"main": {"source_id": "s2", "table": "customers"}},
        }
        dag = build_dag(queries)
        assert dag == {"orders.main": [], "customers.main": []}

    def test_single_binding(self):
        queries = {
            "orders": {"main": {"source_id": "s1", "table": "orders"}},
            "customers": {
                "main": {
                    "source_id": "s2",
                    "table": "customers",
                    "bindings": [
                        {"source_dataset": "orders.main", "mode": "distinct", "mappings": []},
                    ],
                }
            },
        }
        dag = build_dag(queries)
        assert dag == {"orders.main": [], "customers.main": ["orders.main"]}

    def test_diamond_dependencies(self):
        queries = {
            "A": {"q": {"source_id": "s1", "table": "a"}},
            "B": {
                "q": {
                    "source_id": "s2",
                    "table": "b",
                    "bindings": [{"source_dataset": "A.q", "mode": "distinct", "mappings": []}],
                }
            },
            "C": {
                "q": {
                    "source_id": "s3",
                    "table": "c",
                    "bindings": [{"source_dataset": "A.q", "mode": "distinct", "mappings": []}],
                }
            },
            "D": {
                "q": {
                    "source_id": "s4",
                    "table": "d",
                    "bindings": [
                        {"source_dataset": "B.q", "mode": "distinct", "mappings": []},
                        {"source_dataset": "C.q", "mode": "row_pair", "mappings": []},
                    ],
                }
            },
        }
        dag = build_dag(queries)
        assert dag["A.q"] == []
        assert dag["B.q"] == ["A.q"]
        assert dag["C.q"] == ["A.q"]
        assert sorted(dag["D.q"]) == ["B.q", "C.q"]

    def test_with_pydantic_models(self):
        """build_dag works with Pydantic model objects (ProcessQueryConfig)."""
        from easyweaver.processes.schemas import (
            BindingMapping,
            ProcessQueryBinding,
            ProcessQueryConfig,
        )

        queries = {
            "src": {
                "main": ProcessQueryConfig(
                    source_id="s1", table="src_table"
                )
            },
            "tgt": {
                "main": ProcessQueryConfig(
                    source_id="s2",
                    table="tgt_table",
                    bindings=[
                        ProcessQueryBinding(
                            source_dataset="src.main",
                            mode="distinct",
                            mappings=[BindingMapping(source_column="id", target_column="src_id")],
                        )
                    ],
                )
            },
        }
        dag = build_dag(queries)
        assert dag == {"src.main": [], "tgt.main": ["src.main"]}

    def test_empty_queries(self):
        assert build_dag({}) == {}

    def test_duplicate_source_in_bindings(self):
        """Same source referenced twice should only appear once."""
        queries = {
            "A": {"q": {"source_id": "s1", "table": "a"}},
            "B": {
                "q": {
                    "source_id": "s2",
                    "table": "b",
                    "bindings": [
                        {"source_dataset": "A.q", "mode": "distinct", "mappings": []},
                        {"source_dataset": "A.q", "mode": "row_pair", "mappings": []},
                    ],
                }
            },
        }
        dag = build_dag(queries)
        assert dag["B.q"] == ["A.q"]


# ── detect_cycles ─────────────────────────────────────────────────────

class TestDetectCycles:
    def test_no_cycle(self):
        dag = {"A": [], "B": ["A"], "C": ["B"]}
        detect_cycles(dag)  # should not raise

    def test_self_cycle(self):
        dag = {"A": ["A"]}
        with pytest.raises(ProcessExecutionError, match="Cycle detected"):
            detect_cycles(dag)

    def test_two_node_cycle(self):
        dag = {"A": ["B"], "B": ["A"]}
        with pytest.raises(ProcessExecutionError, match="Cycle detected"):
            detect_cycles(dag)

    def test_three_node_cycle(self):
        dag = {"A": ["B"], "B": ["C"], "C": ["A"]}
        with pytest.raises(ProcessExecutionError, match="Cycle detected"):
            detect_cycles(dag)

    def test_diamond_no_cycle(self):
        dag = {"A": [], "B": ["A"], "C": ["A"], "D": ["B", "C"]}
        detect_cycles(dag)  # should not raise

    def test_empty_dag(self):
        detect_cycles({})  # should not raise


# ── topological_sort ──────────────────────────────────────────────────

class TestTopologicalSort:
    def test_empty(self):
        assert topological_sort({}) == []

    def test_no_dependencies(self):
        dag = {"A": [], "B": [], "C": []}
        waves = topological_sort(dag)
        assert len(waves) == 1
        assert sorted(waves[0]) == ["A", "B", "C"]

    def test_linear_chain(self):
        dag = {"A": [], "B": ["A"], "C": ["B"]}
        waves = topological_sort(dag)
        assert waves == [["A"], ["B"], ["C"]]

    def test_diamond(self):
        dag = {"A": [], "B": ["A"], "C": ["A"], "D": ["B", "C"]}
        waves = topological_sort(dag)
        assert len(waves) == 3
        assert waves[0] == ["A"]
        assert sorted(waves[1]) == ["B", "C"]
        assert waves[2] == ["D"]

    def test_two_independent_chains(self):
        dag = {"A": [], "B": ["A"], "X": [], "Y": ["X"]}
        waves = topological_sort(dag)
        assert len(waves) == 2
        assert sorted(waves[0]) == ["A", "X"]
        assert sorted(waves[1]) == ["B", "Y"]


# ── get_ready_datasets ────────────────────────────────────────────────

class TestGetReadyDatasets:
    def test_all_independent(self):
        dag = {"A": [], "B": [], "C": []}
        ready = get_ready_datasets(dag, completed=set())
        assert ready == ["A", "B", "C"]

    def test_chain_nothing_completed(self):
        dag = {"A": [], "B": ["A"], "C": ["B"]}
        ready = get_ready_datasets(dag, completed=set())
        assert ready == ["A"]

    def test_chain_one_completed(self):
        dag = {"A": [], "B": ["A"], "C": ["B"]}
        ready = get_ready_datasets(dag, completed={"A"})
        assert ready == ["B"]

    def test_chain_two_completed(self):
        dag = {"A": [], "B": ["A"], "C": ["B"]}
        ready = get_ready_datasets(dag, completed={"A", "B"})
        assert ready == ["C"]

    def test_diamond_partial(self):
        dag = {"A": [], "B": ["A"], "C": ["A"], "D": ["B", "C"]}
        # Only B completed (A already done), C not yet
        ready = get_ready_datasets(dag, completed={"A", "B"})
        assert ready == ["C"]  # D still waiting on C

    def test_diamond_both_branches_done(self):
        dag = {"A": [], "B": ["A"], "C": ["A"], "D": ["B", "C"]}
        ready = get_ready_datasets(dag, completed={"A", "B", "C"})
        assert ready == ["D"]

    def test_all_completed(self):
        dag = {"A": [], "B": ["A"]}
        ready = get_ready_datasets(dag, completed={"A", "B"})
        assert ready == []

    def test_external_dependency_treated_as_satisfied(self):
        """Deps referencing keys NOT in the DAG are treated as satisfied."""
        dag = {"A": ["external.source"]}
        ready = get_ready_datasets(dag, completed=set())
        assert ready == ["A"]
