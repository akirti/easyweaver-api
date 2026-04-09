"""Static code quality checks for new batch-execution files.

Replaces a full SonarQube analysis with lightweight AST-based checks:
- No TODO / FIXME / HACK comments
- All classes and public functions have docstrings
- No bare ``except:`` clauses
- Import ordering follows stdlib -> third-party -> local convention
"""

import ast
import re
import sys
import textwrap
from pathlib import Path

import pytest

# Files under inspection
_PROJECT_ROOT = Path(__file__).resolve().parents[2] / "src" / "easyweaver"
_TARGET_FILES: list[Path] = [
    _PROJECT_ROOT / "processes" / "dag.py",
    _PROJECT_ROOT / "processes" / "batch_adapter.py",
    _PROJECT_ROOT / "processes" / "progress.py",
    _PROJECT_ROOT / "processes" / "ws_handler.py",
    _PROJECT_ROOT / "connectors" / "base.py",
]


def _existing_files() -> list[Path]:
    """Return only target files that exist on disk."""
    return [f for f in _TARGET_FILES if f.exists()]


# ---------------------------------------------------------------------------
# 1. No TODO / FIXME / HACK comments
# ---------------------------------------------------------------------------

_BAD_COMMENT_RE = re.compile(r"#\s*(TODO|FIXME|HACK)\b", re.IGNORECASE)


class TestNoTodoComments:
    """Ensure new files contain no TODO/FIXME/HACK markers."""

    @pytest.mark.parametrize("filepath", _existing_files(), ids=lambda p: p.name)
    def test_no_bad_comments(self, filepath: Path):
        source = filepath.read_text(encoding="utf-8")
        violations: list[str] = []
        for lineno, line in enumerate(source.splitlines(), start=1):
            if _BAD_COMMENT_RE.search(line):
                violations.append(f"  {filepath.name}:{lineno}: {line.strip()}")
        assert not violations, (
            f"Found TODO/FIXME/HACK comments:\n" + "\n".join(violations)
        )


# ---------------------------------------------------------------------------
# 2. Docstrings on classes and public functions
# ---------------------------------------------------------------------------

class TestDocstrings:
    """All classes and public functions/methods must have docstrings."""

    @pytest.mark.parametrize("filepath", _existing_files(), ids=lambda p: p.name)
    def test_classes_have_docstrings(self, filepath: Path):
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
        missing: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if not ast.get_docstring(node):
                    missing.append(f"  class {node.name} (line {node.lineno})")
        assert not missing, (
            f"{filepath.name}: classes missing docstrings:\n" + "\n".join(missing)
        )

    @pytest.mark.parametrize("filepath", _existing_files(), ids=lambda p: p.name)
    def test_public_functions_have_docstrings(self, filepath: Path):
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
        missing: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Skip private/dunder helpers
                if node.name.startswith("_"):
                    continue
                if not ast.get_docstring(node):
                    missing.append(f"  def {node.name} (line {node.lineno})")
        assert not missing, (
            f"{filepath.name}: public functions missing docstrings:\n"
            + "\n".join(missing)
        )


# ---------------------------------------------------------------------------
# 3. No bare except clauses
# ---------------------------------------------------------------------------

class TestNoBareExcept:
    """No bare ``except:`` (must specify exception type)."""

    @pytest.mark.parametrize("filepath", _existing_files(), ids=lambda p: p.name)
    def test_no_bare_except(self, filepath: Path):
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
        violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    violations.append(f"  bare except at line {node.lineno}")
        assert not violations, (
            f"{filepath.name}: bare except clauses found:\n" + "\n".join(violations)
        )


# ---------------------------------------------------------------------------
# 4. Import ordering: stdlib -> third-party -> local
# ---------------------------------------------------------------------------

# Well-known third-party top-level packages used in this project
_THIRD_PARTY = frozenset({
    "polars", "pydantic", "pydantic_settings", "fastapi", "motor",
    "structlog", "redis", "starlette", "uvicorn", "httpx", "anyio",
    "pytest", "cryptography",
})


def _classify_import(node: ast.Import | ast.ImportFrom) -> str:
    """Classify an import as 'stdlib', 'third_party', or 'local'."""
    if isinstance(node, ast.Import):
        module = node.names[0].name.split(".")[0]
    else:
        module = (node.module or "").split(".")[0]

    if module in _THIRD_PARTY:
        return "third_party"

    # Check if it is a stdlib module
    if module in sys.stdlib_module_names:
        return "stdlib"

    # Anything else is local (easyweaver.*)
    return "local"


class TestImportOrdering:
    """Imports should follow stdlib -> third-party -> local ordering."""

    @pytest.mark.parametrize("filepath", _existing_files(), ids=lambda p: p.name)
    def test_import_order(self, filepath: Path):
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))

        # Collect top-level import nodes in order
        imports: list[tuple[int, str, str]] = []  # (lineno, category, text)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                cat = _classify_import(node)
                text = ast.get_source_segment(source, node) or ""
                imports.append((node.lineno, cat, text.strip()))

        if not imports:
            return  # No imports to check

        # Build the expected category sequence (ignoring __future__ which is always first)
        _ORDER = {"stdlib": 0, "third_party": 1, "local": 2}
        seen_order: list[int] = []
        for lineno, cat, text in imports:
            # Skip __future__ imports (always first, special case)
            if "from __future__" in text:
                continue
            # Skip TYPE_CHECKING guarded imports (they follow different rules)
            seen_order.append(_ORDER[cat])

        # Verify non-decreasing order
        violations: list[str] = []
        for i in range(1, len(seen_order)):
            if seen_order[i] < seen_order[i - 1]:
                lineno_i = imports[i][0]
                violations.append(
                    f"  line {lineno_i}: {imports[i][2]!r} ({imports[i][1]}) "
                    f"appears after {imports[i-1][1]} import"
                )

        assert not violations, (
            f"{filepath.name}: import ordering violations:\n" + "\n".join(violations)
        )
