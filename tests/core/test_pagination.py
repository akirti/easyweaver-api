"""Tests for easyweaver.core.pagination"""
import pytest

from easyweaver.core.pagination import PaginatedResult, PaginationParams


# ---------------------------------------------------------------------------
# PaginationParams
# ---------------------------------------------------------------------------


class TestPaginationParams:
    def test_defaults(self):
        p = PaginationParams()
        assert p.page == 1
        assert p.page_size == 50

    def test_custom_values(self):
        p = PaginationParams(page=3, page_size=20)
        assert p.page == 3
        assert p.page_size == 20

    def test_offset_page_1(self):
        p = PaginationParams(page=1, page_size=50)
        assert p.offset == 0

    def test_offset_page_2(self):
        p = PaginationParams(page=2, page_size=50)
        assert p.offset == 50

    def test_offset_page_3_size_20(self):
        p = PaginationParams(page=3, page_size=20)
        assert p.offset == 40

    def test_offset_page_1_size_100(self):
        p = PaginationParams(page=1, page_size=100)
        assert p.offset == 0

    def test_offset_formula(self):
        # offset = (page - 1) * page_size
        for page in range(1, 6):
            for size in [10, 25, 50]:
                p = PaginationParams(page=page, page_size=size)
                assert p.offset == (page - 1) * size


# ---------------------------------------------------------------------------
# PaginatedResult
# ---------------------------------------------------------------------------


class TestPaginatedResult:
    def _make(self, items, total, page=1, page_size=10):
        return PaginatedResult(items=items, total=total, page=page, page_size=page_size)

    def test_total_pages_exact(self):
        r = self._make([], total=100, page_size=10)
        assert r.total_pages == 10

    def test_total_pages_with_remainder(self):
        r = self._make([], total=101, page_size=10)
        assert r.total_pages == 11

    def test_total_pages_zero_total(self):
        r = self._make([], total=0, page_size=10)
        assert r.total_pages == 0

    def test_total_pages_less_than_page_size(self):
        r = self._make([], total=5, page_size=10)
        assert r.total_pages == 1

    def test_total_pages_single_item(self):
        r = self._make(["x"], total=1, page_size=50)
        assert r.total_pages == 1

    def test_to_dict_no_serializer(self):
        items = [1, 2, 3]
        r = self._make(items, total=3, page=1, page_size=10)
        d = r.to_dict()
        assert d["items"] == [1, 2, 3]
        assert d["total"] == 3
        assert d["page"] == 1
        assert d["page_size"] == 10
        assert d["total_pages"] == 1

    def test_to_dict_with_serializer(self):
        items = [{"id": 1}, {"id": 2}]
        r = self._make(items, total=2, page=1, page_size=10)
        d = r.to_dict(item_serializer=lambda x: x["id"])
        assert d["items"] == [1, 2]

    def test_to_dict_empty_items(self):
        r = self._make([], total=0, page=1, page_size=10)
        d = r.to_dict()
        assert d["items"] == []
        assert d["total"] == 0
        assert d["total_pages"] == 0

    def test_to_dict_contains_all_keys(self):
        r = self._make([1], total=1)
        d = r.to_dict()
        for key in ("items", "total", "page", "page_size", "total_pages"):
            assert key in d

    def test_items_preserved_as_list(self):
        items = ["a", "b", "c"]
        r = self._make(items, total=3)
        assert r.items is items

    def test_generic_type_any(self):
        # PaginatedResult is Generic — works with strings, dicts, ints
        r: PaginatedResult[str] = PaginatedResult(
            items=["hello", "world"], total=2, page=1, page_size=10
        )
        assert r.items == ["hello", "world"]

    def test_page_and_page_size_in_result(self):
        r = PaginatedResult(items=[], total=50, page=3, page_size=15)
        assert r.page == 3
        assert r.page_size == 15
