"""Tests for easyweaver.tasks.celery_app — Celery app creation and config."""
import pytest


class TestCeleryApp:
    @pytest.fixture
    def celery_app(self):
        from easyweaver.tasks.celery_app import celery_app
        return celery_app

    def test_celery_app_is_created(self, celery_app):
        from celery import Celery
        assert isinstance(celery_app, Celery)

    def test_celery_app_name(self, celery_app):
        assert celery_app.main == "easyweaver"

    def test_task_serializer_is_json(self, celery_app):
        assert celery_app.conf.task_serializer == "json"

    def test_accept_content_includes_json(self, celery_app):
        assert "json" in celery_app.conf.accept_content

    def test_result_serializer_is_json(self, celery_app):
        assert celery_app.conf.result_serializer == "json"

    def test_timezone_is_utc(self, celery_app):
        assert celery_app.conf.timezone == "UTC"

    def test_enable_utc_is_true(self, celery_app):
        assert celery_app.conf.enable_utc is True

    def test_task_track_started_is_true(self, celery_app):
        assert celery_app.conf.task_track_started is True

    def test_task_acks_late_is_true(self, celery_app):
        assert celery_app.conf.task_acks_late is True

    def test_worker_prefetch_multiplier_is_one(self, celery_app):
        assert celery_app.conf.worker_prefetch_multiplier == 1

    def test_broker_url_comes_from_settings(self, celery_app):
        from easyweaver.settings import settings
        assert celery_app.conf.broker_url == settings.celery_broker_url

    def test_result_backend_comes_from_settings(self, celery_app):
        from easyweaver.settings import settings
        assert celery_app.conf.result_backend == settings.celery_result_backend

    def test_celery_app_is_importable(self):
        from easyweaver.tasks.celery_app import celery_app
        assert celery_app is not None
