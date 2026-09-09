"""
Tests for what the application reports about itself.

An update that silently did not take looks exactly like one that did, so these
four facts are what make the difference visible from a browser rather than
from an SSH session.
"""

import pytest

from pyKES.database.index_schema import INDEX_SCHEMA_VERSION, IndexPaths, open_index
from pyKES.database_app.config import DatabaseAppConfig
from pyKES.database_app.config import PRODUCTION_ENVIRONMENT
from pyKES.database_app.deployment import (
    deployment_environment,
    deployment_summary,
)
from pyKES.utilities.version_information import UNKNOWN_VERSION


@pytest.fixture
def connection(tmp_path):
    handle = open_index(IndexPaths(root=tmp_path))
    yield handle
    handle.close()


def test_the_summary_names_the_running_code(connection, tmp_path):
    summary = deployment_summary(connection, DatabaseAppConfig(data_root=tmp_path))

    assert summary["pykes_version"] != UNKNOWN_VERSION
    assert summary["code_schema"] == INDEX_SCHEMA_VERSION
    assert summary["data_root"] == str(tmp_path)


def test_the_recorded_and_expected_schema_are_both_reported(connection, tmp_path):
    """
    They agree on a healthy index. Reporting both is what makes a mismatch —
    an index opened for repair, or one mid-migration — visible.
    """
    summary = deployment_summary(connection, DatabaseAppConfig(data_root=tmp_path))

    assert summary["recorded_schema"] == summary["code_schema"]


def test_the_image_and_commit_are_reported_when_the_deployment_sets_them(
        connection, tmp_path, monkeypatch):
    """
    Set by the container image and the systemd unit, and absent in a source
    checkout — which is itself worth being able to see.
    """
    monkeypatch.setenv("PHOTOCAT_IMAGE_TAG", "v0.3.0")
    monkeypatch.setenv("PHOTOCAT_GIT_SHA", "0123456789abcdef")

    summary = deployment_summary(connection, DatabaseAppConfig(data_root=tmp_path))

    assert summary["image_tag"] == "v0.3.0"
    assert summary["git_sha"] == "01234567"


def test_a_deployment_that_does_not_claim_production_is_not_treated_as_one(
        monkeypatch):
    """
    The banner is a backstop, so the default has to fail the safe way: an
    instance that forgot to say what it is gets the warning, rather than
    passing itself off as the group's record.
    """
    monkeypatch.delenv("PHOTOCAT_ENV", raising=False)

    assert deployment_environment() != PRODUCTION_ENVIRONMENT

    monkeypatch.setenv("PHOTOCAT_ENV", "production")

    assert deployment_environment() == PRODUCTION_ENVIRONMENT
