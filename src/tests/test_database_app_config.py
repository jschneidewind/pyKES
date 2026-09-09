"""
Tests for the deployment configuration surface.

One property here matters more than the rest: development login must be off
unless a deployment explicitly asks for it. Its failure mode is silent — a
proxy that stops sending the identity header would otherwise make every
visitor an administrator called `developer`, and attribute their uploads to a
person who does not exist — so the default is asserted rather than assumed.
"""

from pathlib import Path

import pytest
import streamlit as st

from pyKES.database_app.config import (
    DEVELOPMENT_USER,
    DatabaseAppConfig,
    as_boolean,
    reference_instructions,
    setting_from_environment,
)
from pyKES.database_app.session import read_identity


# =============================================================================
# Fixtures
# =============================================================================

class FakeContext:
    """Stands in for `st.context`, whose headers come from the live request."""

    def __init__(self, headers):
        self.headers = headers


@pytest.fixture
def proxy_headers(monkeypatch):
    """Install a callable that sets the headers `read_identity` will read."""
    def set_headers(headers):
        monkeypatch.setattr(st, "context", FakeContext(headers))

    return set_headers


@pytest.fixture
def schema_directory(tmp_path):
    """A one-file schema directory declaring a single reference column."""
    directory = tmp_path / "schemas"
    directory.mkdir()
    (directory / "catalyst_batch.yaml").write_text(
        "entity_type: catalyst_batch\n"
        "identifier_field: Batch\n"
        "fields:\n"
        "  - name: Batch\n"
        "    type: text\n"
        "  - name: Semiconductor\n"
        "    type: reference\n"
        "    role: finished_semiconductor\n"
        "    accepts: [finished_semiconductor]\n",
        encoding="utf-8")

    return directory


# =============================================================================
# Reading settings from the environment
# =============================================================================

def test_setting_falls_back_when_unset(monkeypatch):
    monkeypatch.delenv("PHOTOCAT_PAGE_SIZE", raising=False)

    assert setting_from_environment("PAGE_SIZE", 50, int) == 50


def test_setting_is_cast(monkeypatch):
    monkeypatch.setenv("PHOTOCAT_PAGE_SIZE", "25")

    assert setting_from_environment("PAGE_SIZE", 50, int) == 25


@pytest.mark.parametrize("text", ["1", "true", "TRUE", "yes", "on", " on "])
def test_affirmative_values_are_true(text):
    assert as_boolean(text) is True


@pytest.mark.parametrize("text", ["0", "false", "no", "off", "", "maybe", "tru"])
def test_everything_else_is_false(text):
    """A misspelt value must fail closed, not enable what it names."""
    assert as_boolean(text) is False


# =============================================================================
# The configuration dataclass
# =============================================================================

def test_development_login_is_off_by_default(monkeypatch):
    monkeypatch.delenv("PHOTOCAT_ALLOW_DEV_LOGIN", raising=False)

    assert DatabaseAppConfig().allow_development_login is False


def test_development_login_is_opt_in(monkeypatch):
    monkeypatch.setenv("PHOTOCAT_ALLOW_DEV_LOGIN", "1")

    assert DatabaseAppConfig().allow_development_login is True


def test_deployment_settings_come_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("PHOTOCAT_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("PHOTOCAT_PAGE_SIZE", "10")
    monkeypatch.setenv("PHOTOCAT_TITLE", "Group Database")
    monkeypatch.setenv("PHOTOCAT_ADMIN_GROUP", "photocat-admins")
    monkeypatch.setenv("PHOTOCAT_USER_HEADER", "X-Forwarded-User")

    config = DatabaseAppConfig()

    assert config.data_root == tmp_path / "data"
    assert config.page_size == 10
    assert config.title == "Group Database"
    assert config.admin_group == "photocat-admins"
    assert config.user_header == "X-Forwarded-User"


def test_reference_instructions_follow_the_configured_schemas(schema_directory):
    """
    A deployment with its own vocabulary must have its uploads and its
    corrections read with the *same* declarations. Held apart, the same column
    means one thing on upload and another on an edit.
    """
    config = DatabaseAppConfig(schema_directory=schema_directory)

    declared = config.reference_instructions_by_type

    assert set(declared) == {"catalyst_batch"}
    assert declared["catalyst_batch"]["Semiconductor"]["role"] == \
        "finished_semiconductor"
    assert declared == reference_instructions(schema_directory)


def test_declaring_no_references_is_distinguishable_from_declaring_none(
        schema_directory):
    """
    Tested against the falsy default it replaced: `{}` used to be
    indistinguishable from "not passed", so a deployment that wanted no
    reference columns silently got the shipped schemas' instead.
    """
    explicit = DatabaseAppConfig(schema_directory=schema_directory,
                                 reference_instructions_by_type={})

    assert explicit.reference_instructions_by_type == {}

    derived = DatabaseAppConfig(schema_directory=schema_directory)

    assert derived.reference_instructions_by_type


def test_paths_are_coerced_from_text():
    config = DatabaseAppConfig(data_root="/srv/photocat/data")

    assert isinstance(config.data_root, Path)
    assert isinstance(config.schema_directory, Path)


# =============================================================================
# Identity
# =============================================================================

def test_identity_comes_from_the_proxy_headers(proxy_headers):
    proxy_headers({"Remote-User": "ae", "Remote-Groups": "users, admins"})
    config = DatabaseAppConfig(allow_development_login=False)

    identity = read_identity(config)

    assert identity.name == "ae"
    assert identity.groups == ["users", "admins"]
    assert identity.is_admin is True
    assert identity.authenticated is True


def test_a_missing_header_is_refused_in_a_deployment(proxy_headers):
    """The whole point of the default: no header, no service."""
    proxy_headers({})
    config = DatabaseAppConfig(allow_development_login=False)

    with pytest.raises(PermissionError, match="Remote-User"):
        read_identity(config)


def test_a_missing_header_falls_back_only_when_allowed(proxy_headers):
    proxy_headers({})
    config = DatabaseAppConfig(allow_development_login=True)

    identity = read_identity(config)

    assert identity.name == DEVELOPMENT_USER
    assert identity.authenticated is False


def test_configured_header_and_group_names_are_honoured(proxy_headers):
    proxy_headers({"X-Forwarded-User": "js",
                   "X-Forwarded-Groups": "photocat-admins"})
    config = DatabaseAppConfig(user_header="X-Forwarded-User",
                               groups_header="X-Forwarded-Groups",
                               admin_group="photocat-admins",
                               allow_development_login=False)

    identity = read_identity(config)

    assert identity.name == "js"
    assert identity.is_admin is True


def test_a_user_outside_the_admin_group_is_not_an_admin(proxy_headers):
    proxy_headers({"Remote-User": "ae", "Remote-Groups": "users"})
    config = DatabaseAppConfig(allow_development_login=False)

    assert read_identity(config).is_admin is False
