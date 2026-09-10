"""
Tests for the key-escaping convention of the HDF5 layer.

`sanitize_key` and `restore_key` are the contract between the on-disk layout
and anything that builds a path-addressed index over it. The escaping has to
survive a round trip *and* survive being joined into a slash-separated path and
split apart again — a metadata key such as 'Irradiance A [mW/cm2]' contains the
same character that separates path components, and the whole point of the
placeholder is that the split does not cut the key in half.

These tests exist because that convention is now consumed outside this package
(pyKES-database indexes files written here). A silent change to the placeholder
or to either function would corrupt an index built against files written by a
different pyKES version, and would otherwise fail nowhere.
"""

import numpy as np
import pytest

from pyKES.database.database_experiments import (KEY_SLASH_PLACEHOLDER,
                                                 compression_arguments,
                                                 restore_key,
                                                 sanitize_key)


KEYS_WITH_SLASHES = [
    'Catalyst loading [wt% Rh/Cr]',
    'Irradiance A [mW/cm2]',
    'Catalyst concentration [g/L]',
    'a/b/c',
    '/leading',
    'trailing/',
    '//',
]

KEYS_WITHOUT_SLASHES = [
    'Experiment',
    'Synthesis temperature [°C]',
    'plain',
    '',
]


@pytest.mark.parametrize('key', KEYS_WITH_SLASHES + KEYS_WITHOUT_SLASHES)
def test_round_trip_returns_the_original_key(key):
    assert restore_key(sanitize_key(key)) == key


@pytest.mark.parametrize('key', KEYS_WITH_SLASHES)
def test_an_escaped_key_carries_no_separator(key):
    """
    The escaped form must be free of '/', or joining it into a path and
    splitting again would produce more components than keys.
    """
    assert '/' not in sanitize_key(key)


@pytest.mark.parametrize('key', KEYS_WITHOUT_SLASHES)
def test_a_key_without_slashes_is_unchanged(key):
    assert sanitize_key(key) == key


def test_a_joined_path_splits_back_into_the_original_keys():
    """
    The property the index actually depends on: escape each key, join with the
    separator, split on the separator, restore each component.
    """
    keys = ['metadata', 'Catalyst loading [wt% Rh/Cr]', 'Irradiance A [mW/cm2]']

    path = '/'.join(sanitize_key(key) for key in keys)
    recovered = [restore_key(component) for component in path.split('/')]

    assert recovered == keys


def test_non_string_keys_are_stringified():
    """Uploads can carry integer keys in nested dictionaries."""
    assert sanitize_key(3) == '3'
    assert sanitize_key(None) == 'None'


def test_the_placeholder_is_what_restore_key_looks_for():
    """
    Guards against the constant and the two functions drifting apart — the
    failure mode that would otherwise surface only as a corrupt index.
    """
    assert sanitize_key('a/b') == f'a{KEY_SLASH_PLACEHOLDER}b'
    assert restore_key(f'a{KEY_SLASH_PLACEHOLDER}b') == 'a/b'


# =============================================================================
# Compression
# =============================================================================

def test_no_compression_requested_gives_no_keywords():
    assert compression_arguments(np.arange(1000), None) == {}


def test_scalars_and_non_arrays_are_never_compressed():
    """HDF5 refuses to compress scalar datasets outright."""
    assert compression_arguments(np.float64(1.0), 'gzip') == {}
    assert compression_arguments('a string', 'gzip') == {}
    assert compression_arguments(np.array(5), 'gzip') == {}


def test_small_arrays_are_not_compressed():
    assert compression_arguments(np.arange(10), 'gzip') == {}


def test_a_large_array_is_compressed_at_the_default_level():
    arguments = compression_arguments(np.arange(1000), 'gzip')

    assert arguments['compression'] == 'gzip'
    assert arguments['compression_opts'] == 4
