"""Coerce arbitrary structures into something `json.dumps` accepts."""

import json

def make_json_serializable(obj):
    """
    Recursively replace anything JSON cannot represent with a string.

    Used before storing configuration dictionaries in an HDF5 attribute — the
    ``other_multipliers`` of a fitted model, for instance, hold callables, which
    have no JSON representation but are worth recording by name so a stored fit
    says which absorption function it used.

    Parameters
    ----------
    obj : object
        Value to convert. Dictionaries and sequences are walked recursively.

    Returns
    -------
    object
        A structure of JSON-representable values. Callables become
        ``'<function: module.name>'``; anything else that cannot be serialized
        becomes its `str`.
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif callable(obj):
        return f"<function: {obj.__module__}.{obj.__name__}>"
    elif hasattr(obj, '__dict__'):
        return str(obj)
    else:
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return str(obj)

