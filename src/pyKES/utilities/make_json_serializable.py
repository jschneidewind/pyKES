import json

def make_json_serializable(obj):
    """Recursively convert non-JSON-serializable objects to strings."""
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

