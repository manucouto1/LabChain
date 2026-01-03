import inspect
import hashlib


def method_is_overridden(cls, method_name):
    method = getattr(cls, method_name)
    for base in inspect.getmro(cls)[1:]:  # Skip the class itself
        if hasattr(base, method_name):
            return method is not getattr(base, method_name)
    return False


def hash_normalize(obj):
    if obj is None:
        return ("None",)

    elif isinstance(obj, (bool, int, float, str)):
        return ("scalar", obj)

    elif isinstance(obj, bytes):
        return ("bytes", hashlib.sha256(obj).hexdigest())

    elif isinstance(obj, tuple):
        return ("tuple", tuple(hash_normalize(x) for x in obj))

    elif isinstance(obj, list):
        return ("list", tuple(hash_normalize(x) for x in obj))

    elif isinstance(obj, set):
        return ("set", tuple(sorted(hash_normalize(x) for x in obj)))

    elif isinstance(obj, dict):
        return (
            "dict",
            tuple(
                (hash_normalize(k), hash_normalize(v))
                for k, v in sorted(obj.items(), key=lambda x: repr(x[0]))
            ),
        )

    elif hasattr(obj, "__dict__"):
        return (
            "object",
            obj.__class__.__qualname__,
            hash_normalize(obj.__dict__),
        )

    elif hasattr(obj, "__slots__"):
        return (
            "slots",
            obj.__class__.__qualname__,
            tuple(hash_normalize(getattr(obj, s)) for s in obj.__slots__),
        )

    else:
        # Fallback explícito (mejor que pickle)
        return ("repr", repr(obj))


# def method_is_overridden(cls, method_name):
#     return getattr(cls, method_name) != getattr(cls.__base__, method_name, None)


# def method_is_overridden(cls, method_name):
#     return method_name in cls.__dict__


# def method_is_overridden(obj, method_name):
#     method = getattr(obj, method_name)
#     return isinstance(method, types.MethodType) and method.__self__.__class__ is type(obj)
