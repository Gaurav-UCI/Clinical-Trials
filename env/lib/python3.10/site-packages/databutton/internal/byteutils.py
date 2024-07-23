import base64


def stringlike_bytes(value: str | bytes) -> bytes:
    """Return bytes or utf-8 encoding of str."""
    if isinstance(value, bytes):
        return value
    return value.encode("utf-8")


def str_to_base64str(value: str | bytes) -> str:
    """Encode a string with utf-8 and urlsafe base64."""
    return base64.urlsafe_b64encode(stringlike_bytes(value)).decode("utf-8")


def base64str_to_str(value_base64: str) -> str:
    """Decode a base64 string and return it as a utf-8 string."""
    return base64.urlsafe_b64decode(value_base64).decode("utf-8")


def base64str_to_bytes(value_base64: str) -> bytes:
    """Decode a base64 string and return it as a bytes string."""
    return base64.urlsafe_b64decode(value_base64)
