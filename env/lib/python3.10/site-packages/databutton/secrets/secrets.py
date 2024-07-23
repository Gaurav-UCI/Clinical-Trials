import http
import re
from typing import List

from databutton.internal.byteutils import (
    base64str_to_bytes,
    base64str_to_str,
    str_to_base64str,
)
from databutton.internal.dbapiclient import get_dbapi_client
from databutton.internal.retries import default_dbapi_retry

invalid_secret_key_error_message = 'Secret key can only consist of letters (A-Z, a-z), digits (0-9), or the symbols "._-".'  # noqa

secret_key_validator = re.compile("^[a-zA-Z0-9-_.]+$")


def secret_key_is_valid(key: str) -> bool:
    return bool(secret_key_validator.match(key))


def key_not_found_msg(name: str) -> str:
    return f"Secret named {name} not found in this app"


@default_dbapi_retry
def put(name: str, value: str | bytes):
    """Add or update a new secret to your app."""

    if not secret_key_is_valid(name):
        raise ValueError(invalid_secret_key_error_message)

    # Note: Api expects base64 encoding of the raw bytes.
    res = get_dbapi_client().post(
        "/secrets/add",
        json={
            "name": name,
            "value": str_to_base64str(value),
        },
    )

    if res.status_code in (
        http.HTTPStatus.OK,
        http.HTTPStatus.CREATED,
    ):
        return

    if res.status_code == http.HTTPStatus.NOT_FOUND:  # 404
        raise KeyError(key_not_found_msg(name))

    raise RuntimeError(f"Failed to add secret {name}, status_code = {res.status_code}")


@default_dbapi_retry
def delete(name: str) -> bool:
    """Delete a secret from your app.

    Returns True if secret was deleted,
    returns False if it was not found,
    raises exception in other cases.
    """
    res = get_dbapi_client().post(
        "/secrets/delete",
        json={"name": name},
    )

    if res.status_code == http.HTTPStatus.OK:
        return True

    if res.status_code == http.HTTPStatus.NOT_FOUND:
        return False

    raise RuntimeError(
        f"Failed to delete secret {name}, status_code = {res.status_code}"
    )


@default_dbapi_retry
def _get_base64(name: str) -> str:
    """Get value of named secret in your app.

    Returns urlsafe base64 encoding of raw bytes value.
    """
    # Using post to pass parameters in body
    res = get_dbapi_client().post(
        "/secrets/get",
        json={"name": name},
    )

    if res.status_code == http.HTTPStatus.OK:
        value_base64 = res.json()["value"]
        return value_base64

    if res.status_code == http.HTTPStatus.NOT_FOUND:
        raise KeyError(key_not_found_msg(name))

    raise RuntimeError(f"Failed to get secret {name}, status_code = {res.status_code}")


def get(name: str) -> str:
    """Get value of named secret in your app.

    Returns value as a str, assuming the secret is
    an utf-8 encoded string.
    """
    return base64str_to_str(_get_base64(name))


def get_as_bytes(name: str) -> bytes:
    """Get value of named secret in your app.

    Returns value as a str, assuming the secret is
    an utf-8 encoded string.
    """
    return base64str_to_bytes(_get_base64(name))


@default_dbapi_retry
def get_names() -> List[str]:
    """Get list of all secret names in your app."""
    res = get_dbapi_client().get("/secrets/list")

    if res.status_code == http.HTTPStatus.OK:
        body = res.json()
        return sorted([s["name"] for s in body["secrets"]])

    raise RuntimeError(f"Failed to list secrets, status_code = {res.status_code}")
