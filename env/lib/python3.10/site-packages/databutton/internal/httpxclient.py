import functools

import httpx


@functools.cache
def get_httpx_client() -> httpx.Client:
    """Cached generic httpx client with no particular properties set.

    Use get_dbapi_client() instead for calls towards the
    databutton-api service.
    """
    return httpx.Client()
