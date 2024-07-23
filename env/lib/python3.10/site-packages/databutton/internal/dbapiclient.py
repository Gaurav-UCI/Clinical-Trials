import functools
import os

import httpx

from databutton import version
from databutton.internal.auth import databutton_auth


def get_databutton_project_id() -> str:
    project_id: str | None = os.environ.get("DATABUTTON_PROJECT_ID")
    if not project_id:
        raise EnvironmentError(
            "Failed to determine databutton project id. Make sure the DATABUTTON_PROJECT_ID environment variable is set."
        )
    return project_id


def get_dbapi_url() -> str:
    u = os.environ.get("DATABUTTON_API_URL")
    if u is None:
        project_id = get_databutton_project_id()
        u = f"https://api.databutton.com/_projects/{project_id}/dbtn"
    return u


def is_localdev() -> bool:
    # TODO: Previously used in workaround for local testing, get local testing working again
    return get_dbapi_url().startswith("http://localhost")


@functools.cache
def get_dbapi_client() -> httpx.Client:
    """Cached httpx client for the databutton-api service.

    With base url and auth middleware set.

    Use get_httpx_client() instead for a cached client
    for calls towards generic endpoints.
    """

    if is_localdev():
        headers = {
            "X-DATABUTTON-RELEASE": os.environ.get("DATABUTTON_RELEASE", "") + "-dev",
            "X-DATABUTTON-DEVX-VERSION": os.environ.get("DEVX_VERSION", "") + "-dev",
            "X-DATABUTTON-SDK-VERSION": (version.__version__ or "local") + "-dev",
            "X-DATABUTTON-PROJECT-ID": os.environ.get("DATABUTTON_PROJECT_ID", ""),
        }
    else:
        headers = {
            "X-DATABUTTON-RELEASE": os.environ.get("DATABUTTON_RELEASE", "dev"),
            "X-DATABUTTON-DEVX-VERSION": os.environ.get("DEVX_VERSION", ""),
            "X-DATABUTTON-SDK-VERSION": version.__version__ or "",
        }

    return httpx.Client(
        base_url=get_dbapi_url(),
        headers=headers,
        timeout=httpx.Timeout(20.0, connect=30.0),
        auth=databutton_auth(),
    )
