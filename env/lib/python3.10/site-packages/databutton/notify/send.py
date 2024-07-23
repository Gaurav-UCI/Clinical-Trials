import http

from databutton.internal.dbapiclient import get_dbapi_client
from databutton.internal.retries import default_dbapi_retry
from databutton.pydantic_v1 import BaseModel


@default_dbapi_retry
def send(message: BaseModel):
    """Internal helper to make request towards notification api."""

    # The idea here is that perhaps we can reuse some notification
    # queue system etc across kinds, e.g. email, slack, etc.
    kind = message.__class__.__name__.lower()

    res = get_dbapi_client().post(
        f"/notify/{kind}",
        json={"message": message.dict()},
    )

    if res.status_code in (
        http.HTTPStatus.OK,
        http.HTTPStatus.ACCEPTED,
    ):
        return

    raise RuntimeError(f"Failed to send {kind} notification")
