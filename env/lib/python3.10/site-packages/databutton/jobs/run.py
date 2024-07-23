import http

from databutton.internal.dbapiclient import get_dbapi_client
from databutton.internal.retries import default_dbapi_retry


@default_dbapi_retry
def run_soon(job_name: str):
    """Schedule a job to run soon.

    Note: This functionality is not yet complete.
    """
    res = get_dbapi_client().post(
        url=f"/jobs/run-soon/{job_name}",
        json={"trigger": "code"},
    )

    if res.status_code == http.HTTPStatus.ACCEPTED:
        return

    if res.status_code == http.HTTPStatus.NOT_FOUND:
        raise KeyError(f"Job {job_name} not found")

    raise RuntimeError(f"Failed to enqueue job {job_name}")
