import tenacity
from httpx import NetworkError, TimeoutException

# List to be tuned!
# retryable_status_codes = (
#     http.HTTPStatus.REQUEST_TIMEOUT,  # 408
#     http.HTTPStatus.CONFLICT,  # 409
#     http.HTTPStatus.LOCKED,  # 423
#     http.HTTPStatus.TOO_EARLY,  # 425 Should delay further retries?
#     http.HTTPStatus.TOO_MANY_REQUESTS,  # 429 Should delay further retries?
#     http.HTTPStatus.INTERNAL_SERVER_ERROR,  # 500
#     http.HTTPStatus.BAD_GATEWAY,  # 502
#     http.HTTPStatus.SERVICE_UNAVAILABLE,  # 503
#     http.HTTPStatus.GATEWAY_TIMEOUT,  # 504
# )


def is_retryable_exception(e: BaseException) -> bool:
    # This covers the read timeouts we've seen in practice
    if isinstance(e, TimeoutException):
        return True

    # This covers some additional network errors
    if isinstance(e, NetworkError):
        return True

    # Explicit request for retry
    if isinstance(e, tenacity.TryAgain):
        return True

    # TODO: When rate limited we should respect Retry-After headers
    #   for 425 and 429 responses, but it's not obvious to me how to
    #   implement this with tenacity, since they've decoupled the
    #   "should retry" and "how long to wait" parts.
    # if isinstance(e, HTTPException):
    #     # if e.headers:
    #     #     retry_after = e.headers.get("Retry-After")
    #     return e.status_code in retryable_status_codes

    # Let the rest fail
    return False


# def before_sleep_sentry_log(retry_state: tenacity.RetryCallState):
#     """Log retries to sentry."""
#     try:
#         import sentry_sdk
#         sentry_sdk.capture_message(
#             f"Pending retry after {retry_state.attempt_number} attempts"
#         )
#     except:
#         pass


default_dbapi_retry = tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_random_exponential(multiplier=0.5),
    retry=tenacity.retry_if_exception(predicate=is_retryable_exception),
    reraise=True,
    # before_sleep=before_sleep_sentry_log,
)
