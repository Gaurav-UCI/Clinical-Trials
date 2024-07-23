import http
import os
from time import time
from typing import Generator

import httpx
import jwt

from databutton.internal.httpxclient import get_httpx_client

FIREBASE_API_KEY = "AIzaSyAdgR9BGfQrV2fzndXZLZYgiRtpydlq8ug"


class DatabuttonAuth(httpx.Auth):
    requires_response_body = True
    refresh_interval = 15 * 60

    def __init__(self, api_key, refresh_token):
        self.api_key = api_key
        self.refresh_token = refresh_token
        self.access_token = ""
        self.last_refresh = time() - 2 * self.refresh_interval

    def token_expires_soon(self) -> bool:
        return time() - self.last_refresh > self.refresh_interval

    def build_refresh_request(self) -> httpx.Request:
        return httpx.Request(
            method="POST",
            url=f"https://securetoken.googleapis.com/v1/token?key={self.api_key}",
            data={
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token,
            },
        )

    def handle_refresh_response(self, refresh_response: httpx.Response):
        if refresh_response.status_code == http.HTTPStatus.OK:
            self.access_token = refresh_response.json()["id_token"]
            self.last_refresh = time()
        else:
            raise RuntimeError("Token refresh failed")

    def update_access_token_now(self):
        self.handle_refresh_response(
            get_httpx_client().send(self.build_refresh_request())
        )

    def get_auth_token(self) -> str:
        if not self.access_token:
            self.update_access_token_now()
        return self.access_token

    def apply_access_token(self, request: httpx.Request):
        """Apply authorization header to original request."""
        if not self.access_token:
            raise RuntimeError("databutton auth flow missing access token!")
        request.headers["Authorization"] = f"Bearer {self.access_token}"

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        """This is the function that implements the httpx.Auth interface."""
        # First yield a Request to refresh auth token if it's old
        if self.token_expires_soon():
            refresh_response = yield self.build_refresh_request()
            self.handle_refresh_response(refresh_response)

        # Authorize original request and yield it for execution
        self.apply_access_token(request)
        yield request


def databutton_auth() -> DatabuttonAuth:
    refresh_token: str | None = os.environ.get("DATABUTTON_TOKEN")
    if refresh_token is None:
        raise RuntimeError("Missing databutton refresh token.")

    return DatabuttonAuth(
        api_key=FIREBASE_API_KEY,
        refresh_token=refresh_token,
    )


def get_auth_token() -> str:
    return databutton_auth().get_auth_token()


def get_auth_token_user() -> tuple[str | None, str | None]:
    """Extract user id and name from firebase auth token."""
    token_claims = jwt.decode(
        get_auth_token(),
        options={"verify_signature": False},
    )
    user_id = token_claims.get("sub")
    user_name = token_claims.get("name")
    return user_id, user_name
