import re
from datetime import datetime, timedelta

import httpx
import jwt
from cryptography.x509 import load_pem_x509_certificate
from fastapi import HTTPException, Request, status
from pydantic import BaseModel

import databutton as db


class GooglePublicKeySet:
    def __init__(self):
        self.exp = None
        self.keys = None

    def get_keys(self):
        # Load certs from Google
        response = httpx.get(
            "https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com"
        )

        x509_certs: dict = response.json()

        # Extract the public keys
        keys = {}
        for key, value in x509_certs.items():
            cert_obj = load_pem_x509_certificate(bytes(value, "utf-8"))
            keys[key] = cert_obj.public_key()

        # Set expiration time for the keys
        cache_control = response.headers.get("cache-control", "")
        max_age_match = re.search(r"max-age=(\d+)", cache_control)
        max_age = int(max_age_match.group(1))
        exp = datetime.now() + timedelta(seconds=max_age)

        self.exp = exp
        self.keys = keys

    def get_key(self, kid: str) -> str:
        if self.keys is None:
            self.get_keys()
        if datetime.now() > self.exp:
            self.get_keys()
        assert self.keys is not None
        return self.keys.get(kid)


public_key_set = GooglePublicKeySet()


class FirebaseUser(BaseModel):
    user_id: str
    email: str
    name: str | None
    picture: str | None


_firebase_project_name: str | None = None


def get_firebase_user(request: Request) -> FirebaseUser:
    # Get auth token
    global _firebase_project_name
    firebase_project_name = _firebase_project_name or db.secrets.get(
        "FIREBASE_PROJECT_NAME"
    )
    auth_header = request.headers.get("x-authorization")
    if not firebase_project_name:
        print(
            "Could not find firebase project name. Make sure to set the FIREBASE_PROJECT_NAME secret."
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not find firebase project name.",
        )
    if not auth_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No x-authorization header provided",
        )

    token = auth_header.replace("Bearer", "").strip()

    # Get KID from unverified auth token
    try:
        unverified = jwt.get_unverified_header(token)
        kid = unverified.get("kid")

        # Lookup public key with that KID
        public_key = public_key_set.get_key(kid)

        # Verify token with public key from Google
        data = jwt.decode(
            token, public_key, algorithms=["RS256"], audience=firebase_project_name
        )

        user = FirebaseUser(
            name=data.get("name"),
            picture=data.get("picture"),
            user_id=str(data.get("user_id")),
            email=str(data.get("email")),
        )
        return user

    except Exception as e:
        print(f"Could not authenticate request: error {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not authenticate request",
        )


class UserResponse(BaseModel):
    user: FirebaseUser
