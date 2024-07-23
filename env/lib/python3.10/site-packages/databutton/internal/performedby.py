import functools
import os
from datetime import datetime
from typing import Literal, Optional

from databutton.internal.auth import get_auth_token_user
from databutton.pydantic_v1 import BaseModel


class Performer(BaseModel):
    type: Literal["system", "user"]
    id: Optional[str] = None
    name: Optional[str] = None


@functools.cache
def get_performer() -> Performer:
    # Get userId from auth token or systemid from environment.
    # I.e. which user called, or which scheduled job is running.

    # TODO: Set these env vars in devx when calling view/job code
    system_id = os.environ.get("DATABUTTON_SYSTEM_ID")
    user_id = os.environ.get("DATABUTTON_USER_ID")
    user_name = os.environ.get("DATABUTTON_USER_NAME")

    if system_id:
        return Performer(type="system", id=system_id)

    if not user_id:
        user_id, user_name = get_auth_token_user()

    if not user_id:
        raise RuntimeError("Missing user or system id")

    return Performer(type="user", id=user_id, name=user_name)


class PerformedBy(BaseModel):
    type: Literal["system", "user"]
    id: str
    name: Optional[str] = None
    timestamp: datetime


def get_performed_by_now() -> PerformedBy:
    performer = get_performer()
    if performer.id is None:
        raise RuntimeError("Missing performer id.")
    return PerformedBy(
        type=performer.type,
        id=performer.id,
        name=performer.name,
        timestamp=datetime.utcnow(),
    )
