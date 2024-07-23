import base64
import hashlib
import http
import json
import re
import tempfile
from abc import ABC
from collections.abc import Iterable
from enum import Enum
from typing import Any, Optional

import httpx
import pandas as pd

from databutton.internal.dbapiclient import get_dbapi_client
from databutton.internal.httpxclient import get_httpx_client
from databutton.internal.performedby import PerformedBy, get_performed_by_now
from databutton.internal.retries import default_dbapi_retry
from databutton.pydantic_v1 import BaseModel

# This could be tuned
CHUNKSIZE = 4 * 1024 * 1024

invalid_data_key_error_message = 'Data key can only consist of letters (A-Z, a-z), digits (0-9), or the symbols "._-".'  # noqa

data_key_validator = re.compile("^[a-zA-Z0-9-_.]+$")


def data_key_is_valid(key: str) -> bool:
    return bool(data_key_validator.match(key))


class ContentTypes(str, Enum):
    arrow = "vnd.apache.arrow.file"
    json = "application/json"
    text = "text/plain"
    binary = "application/octet-stream"


class Property(BaseModel):
    name: str | None
    dtype: str | None


class ContentShape(BaseModel):
    numberOfRows: int
    numberOfProperties: int
    properties: list[Property] | None = None


class Serializer(ABC):
    content_type: str = ContentTypes.binary

    def content_shape_of(self, value: Any) -> Optional[ContentShape]:
        return None

    def encode_into(self, value: Any, data_file: tempfile.SpooledTemporaryFile) -> None:
        raise NotImplementedError("Must be implemented by serializer.")

    def decode_from(self, data_file: tempfile.SpooledTemporaryFile) -> Any:
        raise NotImplementedError("Must be implemented by serializer.")

    # TODO: Can probably delete encode/decode now
    def encode(self, value: Any) -> Iterable[bytes]:
        raise NotImplementedError("Must be implemented by serializer.")

    def decode(self, data: Iterable[bytes]) -> Any:
        raise NotImplementedError("Must be implemented by serializer.")


class BinarySerializer(Serializer):
    content_type = ContentTypes.binary

    def encode_into(
        self, value: bytes, data_file: tempfile.SpooledTemporaryFile
    ) -> None:
        data_file.write(value)

    def decode_from(self, data_file: tempfile.SpooledTemporaryFile) -> bytes:
        return data_file.read()

    def encode(self, value: bytes) -> Iterable[bytes]:
        yield value

    def decode(self, data: Iterable[bytes]) -> bytes:
        return b"".join(data)


class TextSerializer(Serializer):
    content_type = ContentTypes.text

    def encode_into(self, value: str, data_file: tempfile.SpooledTemporaryFile) -> None:
        data_file.write(value.encode("utf-8"))

    def decode_from(self, data_file: tempfile.SpooledTemporaryFile) -> str:
        return data_file.read().decode("utf-8", errors="strict")

    def encode(self, value: str) -> Iterable[bytes]:
        yield value.encode("utf-8")

    def decode(self, data: Iterable[bytes]) -> str:
        return b"".join(data).decode(encoding="utf-8", errors="strict")


class JsonSerializer(Serializer):
    content_type = ContentTypes.json

    def encode_into(
        self, value: dict, data_file: tempfile.SpooledTemporaryFile
    ) -> None:
        # TODO: Let serializer determine file open more text or binary
        # json.dump(value, data_file)
        data_file.write(json.dumps(value).encode("utf-8"))

    def decode_from(self, data_file: tempfile.SpooledTemporaryFile) -> dict:
        return json.load(fp=data_file)

    def encode(self, value: dict) -> Iterable[bytes]:
        yield json.dumps(value).encode("utf-8")

    def decode(self, data: Iterable[bytes]) -> dict:
        return json.loads(b"".join(data))


class DataFrameSerializer(Serializer):
    content_type = ContentTypes.arrow

    def content_shape_of(self, value: pd.DataFrame) -> Optional[ContentShape]:
        rows, cols = value.shape
        props = [Property(name=str(k), dtype=str(v)) for k, v in value.dtypes.items()]
        return ContentShape(
            numberOfRows=rows,
            numberOfProperties=cols,
            properties=props,
        )

    def encode_into(
        self, value: pd.DataFrame, data_file: tempfile.SpooledTemporaryFile
    ) -> None:
        value.to_feather(data_file)

    def decode_from(self, data_file: tempfile.SpooledTemporaryFile) -> pd.DataFrame:
        return pd.read_feather(data_file)

    def encode(self, value: pd.DataFrame) -> Iterable[bytes]:
        with tempfile.SpooledTemporaryFile(mode="w+b") as data_file:
            # Serialize entire file, possibly to file if it's large
            value.to_feather(data_file)

            # Yield chunks from file
            data_file.seek(0)
            while chunk := data_file.read(CHUNKSIZE):
                yield chunk

    def decode(self, data: Iterable[bytes]) -> pd.DataFrame:
        with tempfile.SpooledTemporaryFile(mode="w+b") as data_file:
            for chunk in data:
                data_file.write(chunk)
            data_file.seek(0)
            df = pd.read_feather(data_file)
            return df


class FileListEntry(BaseModel):
    name: str
    size: int


class FileList(BaseModel):
    items: list[FileListEntry]


@default_dbapi_retry
def list_files(
    *,
    content_type: str,
) -> list[FileListEntry]:
    res = get_dbapi_client().post(
        url="/storage/list",
        json={
            "contentType": content_type,
        },
    )

    if res.status_code == http.HTTPStatus.OK:
        return FileList.parse_raw(res.content).items

    raise RuntimeError(f"File list request failed, status_code={res.status_code}")


def delete_file(
    *,
    data_key: str,
    content_type: str,
):
    res = get_dbapi_client().post(
        url="/storage/delete",
        json={
            "dataKey": data_key,
            "contentType": content_type,
        },
    )

    if res.status_code == http.HTTPStatus.OK:
        return

    if res.status_code == http.HTTPStatus.NOT_FOUND:
        # Silently ignore
        return

    raise RuntimeError(f"File delete request failed, status_code={res.status_code}")


class PrepareRequest(BaseModel):
    uploadedBy: PerformedBy
    dataKey: str
    contentType: str
    contentLength: int
    contentMd5Checksum: str
    contentShape: ContentShape | None = None


class PrepareResponse(BaseModel):
    sessionUrl: str
    blobKey: str


@default_dbapi_retry
def prepare_upload(
    prepare_request: PrepareRequest,
) -> PrepareResponse:
    """Send metadata to db-api to prepare for blob upload."""
    res = get_dbapi_client().post(
        url="/storage/prepare",
        content=prepare_request.json(),
        headers={
            "Content-Type": "application/json",
        },
    )

    if res.status_code == http.HTTPStatus.OK:
        return PrepareResponse.parse_raw(res.content)

    raise RuntimeError(
        f"Upload preparations for '{prepare_request.dataKey}' failed, status_code={res.status_code}"
    )


# Note: Maybe we want to tune this retry differently
@default_dbapi_retry
def gcs_upload(
    data_key: str,
    data_file: tempfile.SpooledTemporaryFile,
    content_length: int,
    session_url: str,
    blob_key: str,
):
    """Upload data file to google cloud storage via session url"""
    # Note: For huge files, using Content-Range headers and looping could be less fragile.
    data_file.seek(0)
    res = get_httpx_client().put(
        url=session_url,
        headers={
            "Content-Length": str(content_length),
        },
        content=data_file,
        timeout=httpx.Timeout(timeout=20.0, connect=30.0),
    )

    if res.status_code == http.HTTPStatus.OK:
        return

    raise RuntimeError(
        f"Upload of '{data_key}' to '{blob_key}' failed, status_code={res.status_code}"
    )


class GetUrlResponse(BaseModel):
    signedUrl: str
    md5: str
    size: int


@default_dbapi_retry
def get_download_url(
    *,
    data_key: str,
    content_type: str,
):
    # Get metadata and download url for current version from db-api
    res = get_dbapi_client().post(
        url="/storage/geturl",
        json={
            "dataKey": data_key,
            "contentType": content_type,
        },
    )

    if res.status_code == http.HTTPStatus.OK:
        return GetUrlResponse.parse_raw(res.content)

    if res.status_code == http.HTTPStatus.NOT_FOUND:
        raise FileNotFoundError(f"{data_key} not found")

    raise RuntimeError(
        f"Download preparations for '{data_key}' failed, status_code={res.status_code}"
    )


def stream_download_response_to_file(resp, data_file):
    """Stream response bytes to tempfile in chunks, hash and count while at it."""
    actual_size = 0
    hashalg = hashlib.md5()
    for chunk in resp.iter_bytes(chunk_size=CHUNKSIZE):
        # This line is the actual data write from response to disk
        data_file.write(chunk)
        # Compute hash and actual size chunk wise
        hashalg.update(chunk)
        actual_size += len(chunk)
    actual_md5_hash = base64.b64encode(hashalg.digest()).decode("utf-8")
    return actual_md5_hash, actual_size


def gcs_download(
    *,
    data_key: str,
    geturl_response: GetUrlResponse,
    data_file: tempfile.SpooledTemporaryFile,
):
    """Download from signed gcs url into temp file."""

    resp = get_httpx_client().get(
        url=geturl_response.signedUrl,
        timeout=httpx.Timeout(timeout=20.0, connect=30.0),
    )

    if resp.status_code == http.HTTPStatus.OK:
        actual_md5_hash, actual_size = stream_download_response_to_file(resp, data_file)

        # Sanity checking
        if geturl_response.size != actual_size:
            raise RuntimeError(
                f"File sizes do not match: {geturl_response.size} != {actual_size}"
            )
        if geturl_response.md5 != actual_md5_hash:
            raise RuntimeError(
                f"Checksums do not match: {geturl_response.md5} != {actual_md5_hash}"
            )
        return

    if resp.status_code == http.HTTPStatus.NOT_FOUND:
        raise FileNotFoundError(f"{data_key} not found")

    raise RuntimeError(
        f"Download of '{data_key}' failed, status_code={resp.status_code}, url={geturl_response.signedUrl}"
    )


def serialized_upload(
    serializer: Serializer,
    key: str,
    value: Any,
):
    if not data_key_is_valid(key):
        raise ValueError(invalid_data_key_error_message)

    with tempfile.SpooledTemporaryFile(mode="w+b") as data_file:
        # Serialize entire file (possibly to file if it's large)
        serializer.encode_into(value, data_file)

        # Compute size and md5 from serialized bytes
        # (could probably be streamlined to avoid iterating over file again)
        hashalg = hashlib.md5()
        size = 0
        data_file.seek(0)
        while chunk := data_file.read(CHUNKSIZE):
            size += len(chunk)
            hashalg.update(chunk)
        md5 = base64.b64encode(hashalg.digest()).decode("utf-8")

        # To implement client side encryption we would add something like this here:
        # 1. generate a random dek (data encryption key)
        # dek = generate_key()
        # 2. encrypt dek with a call to a databutton-keyring service wrapping kms
        # encrypted_dek = encrypt_key(dek)
        # 3. encrypt bytes_value with dek before storing in blob
        # bytes_value = encrypt_user_data(bytes_value, encryption_key)
        # 4. store encrypted dek in metadata

        # Make request to db-api with metadata,
        # returning a session url to upload to
        prepare_response = prepare_upload(
            PrepareRequest(
                uploadedBy=get_performed_by_now(),
                dataKey=key,
                contentType=serializer.content_type,
                contentLength=size,
                contentMd5Checksum=md5,
                contentShape=serializer.content_shape_of(value),
            )
        )

        # Upload to sessio url from file (it could still be in memory if it's small)
        gcs_upload(
            data_key=key,
            data_file=data_file,
            content_length=size,
            session_url=prepare_response.sessionUrl,
            blob_key=prepare_response.blobKey,
        )


def serialized_download(
    serializer: Serializer,
    key: str,
    default: Optional[Any] = None,
) -> Optional[Any]:
    try:
        # Get a signed download url
        geturl_response = get_download_url(
            data_key=key,
            content_type=serializer.content_type,
        )

        with tempfile.SpooledTemporaryFile(mode="w+b") as data_file:
            # Download into tempfile (if it's small it stays in memory)
            gcs_download(
                data_key=key,
                geturl_response=geturl_response,
                data_file=data_file,
            )

            # To implement client side encryption we would add this here:
            # 1. get encrypted dek (data encryption key) from geturl metadata
            # 2. decrypt dek with a call to a databutton-keyring service wrapping kms
            # 3. decrypt bytes_value with dek
            # if encrypted_dek is not None:
            #     bytes_value = decrypt_user_data(encrypted_dek, bytes_value, aad=something)

            # Now decode from tempfile
            data_file.seek(0)
            value = serializer.decode_from(data_file)
    except FileNotFoundError:
        if default is not None:
            if callable(default):
                return default()
            return default
        raise
    return value


class BinaryStorage:
    """Manage storage of raw binary files."""

    Serializer = BinarySerializer

    def __init__(self):
        self.serializer = self.Serializer()

    def put(self, key: str, value: bytes):
        """Store bytes with given key."""
        serialized_upload(self.serializer, key=key, value=value)

    def get(self, key: str, *, default: Optional[bytes] = None) -> Optional[bytes]:
        """Get bytes value at given key."""
        return serialized_download(serializer=self.serializer, key=key, default=default)

    def delete(self, key: str):
        """Delete binary file with given key."""
        delete_file(data_key=key, content_type=self.serializer.content_type)

    def list(self) -> list[FileListEntry]:
        """List all binary files."""
        return list_files(content_type=self.serializer.content_type)


class TextStorage:
    """Manage storage of plain text files."""

    Serializer = TextSerializer

    def __init__(self):
        self.serializer = self.Serializer()

    def put(self, key: str, value: str):
        """Store text with given key."""
        serialized_upload(self.serializer, key=key, value=value)

    def get(self, key: str, *, default: Optional[str] = None) -> Optional[str]:
        """Get text value at given key."""
        return serialized_download(serializer=self.serializer, key=key, default=default)

    def delete(self, key: str):
        """Delete text file with given key."""
        delete_file(data_key=key, content_type=self.serializer.content_type)

    def list(self) -> list[FileListEntry]:
        """List all text files."""
        return list_files(content_type=self.serializer.content_type)


class JsonStorage:
    """Manage storage of json files, assumed to be a dict on the python side."""

    Serializer = JsonSerializer

    def __init__(self):
        self.serializer = self.Serializer()

    def put(self, key: str, value: dict):
        """Store json-compatible dict with given key.

        Dict keys must be strings, valid values can be anything that is default
        serializable as json such as strings, numbers, or nested lists or dicts
        with valid values. Be aware that floating point values are not stored
        exactly as json.
        """
        serialized_upload(self.serializer, key=key, value=value)

    def get(self, key: str, *, default: Optional[dict] = None) -> Optional[dict]:
        """Get dict stored as json with given key."""
        return serialized_download(serializer=self.serializer, key=key, default=default)

    def delete(self, key: str):
        """Delete json file with given key."""
        delete_file(data_key=key, content_type=self.serializer.content_type)

    def list(self) -> list[FileListEntry]:
        """List all json files."""
        return list_files(content_type=self.serializer.content_type)


class DataFramesStorage:
    """Manage storage of pandas dataframes as arrow files."""

    Serializer = DataFrameSerializer

    def __init__(self):
        self.serializer = self.Serializer()

    def put(
        self,
        key: str,
        value: Optional[pd.DataFrame] = None,
        *,
        # This is kept because of backwards-compat
        df: Optional[pd.DataFrame] = None,
        persist_index: bool = False,
    ):
        """Store a dataframe under key in your Databutton project storage.

        Usage:
            db.storage.dataframes.put("key", mydataframe)
            db.storage.dataframes.put(key="key", value=mydataframe)

        Deprecated notation:
            db.storage.dataframes.put(mydataframe, "key")
            db.storage.dataframes.put(key="key", df=mydataframe)
        """
        # Backwards compatibility: Exactly one of value and df must be provided
        if value is not None and df is not None:
            raise ValueError(
                "'df' is provided for backwards compatibility, use only 'value'"
            )
        if value is None and df is None:
            raise ValueError("Missing 'value' argument")

        # Backwards compatibility: If df is provided, use it and ask user to switch
        if value is None:  # implies df is not None because of above checks
            # TODO: Perhaps we should have some sort of a side channel for
            #  deprecation warnings, so we can measure it and also avoid
            #  polluting user code output?
            print("Deprecation warning: use 'value' instead of 'df'.")
            value, df = df, None

        # Backwards compatibility: If (value, key) is provided, swap them and ask user to switch
        if isinstance(key, pd.DataFrame) and isinstance(value, str):
            print("Deprecation warning: Swap put(value, key) -> put(key, value).")
            key, value = value, key

        if not persist_index and value is not None:
            value = value.reset_index(drop=True)

        serialized_upload(self.serializer, key=key, value=value)

        return True  # From old implementation

    def get(
        self,
        key: str,
        *,
        ignore_not_found: bool = True,
        default: Optional[pd.DataFrame] = None,
    ) -> Optional[pd.DataFrame]:
        """Get dataframe with given key."""
        if default is None and ignore_not_found is True:

            def empty_dataframe() -> pd.DataFrame:
                return pd.DataFrame()

            default = empty_dataframe  # type: ignore
        return serialized_download(serializer=self.serializer, key=key, default=default)

    def concat(
        self,
        key: str,
        other: pd.DataFrame,
        *,
        ignore_index: bool = False,
        verify_integrity: bool = False,
        sort: bool = False,
    ) -> pd.DataFrame:
        try:
            df = self.get(key=key, ignore_not_found=False)
        except FileNotFoundError:
            new_df = other
        else:
            new_df = pd.concat(
                [df, other],  # type: ignore
                ignore_index=ignore_index,
                verify_integrity=verify_integrity,
                sort=sort,
            )
        self.put(key=key, value=new_df)
        return new_df

    def add(self, key: str, entry: Any) -> pd.DataFrame:
        return self.concat(
            key=key, other=pd.DataFrame(entry, index=[0]), ignore_index=True
        )

    def clear(self, key: str):
        """Empty the data at a certain key, leaving you with an empty dataframe on the next get."""
        return self.put(key=key, value=pd.DataFrame(data=[]).reset_index())

    def delete(self, key: str):
        """Delete dataframe with given key."""
        delete_file(data_key=key, content_type=self.serializer.content_type)

    def list(self) -> list[FileListEntry]:
        """List all dataframes."""
        return list_files(content_type=self.serializer.content_type)
