import base64
import io
import mimetypes
import pathlib
import re
from collections.abc import Sequence
from typing import Any, List, Mapping, Optional, Union

import pandas as pd

from databutton.pydantic_v1 import BaseModel

from .send import send


def valid_email(recipient: str) -> bool:
    # Note: We could possibly use some email validation library but it's tricky
    parts = recipient.split("@")
    if len(parts) != 2:
        return False
    return bool(parts[0] and parts[1])


def validate_email_to_arg(to: Union[str, List[str]]) -> List[str]:
    if isinstance(to, str):
        to = [to]
    if not isinstance(to, (list, tuple)) and len(to) > 0:
        raise ValueError(
            "Invalid recipient, expecting 'to' to be a string or list of strings."
        )
    invalid_emails = []
    for recipient in to:
        if not valid_email(recipient):
            invalid_emails.append(recipient)
    if invalid_emails:
        raise ValueError("\n".join(["Invalid email address(es):"] + invalid_emails))
    return to


# This is the type expected in the api
class Attachment(BaseModel):
    """An attachment to be included with an email."""

    # Attachment file name
    file_name: Optional[str] = None

    # MIME type of the attachment
    content_type: Optional[str] = None

    # Content ID (CID) to use for inline attachments
    content_id: Optional[str] = None

    # Base64 encoded data
    content_base64: str


# This is the type expected in the api
class Email(BaseModel):
    to: Union[str, List[str]]
    subject: str
    content_text: Optional[str] = None
    content_html: Optional[str] = None
    attachments: list[Attachment] = []


def determine_type(type: Optional[str], name: Optional[str]) -> Optional[str]:
    if type:
        return type
    if name:
        type, encoding = mimetypes.guess_type(name)
        # if encoding is not None:
        #     return "; ".join([type, encoding])
        return type
    return None


def encode_content(content: bytes | str) -> str:
    if isinstance(content, str):
        content = content.encode()
    return base64.b64encode(content).decode()


def attachment_from_bytes(
    content: bytes,
    *,
    file_name: Optional[str] = None,
    content_type: Optional[str] = None,
    cid: Optional[str] = None,
) -> Attachment:
    """Create attachment with content as raw bytes.

    You can optionally provide a file name and/or content type.

    If missing we will try to guess the content type from the file name.

    To use an attachment as an inline image in the email,
    set the `cid="my_image_id"` parameter,
    and use `<img src="cid:my_image_id">` in the html content.
    """
    return Attachment(
        file_name=file_name,
        content_type=determine_type(content_type, file_name),
        content_base64=encode_content(content),
        content_id=cid,
    )


def attachment_from_str(
    content: str,
    *,
    file_name: Optional[str] = None,
    content_type: Optional[str] = None,
    cid: Optional[str] = None,
) -> Attachment:
    """Create attachment with content as raw str."""
    return attachment_from_bytes(
        content.encode(),
        file_name=file_name,
        content_type=content_type or "text/plain",
        cid=cid,
    )


def attachment_from_file(
    fp: Optional[io.IOBase] = None,
    *,
    file_name: Optional[str] = None,
    content_type: Optional[str] = None,
    cid: Optional[str] = None,
) -> Attachment:
    """Create attachment with content from a file.

    fp can be anything with a .read() method returning bytes or str,
    or omitted to read file_name from file system.
    """
    if fp is None:
        if file_name is None:
            raise ValueError("Either `fp` or `file_name` must be provided.")
        with open(file_name, "rb") as fp:
            buf = fp.read()
    else:
        buf = fp.read()
        if isinstance(buf, str):
            buf = buf.encode()
    return attachment_from_bytes(
        buf,
        file_name=file_name,
        content_type=content_type,
        cid=cid,
    )


# Copied from PIL.Image.registered_extensions() in a recent pillow installation (trying to avoid pillow dependency)
PIL_EXTENSIONS = {
    ".gif": "GIF",
    ".bmp": "BMP",
    ".dib": "DIB",
    ".jfif": "JPEG",
    ".jpe": "JPEG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".pbm": "PPM",
    ".pgm": "PPM",
    ".ppm": "PPM",
    ".pnm": "PPM",
    ".png": "PNG",
    ".apng": "PNG",
    ".blp": "BLP",
    ".bufr": "BUFR",
    ".cur": "CUR",
    ".pcx": "PCX",
    ".dcx": "DCX",
    ".dds": "DDS",
    ".ps": "EPS",
    ".eps": "EPS",
    ".fit": "FITS",
    ".fits": "FITS",
    ".fli": "FLI",
    ".flc": "FLI",
    ".ftc": "FTEX",
    ".ftu": "FTEX",
    ".gbr": "GBR",
    ".grib": "GRIB",
    ".h5": "HDF5",
    ".hdf": "HDF5",
    ".jp2": "JPEG2000",
    ".j2k": "JPEG2000",
    ".jpc": "JPEG2000",
    ".jpf": "JPEG2000",
    ".jpx": "JPEG2000",
    ".j2c": "JPEG2000",
    ".icns": "ICNS",
    ".ico": "ICO",
    ".im": "IM",
    ".iim": "IPTC",
    ".mpg": "MPEG",
    ".mpeg": "MPEG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".mpo": "MPO",
    ".msp": "MSP",
    ".palm": "PALM",
    ".pcd": "PCD",
    ".pdf": "PDF",
    ".pxr": "PIXAR",
    ".psd": "PSD",
    ".qoi": "QOI",
    ".bw": "SGI",
    ".rgb": "SGI",
    ".rgba": "SGI",
    ".sgi": "SGI",
    ".ras": "SUN",
    ".tga": "TGA",
    ".icb": "TGA",
    ".vda": "TGA",
    ".vst": "TGA",
    ".webp": "WEBP",
    ".wmf": "WMF",
    ".emf": "WMF",
    ".xbm": "XBM",
    ".xpm": "XPM",
}


def attachment_from_pil_image(
    image,  # PIL image
    *,
    file_name: Optional[str] = None,
    content_type: Optional[str] = None,
    cid: Optional[str] = None,
    pil_kwargs: Optional[Mapping[str, Any]] = None,
) -> Attachment:
    """Create image attachment with content from a PIL compatible image (such as pillow).

    This convenience function calls the PIL.Image.save function to create an image file,
    with format determined by the extension of the file name.
    Defaults to .jpeg if not file name is given.

    Additional arguments to the save function can be passed in pil_kwargs.

    If missing we will try to guess the content type from the file name.

    To use an attachment as an inline image in the email,
    set the `cid="my_image_id"` parameter,
    and use `<img src="cid:my_image_id">` in the html content.
    """

    if pil_kwargs is None:
        pil_kwargs = {}
    else:
        pil_kwargs = dict(pil_kwargs)

    ext = pathlib.Path(file_name).suffix.lower() if file_name else ""

    format = PIL_EXTENSIONS.get(ext, "JPEG")
    if pil_kwargs.get("format") and pil_kwargs.get("format") != format:
        raise ValueError(
            f"File extension {ext} and format {format} are not compatible."
        )
    pil_kwargs["format"] = format

    # Fill in content type if missing and known by mimetypes
    if content_type is None and ext:
        content_type, encoding = mimetypes.guess_type("f" + ext)
        if content_type is not None:
            pil_kwargs["content_type"] = content_type

    buf = io.BytesIO()
    image.save(buf, **pil_kwargs)
    return attachment_from_bytes(
        buf.getvalue(),
        file_name=file_name,
        content_type=content_type,
        cid=cid,
    )


def attachment_from_dataframe(
    df: pd.DataFrame,
    *,
    file_name: Optional[str] = None,
    content_type: Optional[str] = None,
    cid: Optional[str] = None,
    pandas_kwargs: Optional[Mapping[str, Any]] = None,
) -> Attachment:
    """Create attachment with content from a pandas dataframe.

    File name extension is used to determine format to use, defaults to .csv if omitted.
    Writing to .xlsx requires the openpyxl package to be installed.

    Supported formats include:

        - .csv
        - .xlsx
        - .feather
        - .parquet
        - .orc

    """
    buf = io.BytesIO()

    ext = pathlib.Path(file_name).suffix if file_name else ".csv"

    if pandas_kwargs is None:
        pandas_kwargs = {}

    if content_type is None:
        content_type, encoding = mimetypes.guess_type("f" + ext)

    if ext == ".csv":
        df.to_csv(buf, **pandas_kwargs)
    elif ext == ".xlsx":
        df.to_excel(buf, **pandas_kwargs)
    elif ext == ".feather":
        df.to_feather(buf, **pandas_kwargs)
    elif ext == ".parquet":
        df.to_parquet(buf, **pandas_kwargs)
    elif ext == ".orc":
        df.to_orc(buf, **pandas_kwargs)
    else:
        raise ValueError(f"Unknown file extension {ext} for dataframe")

    return attachment_from_bytes(
        buf.getvalue(),
        file_name=file_name,
        content_type=content_type,
        cid=cid,
    )


def attachment(
    content: Optional[Any] = None,
    *,
    file_name: Optional[str] = None,
    content_type: Optional[str] = None,
    cid: Optional[str] = None,
):
    if isinstance(content, bytes):
        return attachment_from_bytes(
            content,
            file_name=file_name,
            content_type=content_type,
            cid=cid,
        )

    if isinstance(content, str):
        return attachment_from_str(
            content,
            file_name=file_name,
            content_type=content_type,
            cid=cid,
        )

    # DataFrame
    if content.__class__.__name__ == "DataFrame":
        return attachment_from_dataframe(
            content,  # type: ignore
            file_name=file_name,
            content_type=content_type,
            cid=cid,
        )

    # PIL (pillow) image
    if content.__class__.__module__.split(".")[0] == "PIL":
        return attachment_from_pil_image(
            content,
            file_name=file_name,
            content_type=content_type,
            cid=cid,
        )

    # No content, read from filename
    if content is None and file_name:
        return attachment_from_file(
            file_name=file_name,
            content_type=content_type,
            cid=cid,
        )

    # Fallback to something file-like with a read function
    if callable(getattr(content, "read", None)):
        return attachment_from_file(
            content,
            file_name=file_name,
            content_type=content_type,
            cid=cid,
        )

    raise ValueError(f"Don't know how to handle content of type {repr(type(content))}")


def validate_attachment(att: Attachment) -> Attachment:
    # assert isinstance(att, Attachment)
    # assert att.content_type
    # assert att.file_name
    assert att.content_base64
    assert isinstance(att.content_base64, str)
    assert re.match(r"^[A-Za-z0-9+/=]+$", att.content_base64)
    return att


def create_email(
    *,
    to: Union[str, List[str]],
    subject: str,
    content_text: Optional[str] = None,
    content_html: Optional[str] = None,
    attachments: Sequence[Attachment] = (),
) -> Email:
    attachments = [validate_attachment(att) for att in attachments]

    # Sendgrid has a 30 MB limit on everything, this estimate should be slightly stricter
    size = (
        len(content_html or "")
        + len(content_text or "")
        + sum([len(att.content_base64) for att in attachments])
    )
    max = 30 * 1024**2  # 30 MB
    headroom = 100 * 1024  # leave some room for headers etc
    if size > max - headroom:
        raise ValueError(
            "Email and attachment size exceeds 30MB, please reduce the size of the email."
        )

    return Email(
        to=validate_email_to_arg(to),
        subject=subject,
        content_text=content_text,
        content_html=content_html,
        attachments=attachments,
    )


def email(
    to: Union[str, List[str]],
    subject: str,
    *,
    content_text: Optional[str] = None,
    content_html: Optional[str] = None,
    attachments: Sequence[Attachment] = (),
):
    """Send email notification from databutton.

    At least one of the content arguments must be present.

    A link to the project will be added at the end of the email body.

    If content_text is not provided it will be generated from
    content_html for email clients without html support,
    the result may be less pretty than handcrafted text.
    """
    send(
        create_email(
            to=to,
            subject=subject,
            content_text=content_text,
            content_html=content_html,
            attachments=attachments,
        )
    )
