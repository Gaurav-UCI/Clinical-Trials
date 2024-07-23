from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import databutton as db


def normalize_path(path: str) -> Path:
    """Normalize and return the absolute path."""
    return Path(path).resolve()


def zip_folder_to_bytes(folder_path: str) -> bytes:
    """Zip the contents of a folder and return as bytes."""
    normalized_folder_path = normalize_path(folder_path)
    zip_bytes = BytesIO()  # Use BytesIO as a buffer
    with ZipFile(zip_bytes, "w", ZIP_DEFLATED) as zipf:
        for file_path in normalized_folder_path.rglob("*"):
            if file_path.is_file():
                archive_name = file_path.relative_to(normalized_folder_path).as_posix()
                zipf.write(file_path, archive_name)
    zip_bytes.seek(0)  # Go to the start of the BytesIO buffer
    return zip_bytes.getvalue()


def unzip_bytes_to_folder(bytes_data: bytes, extract_path: str):
    """Unzip bytes into a folder."""
    normalized_extract_path = normalize_path(extract_path)
    normalized_extract_path.mkdir(parents=True, exist_ok=True)
    zip_bytes = BytesIO(bytes_data)
    with ZipFile(zip_bytes, "r") as zip_ref:
        zip_ref.extractall(normalized_extract_path)


def upload(key: str, folder_path: str):
    """Zip and upload a folder to blob storage."""
    normalized_folder_path = normalize_path(folder_path)
    zipped_bytes = zip_folder_to_bytes(normalized_folder_path.as_posix())
    db.storage.binary.put(key, zipped_bytes)


def download(key: str, folder_path: str):
    """Download and unzip a folder from blob storage."""
    normalized_folder_path = normalize_path(folder_path)
    zipped_bytes = db.storage.binary.get(key, default=None)
    if zipped_bytes is not None:
        unzip_bytes_to_folder(zipped_bytes, normalized_folder_path.as_posix())
