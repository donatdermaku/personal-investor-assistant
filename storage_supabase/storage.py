from __future__ import annotations

import os
from typing import Optional

import requests


def _storage_base() -> str:
    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    if not url:
        raise RuntimeError("SUPABASE_URL is not set.")
    return f"{url}/storage/v1/object"


def _auth_headers(content_type: Optional[str] = None) -> dict[str, str]:
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    if not key:
        raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY is not set.")
    headers = {
        "Authorization": f"Bearer {key}",
    }
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def upload_bytes(bucket: str, path: str, data: bytes, content_type: str) -> None:
    url = f"{_storage_base()}/{bucket}/{path}"
    headers = _auth_headers(content_type)
    headers["x-upsert"] = "true"
    resp = requests.put(url, headers=headers, data=data, timeout=30)
    resp.raise_for_status()


def download_bytes(bucket: str, path: str) -> bytes:
    url = f"{_storage_base()}/{bucket}/{path}"
    headers = _auth_headers()
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.content
