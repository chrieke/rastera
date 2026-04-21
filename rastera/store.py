"""Obstore construction, S3 authentication and region discovery.

When opening a file from S3, the store is configured as follows:

**Region** is resolved in priority order:

1. Explicit ``region`` kwarg passed by the caller.
2. Region extracted from HTTPS-style URLs
   (e.g. ``https://bucket.s3.eu-north-1.amazonaws.com/...``).
3. For authenticated access (``skip_signature=False``): the region from
   the active boto3 session (``~/.aws/config``, ``AWS_REGION`` env var, etc.).
4. Fallback ``us-west-2`` (unsigned/public access only).

**Credentials** default to unsigned (public) access.  Pass
``skip_signature=False`` in ``store_kwargs`` to enable authenticated access
via ``Boto3CredentialProvider``, which supports env vars,
``~/.aws/credentials``, SSO, IAM roles, and all other boto3-supported
credential sources.  If ``obstore.auth.boto3`` is not installed, it falls
back to unsigned access silently.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from async_tiff.store import from_url  # type: ignore[reportMissingImports]

_DEFAULT_REGION = "us-west-2"
_S3_REGION_RE = re.compile(r"[./]s3[.-]([a-z0-9-]+)\.amazonaws\.com")


def _is_s3_uri(uri: str) -> bool:
    return uri.startswith("s3://") or ".s3." in uri or ".s3-" in uri


def _detect_region(uri: str) -> str | None:
    """Try to extract the AWS region from an S3 HTTPS URL.

    Returns None when the region cannot be determined from the URL,
    letting obstore fall back to its own discovery (``AWS_REGION`` env
    var, boto3 session region, or ``us-east-1`` default).

    Handles virtual-hosted style:
        https://<bucket>.s3.<region>.amazonaws.com/...
        https://<bucket>.s3-<region>.amazonaws.com/...
    And path style:
        https://s3.<region>.amazonaws.com/...
    """
    m = _S3_REGION_RE.search(uri)
    if m:
        return m.group(1)
    return None


def _extract_key(uri: str) -> str:
    """Extract the object key from a URI, for use with a pre-constructed store."""
    parsed = urlparse(uri)
    if parsed.scheme in {"s3", "gs", "az"}:
        return parsed.path.lstrip("/")
    if parsed.scheme in {"http", "https"}:
        host = parsed.netloc
        if ".s3." in host or ".s3-" in host:
            return parsed.path.lstrip("/")
        # Path-style: https://s3.<region>.amazonaws.com/<bucket>/<key>
        parts = parsed.path.lstrip("/").split("/", 1)
        return parts[1] if len(parts) == 2 else ""
    local_path = _resolve_local_path(uri)
    if local_path is not None:
        return local_path.name
    return parsed.path or uri


def _apply_s3_defaults(store_kwargs: dict[str, Any], uri: str) -> None:
    """Set default S3 credentials/region on *store_kwargs* when *uri* is an S3 URI.

    Region is extracted from HTTPS-style URLs when possible.  The default
    for credentials is ``skip_signature=True`` (unsigned / public access).
    Callers that need authenticated access should pass
    ``skip_signature=False`` — this will automatically set up a
    ``Boto3CredentialProvider`` so that AWS SSO, profiles, and all other
    boto3-supported credential sources work transparently.
    """
    if not _is_s3_uri(uri):
        return
    region = _detect_region(uri)
    if region is not None:
        store_kwargs.setdefault("region", region)
    if store_kwargs.get("skip_signature") is False:
        store_kwargs.pop("skip_signature")
        _apply_boto3_credentials(store_kwargs, url_region=region)
    elif "credential_provider" not in store_kwargs:
        store_kwargs.setdefault("skip_signature", True)
        if region is None:
            store_kwargs.setdefault("region", _DEFAULT_REGION)


def _apply_boto3_credentials(
    store_kwargs: dict[str, Any],
    url_region: str | None,
) -> None:
    """Configure ``Boto3CredentialProvider`` on *store_kwargs*.

    Falls back to ``skip_signature=True`` if boto3/obstore auth is
    not available.
    """
    try:
        from obstore.auth.boto3 import Boto3CredentialProvider

        provider = Boto3CredentialProvider()
        store_kwargs["credential_provider"] = provider
        # Merge the boto3 session config (e.g. region) so obstore
        # targets the right endpoint for s3:// URIs that don't
        # encode a region.  A region already detected from the URL
        # takes precedence.
        if provider.config:
            merged = {**provider.config}
            if url_region is not None:
                merged.pop("region", None)
            if merged:
                existing = store_kwargs.get("config", {}) or {}
                store_kwargs["config"] = {**merged, **existing}
    except ImportError:
        store_kwargs.setdefault("skip_signature", True)


def _build_store_with(uri: str, from_url_fn: Any, **store_kwargs: Any) -> Any:
    """Build an object store rooted at the bucket/host level.

    Accepts any ``from_url`` callable (e.g. ``async_tiff.store.from_url``
    or ``obstore.store.from_url``) so the same logic serves both backends.
    """
    local_path = _resolve_local_path(uri)
    if local_path is not None:
        return from_url_fn(local_path.parent.as_uri(), **store_kwargs)
    _apply_s3_defaults(store_kwargs, uri)
    return from_url_fn(_bucket_url(uri), **store_kwargs)


def _build_store(uri: str, **store_kwargs: Any) -> Any:
    """Build an async-tiff object store rooted at the bucket/host level."""
    return _build_store_with(uri, from_url, **store_kwargs)


def _bucket_url(uri: str) -> str:
    """Extract the bucket-level URL from a full object URI."""
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        # s3://bucket/key -> s3://bucket
        return f"s3://{parsed.netloc}"
    if parsed.scheme in {"gs", "az"}:
        return f"{parsed.scheme}://{parsed.netloc}"
    if parsed.scheme in {"http", "https"}:
        # https://bucket.s3.region.amazonaws.com/key -> https://bucket.s3.region.amazonaws.com
        return f"{parsed.scheme}://{parsed.netloc}"
    local_path = _resolve_local_path(uri)
    if local_path is not None:
        return local_path.parent.as_uri()
    return uri


def _resolve_local_path(uri: str) -> Path | None:
    """Return resolved Path if uri is local, else None."""
    parsed = urlparse(uri)
    if parsed.scheme not in ("", "file") or _is_s3_uri(uri):
        return None
    return Path(parsed.path if parsed.scheme == "file" else uri).resolve()


def _obstore_key(uri: str) -> str:
    """Extract the object key for use with an obstore rooted at bucket level.

    Unlike ``_extract_key`` (used with async-tiff stores), this does not
    distinguish virtual-hosted from path-style S3 HTTP URLs because
    obstore handles that internally when the store is rooted via
    ``_bucket_url``.
    """
    local_path = _resolve_local_path(uri)
    if local_path is not None:
        return local_path.name
    parsed = urlparse(uri)
    if parsed.scheme in ("s3", "gs", "az"):
        return parsed.path.lstrip("/")
    if parsed.scheme in ("http", "https"):
        host = parsed.netloc
        path = parsed.path.lstrip("/")
        if ".s3." not in host and ".s3-" not in host:
            parts = path.split("/", 1)
            return parts[1] if len(parts) == 2 else ""
        return path
    return parsed.path or uri
