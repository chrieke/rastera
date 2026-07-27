"""Obstore construction, S3 authentication and region discovery.

Every URI is normalised once by :func:`_parse_uri` into a ``(root, key)`` pair
such that ``from_url(root)`` plus ``key`` addresses the same object that
``from_url(uri)`` would. All other helpers derive from that parse.

Supported S3 forms, all rewritten to ``s3://<bucket>/<key>`` before they reach
obstore — whose own URL parser rejects the dash-region, legacy-global and
dotted-bucket variants::

    s3://<bucket>/<key>
    https://<bucket>.s3.<region>.amazonaws.com/<key>
    https://<bucket>.s3-<region>.amazonaws.com/<key>
    https://<bucket>.s3.amazonaws.com/<key>
    https://s3.<region>.amazonaws.com/<bucket>/<key>
    https://s3.amazonaws.com/<bucket>/<key>

Other ``amazonaws.com`` hosts — dual-stack, transfer acceleration, FIPS, access
points, S3 Express, VPC endpoints — are rejected. Each implies an endpoint and
addressing style that cannot be inferred from the URL, and silently serving them
from the standard endpoint would defeat the reason for using them.

Non-AWS S3-compatible services (Wasabi, MinIO, Ceph) are read over plain HTTP,
which works for public objects only; signed access needs an explicit
``store=S3Store(bucket=..., endpoint=..., region=...)``.

**Region** for AWS URIs, in priority order: encoded in the host, explicit
``region`` kwarg, ``AWS_REGION``/``AWS_DEFAULT_REGION``, the boto3 session (only
consulted when authenticating), then ``us-west-2``. A host-encoded region that
contradicts an explicit kwarg is an error rather than a silent pick.

**Credentials** default to unsigned. ``skip_signature=False`` switches to
``Boto3CredentialProvider`` (env vars, ``~/.aws/credentials``, SSO, IAM roles);
if it cannot be constructed, access falls back to unsigned.
"""

from __future__ import annotations

import os
import posixpath
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import ParseResult, unquote, urlparse, urlunparse

import obstore
from async_tiff.store import from_url  # type: ignore[reportMissingImports]
from obstore.store import from_url as obstore_from_url

_DEFAULT_REGION = "us-west-2"

_AWS_SUFFIX = r"amazonaws\.com(?:\.cn)?"
_AWS_REGION = r"[a-z]{2}(?:-[a-z]+)+-\d+"
# Anchored on the host alone: a substring search also matches a region spelled
# out in an attacker-controlled path segment.
_AWS_HOST_RE = re.compile(
    rf"^(?:(?P<bucket>[^/]+)\.)?s3(?:[.-](?P<region>{_AWS_REGION}))?\.{_AWS_SUFFIX}$",
    re.IGNORECASE,
)
_AWS_SUFFIX_RE = re.compile(rf"\.{_AWS_SUFFIX}$", re.IGNORECASE)

# Rejected outright by LocalFileSystem and HTTPStore ("Cannot pass config or
# keyword parameters for scheme ..."), so they cannot be forwarded blindly.
_S3_ONLY_KWARGS = ("skip_signature", "region", "credential_provider")

UriKind = Literal["local", "aws", "cloud", "http"]


@dataclass(frozen=True)
class ParsedURI:
    """A URI split into the store root and the key relative to it."""

    uri: str
    kind: UriKind
    root: str
    key: str
    local_path: Path | None = None
    region: str | None = None
    virtual_hosted: bool = False

    @property
    def identity(self) -> tuple[str, str | None]:
        """What must match for two URIs to share one store.

        Not addressing style — both spellings reach the same objects.
        """
        return (self.root, self.region)


def _parse_uri(uri: str) -> ParsedURI:
    parsed = urlparse(uri)
    scheme = parsed.scheme.lower()

    if scheme in ("", "file"):
        raw = unquote(parsed.path) if scheme == "file" else uri
        path = Path(raw).resolve()
        return ParsedURI(uri, "local", path.parent.as_uri(), path.name, local_path=path)

    if scheme == "s3":
        return ParsedURI(uri, "aws", f"s3://{parsed.netloc}", _path_key(parsed))

    if scheme in ("gs", "az"):
        return ParsedURI(uri, "cloud", f"{scheme}://{parsed.netloc}", _path_key(parsed))

    if scheme in ("http", "https"):
        return _parse_http_uri(uri, parsed)

    raise ValueError(
        f"Unsupported URI scheme {scheme!r} in {uri!r}. Expected one of "
        f"s3, gs, az, http, https, file, or a local path."
    )


def _build_store_with(uri: str, from_url_fn: Any, **store_kwargs: Any) -> Any:
    """Build an object store rooted at the bucket/host level.

    Accepts any ``from_url`` callable (e.g. ``async_tiff.store.from_url``
    or ``obstore.store.from_url``) so the same logic serves both backends.
    """
    parsed = _parse_uri(uri)
    return from_url_fn(parsed.root, **_store_kwargs_for(parsed, store_kwargs))


def _build_store(uri: str, **store_kwargs: Any) -> Any:
    """Build an async-tiff object store rooted at the bucket/host level."""
    return _build_store_with(uri, from_url, **store_kwargs)


def _extract_key(uri: str) -> str:
    """The object key relative to ``_parse_uri(uri).root``, percent-decoded."""
    return _parse_uri(uri).key


def _resolve_local_path(uri: str) -> Path | None:
    return _parse_uri(uri).local_path


def _require_same_bucket(uris: Sequence[str], reason: str) -> None:
    """Raise if *uris* do not all resolve to the same store.

    A store is rooted at one bucket and object keys carry no bucket, so a
    mixed-bucket list either 404s or — when two buckets mirror a key path —
    silently serves one file's bytes for another's URI. A region only some URIs
    state also counts: the store comes from the first, so a later one is dropped.
    """
    first = _parse_uri(uris[0])
    mismatched = [u for u in uris[1:] if _parse_uri(u).identity != first.identity]
    if mismatched:
        raise ValueError(
            f"All URIs must belong to the same bucket/host when {reason}. "
            f"First URI resolves to bucket {first.root!r} in region "
            f"{first.region!r}, but these do not: {mismatched}"
        )


async def _fetch_descriptor_bytes(uri: str, **store_kwargs: Any) -> bytes:
    """Fetch a full XML descriptor object (VRT or DIMAP) via obstore,
    with a filesystem fast-path for local paths. Shared by the VRT and
    DIMAP readers — both just GET the whole document once at open time."""
    parsed = _parse_uri(uri)
    if parsed.local_path is not None:
        return parsed.local_path.read_bytes()
    store = _build_store_with(uri, obstore_from_url, **store_kwargs)
    result = await obstore.get_async(store, parsed.key)
    return bytes(await result.bytes_async())


def _join_relative_uri(base_uri: str, relative: str) -> str:
    """Resolve *relative* against *base_uri*'s parent directory. Local
    paths are joined via pathlib; remote URIs via posix path normalization.

    Callers that need to recognize absolute paths or URIs should do that
    check themselves before invoking — this helper always treats its
    input as relative.
    """
    local = _resolve_local_path(base_uri)
    if local is not None:
        return str((local.parent / relative).resolve())
    parsed = urlparse(base_uri)
    parent = posixpath.dirname(parsed.path)
    joined = posixpath.normpath(posixpath.join(parent, relative))
    return urlunparse(parsed._replace(path=joined))


# ── URI parsing ──────────────────────────────────────────────────────────


def _path_key(parsed: ParseResult) -> str:
    return unquote(parsed.path).lstrip("/")


def _parse_http_uri(uri: str, parsed: ParseResult) -> ParsedURI:
    host = parsed.netloc
    match = _AWS_HOST_RE.match(host)

    if match is None:
        if _AWS_SUFFIX_RE.search(host):
            raise ValueError(
                f"Unsupported S3 endpoint {host!r}. rastera resolves "
                f"virtual-hosted and path-style URLs on s3[.-]<region>.amazonaws.com; "
                f"dual-stack, transfer-acceleration, FIPS, access-point, S3 Express "
                f"and VPC-endpoint hosts imply an endpoint and addressing style that "
                f"cannot be inferred from the URL. Pass an explicit store instead: "
                f"store=S3Store(bucket=..., endpoint=..., region=...)."
            )
        # A query or fragment carries auth material that a host-rooted store
        # would silently drop, so root at the whole URI in that case.
        if parsed.query or parsed.fragment:
            return ParsedURI(uri, "http", uri, "")
        return ParsedURI(uri, "http", f"{parsed.scheme}://{host}", _path_key(parsed))

    bucket = match.group("bucket")
    raw_key = parsed.path.lstrip("/")
    if bucket is None:
        # Path-style: the bucket is the first path segment.
        bucket, _, raw_key = raw_key.partition("/")
        bucket = unquote(bucket)
        virtual_hosted = False
    else:
        # A dotted bucket cannot be reached virtual-hosted: the wildcard cert
        # for *.s3.<region>.amazonaws.com covers a single label only.
        virtual_hosted = "." not in bucket
    if not bucket:
        raise ValueError(f"No bucket in S3 URL {uri!r}.")

    region = match.group("region")
    return ParsedURI(
        uri,
        "aws",
        f"s3://{bucket}",
        unquote(raw_key),
        region=region.lower() if region else None,
        virtual_hosted=virtual_hosted,
    )


# ── store kwargs ─────────────────────────────────────────────────────────


def _store_kwargs_for(
    parsed: ParsedURI, store_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """The kwargs to pass ``from_url`` for *parsed*, given the caller's."""
    out = dict(store_kwargs)

    if parsed.kind == "aws":
        return _aws_store_kwargs(parsed, out)

    if parsed.kind == "local":
        out.pop("skip_signature", None)
        return out

    if parsed.kind == "http":
        if out.get("skip_signature") is False or "credential_provider" in out:
            raise ValueError(
                f"{parsed.uri!r} resolves to a plain HTTP store, which cannot sign "
                f"requests. rastera authenticates AWS S3 hosts only; for another "
                f"S3-compatible service pass a configured store, e.g. "
                f"store=S3Store(bucket=..., endpoint=..., region=...)."
            )
        for key in _S3_ONLY_KWARGS:
            out.pop(key, None)
        return out

    return out  # gs://, az:// — let obstore validate its own kwargs


def _aws_store_kwargs(parsed: ParsedURI, out: dict[str, Any]) -> dict[str, Any]:
    if parsed.virtual_hosted:
        out.setdefault("virtual_hosted_style_request", True)

    # obstore rejects a region supplied twice ("Duplicate key aws_region"), so
    # every source is collapsed into a single `region=` here.
    config = dict(out.get("config") or {})
    config_region = None
    for key in [k for k in config if k.lower() in ("region", "aws_region")]:
        config_region = config.pop(key)
    if config:
        out["config"] = config
    else:
        out.pop("config", None)

    caller_region = out.pop("region", None) or config_region
    if parsed.region and caller_region and caller_region != parsed.region:
        raise ValueError(
            f"region={caller_region!r} conflicts with the region encoded in "
            f"{parsed.uri!r} ({parsed.region!r}). Pass only one of the two."
        )
    region = parsed.region or caller_region or _env_region()

    if out.get("skip_signature") is False:
        del out["skip_signature"]
        provider = _boto3_provider()
        if provider is None:
            out["skip_signature"] = True
        else:
            out["credential_provider"] = provider
            session_config: dict[str, Any] = provider.config or {}
            region = region or session_config.get("region")
    elif "credential_provider" not in out:
        out.setdefault("skip_signature", True)

    out["region"] = region or _DEFAULT_REGION
    return out


def _env_region() -> str | None:
    return os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")


def _boto3_provider() -> Any | None:
    """``Boto3CredentialProvider``, or None when it cannot be constructed.

    Absent credentials surface as ``ValueError`` and a bad ``AWS_PROFILE`` as
    botocore's ``ProfileNotFound``; neither is an ``ImportError``, so the
    documented fallback to unsigned access needs the wide catch.
    """
    try:
        from obstore.auth.boto3 import Boto3CredentialProvider

        return Boto3CredentialProvider()
    except Exception:
        return None
