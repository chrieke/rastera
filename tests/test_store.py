"""Unit tests for store helpers."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from rastera.store import (
    _apply_s3_defaults,
    _bucket_url,
    _build_store_with,
    _detect_region,
    _extract_key,
    _is_s3_uri,
    _resolve_local_path,
)

# ── _is_s3_uri ──────────────────────────────────────────────────────────


class TestIsS3Uri:
    def test_s3_scheme(self):
        assert _is_s3_uri("s3://bucket/key") is True

    def test_virtual_hosted_https(self):
        assert _is_s3_uri("https://bucket.s3.us-east-1.amazonaws.com/key") is True

    def test_virtual_hosted_dash(self):
        assert _is_s3_uri("https://bucket.s3-us-west-2.amazonaws.com/key") is True

    def test_non_s3_https(self):
        assert _is_s3_uri("https://example.com/file.tif") is False

    def test_local_path(self):
        assert _is_s3_uri("/tmp/file.tif") is False

    def test_gs_uri(self):
        assert _is_s3_uri("gs://bucket/key") is False


# ── _detect_region ───────────────────────────────────────────────────────


class TestDetectRegion:
    def test_virtual_hosted_dot(self):
        uri = "https://bucket.s3.eu-north-1.amazonaws.com/key"
        assert _detect_region(uri) == "eu-north-1"

    def test_virtual_hosted_dash(self):
        uri = "https://bucket.s3-us-west-2.amazonaws.com/key"
        assert _detect_region(uri) == "us-west-2"

    def test_path_style(self):
        uri = "https://s3.ap-southeast-1.amazonaws.com/bucket/key"
        assert _detect_region(uri) == "ap-southeast-1"

    def test_no_region_in_url(self):
        assert _detect_region("s3://bucket/key") is None

    def test_non_aws_url(self):
        assert _detect_region("https://example.com/file.tif") is None


# ── _bucket_url ──────────────────────────────────────────────────────────


class TestBucketUrl:
    def test_s3_scheme(self):
        assert _bucket_url("s3://my-bucket/path/to/file") == "s3://my-bucket"

    def test_gs_scheme(self):
        assert _bucket_url("gs://my-bucket/key") == "gs://my-bucket"

    def test_az_scheme(self):
        assert _bucket_url("az://container/blob") == "az://container"

    def test_https_virtual_hosted(self):
        uri = "https://bucket.s3.us-east-1.amazonaws.com/key/file.tif"
        assert _bucket_url(uri) == "https://bucket.s3.us-east-1.amazonaws.com"

    def test_local_path_returns_parent_uri(self, tmp_path):
        f = tmp_path / "a.tif"
        f.write_bytes(b"")
        assert _bucket_url(str(f)) == tmp_path.resolve().as_uri()

    def test_local_siblings_share_bucket(self, tmp_path):
        a = tmp_path / "a.tif"
        b = tmp_path / "b.tif"
        a.write_bytes(b"")
        b.write_bytes(b"")
        assert _bucket_url(str(a)) == _bucket_url(str(b))

    def test_file_uri_returns_parent_uri(self, tmp_path):
        f = tmp_path / "a.tif"
        f.write_bytes(b"")
        assert _bucket_url(f.as_uri()) == tmp_path.resolve().as_uri()


# ── _resolve_local_path ─────────────────────────────────────────────────


class TestResolveLocalPath:
    def test_absolute_path(self):
        result = _resolve_local_path("/tmp/file.tif")
        assert result is not None
        assert str(result).endswith("file.tif")

    def test_s3_uri_returns_none(self):
        assert _resolve_local_path("s3://bucket/key") is None

    def test_https_returns_none(self):
        assert _resolve_local_path("https://example.com/f.tif") is None

    def test_virtual_hosted_s3_returns_none(self):
        assert _resolve_local_path("https://b.s3.us-east-1.amazonaws.com/k") is None


# ── _extract_key ─────────────────────────────────────────────────────────


class TestExtractKey:
    def test_s3_scheme(self):
        assert _extract_key("s3://bucket/path/to/file.tif") == "path/to/file.tif"

    def test_virtual_hosted_https(self):
        uri = "https://bucket.s3.us-east-1.amazonaws.com/path/file.tif"
        assert _extract_key(uri) == "path/file.tif"

    def test_path_style_https(self):
        uri = "https://s3.us-east-1.amazonaws.com/bucket/path/file.tif"
        assert _extract_key(uri) == "path/file.tif"

    def test_local_path(self):
        assert _extract_key("/tmp/data/file.tif") == "/tmp/data/file.tif"


# ── _apply_s3_defaults ──────────────────────────────────────────────────


class TestApplyS3Defaults:
    def test_non_s3_uri_is_noop(self):
        kwargs: dict[str, Any] = {}
        _apply_s3_defaults(kwargs, "https://example.com/file.tif")
        assert kwargs == {}

    def test_s3_uri_sets_skip_signature(self):
        kwargs: dict[str, Any] = {}
        _apply_s3_defaults(kwargs, "s3://bucket/key")
        assert kwargs["skip_signature"] is True

    def test_s3_uri_sets_fallback_region(self):
        kwargs: dict[str, Any] = {}
        _apply_s3_defaults(kwargs, "s3://bucket/key")
        assert kwargs["region"] == "us-west-2"

    def test_region_from_url_takes_precedence(self):
        kwargs: dict[str, Any] = {}
        _apply_s3_defaults(kwargs, "https://b.s3.eu-north-1.amazonaws.com/k")
        assert kwargs["region"] == "eu-north-1"

    def test_explicit_region_not_overwritten(self):
        kwargs: dict[str, Any] = {"region": "ap-south-1"}
        _apply_s3_defaults(kwargs, "https://b.s3.eu-north-1.amazonaws.com/k")
        assert kwargs["region"] == "ap-south-1"

    def test_custom_credential_provider_skips_defaults(self):
        provider = MagicMock()
        kwargs: dict[str, Any] = {"credential_provider": provider}
        _apply_s3_defaults(kwargs, "s3://bucket/key")
        assert "skip_signature" not in kwargs
        assert kwargs["credential_provider"] is provider

    def test_skip_signature_false_triggers_boto3(self):
        mock_provider = MagicMock()
        mock_provider.config = None
        with patch(
            "rastera.store.Boto3CredentialProvider",
            return_value=mock_provider,
            create=True,
        ):
            with patch("rastera.store._apply_boto3_credentials") as mock_apply:
                kwargs: dict[str, Any] = {"skip_signature": False}
                _apply_s3_defaults(kwargs, "s3://bucket/key")
                mock_apply.assert_called_once()
                assert "skip_signature" not in kwargs


# ── _build_store_with ────────────────────────────────────────────────────


class TestBuildStoreWith:
    def test_delegates_to_from_url_fn(self):
        mock_from_url = MagicMock(return_value="store")
        result = _build_store_with("s3://bucket/key", mock_from_url)
        assert result == "store"
        mock_from_url.assert_called_once()
        # Should be called with the bucket URL, not the full object path
        call_args = mock_from_url.call_args
        assert call_args[0][0] == "s3://bucket"
