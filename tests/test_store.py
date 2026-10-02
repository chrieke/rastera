"""Unit tests for store helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from async_tiff.store import HTTPStore  # type: ignore[reportMissingImports]
from obstore.store import HTTPStore as ObstoreHTTPStore
from obstore.store import from_url as obstore_from_url

from rastera.store import (
    _build_store,
    _build_store_with,
    _check_source_uri,
    _extract_key,
    _join_relative_uri,
    _parse_uri,
    _require_same_bucket,
    _resolve_local_path,
    _store_kwargs_for,
)

# ── _parse_uri ───────────────────────────────────────────────────────────

# uri -> (kind, root, key, region, virtual_hosted)
AWS_URIS = [
    ("s3://bucket/path/file.tif", "s3://bucket", "path/file.tif", None, False),
    ("s3://my.dotted.bucket/k.tif", "s3://my.dotted.bucket", "k.tif", None, False),
    (
        "https://bucket.s3.us-east-1.amazonaws.com/path/file.tif",
        "s3://bucket",
        "path/file.tif",
        "us-east-1",
        True,
    ),
    (
        "https://bucket.s3-us-west-2.amazonaws.com/path/file.tif",
        "s3://bucket",
        "path/file.tif",
        "us-west-2",
        True,
    ),
    (
        "https://s3.ap-southeast-1.amazonaws.com/bucket/path/file.tif",
        "s3://bucket",
        "path/file.tif",
        "ap-southeast-1",
        False,
    ),
    (
        "https://s3-eu-west-1.amazonaws.com/bucket/path/file.tif",
        "s3://bucket",
        "path/file.tif",
        "eu-west-1",
        False,
    ),
    # Legacy global endpoint carries no region; obstore does not follow S3
    # region redirects, so it must fall through to the ladder rather than
    # being pinned to us-east-1.
    ("https://bucket.s3.amazonaws.com/k.tif", "s3://bucket", "k.tif", None, True),
    ("https://s3.amazonaws.com/bucket/k.tif", "s3://bucket", "k.tif", None, False),
    # A dotted bucket cannot go virtual-hosted: single-label wildcard cert.
    (
        "https://my.dotted.bucket.s3.us-east-1.amazonaws.com/k.tif",
        "s3://my.dotted.bucket",
        "k.tif",
        "us-east-1",
        False,
    ),
    (
        "https://BUCKET.S3.US-EAST-1.AMAZONAWS.COM/k.tif",
        "s3://BUCKET",
        "k.tif",
        "us-east-1",
        True,
    ),
    (
        "https://bucket.s3.cn-north-1.amazonaws.com.cn/k.tif",
        "s3://bucket",
        "k.tif",
        "cn-north-1",
        True,
    ),
    (
        "https://bucket.s3.us-east-1.amazonaws.com/my%20key/a%2Bb.tif",
        "s3://bucket",
        "my key/a+b.tif",
        "us-east-1",
        True,
    ),
    (
        "https://s3.us-east-1.amazonaws.com/bucket",
        "s3://bucket",
        "",
        "us-east-1",
        False,
    ),
]

# Each implies an endpoint and addressing style the URL cannot convey.
UNSUPPORTED_AWS_HOSTS = [
    "https://bucket.s3.dualstack.us-east-1.amazonaws.com/k.tif",
    "https://bucket.s3-accelerate.amazonaws.com/k.tif",
    "https://s3-fips.us-east-1.amazonaws.com/bucket/k.tif",
    "https://bucket.s3-website-us-east-1.amazonaws.com/k.tif",
    "https://ap-123456789012.s3-accesspoint.us-east-1.amazonaws.com/k.tif",
    "https://bucket.s3express-use1-az4.us-east-1.amazonaws.com/k.tif",
    "https://bucket.s3.us-east-1.vpce.amazonaws.com/k.tif",
]


class TestParseUriAws:
    @pytest.mark.parametrize("uri,root,key,region,vhost", AWS_URIS)
    def test_normalizes_to_s3_scheme(
        self, uri: str, root: str, key: str, region: str | None, vhost: bool
    ):
        parsed = _parse_uri(uri)
        assert parsed.kind == "aws"
        assert (parsed.root, parsed.key) == (root, key)
        assert parsed.region == region
        assert parsed.virtual_hosted is vhost

    @pytest.mark.parametrize("uri", UNSUPPORTED_AWS_HOSTS)
    def test_unsupported_endpoint_raises(self, uri: str):
        with pytest.raises(ValueError, match="Unsupported S3 endpoint"):
            _parse_uri(uri)

    @pytest.mark.parametrize(
        "uri",
        [
            "https://bucket.s3.eu-north-1.amazonaws.com/k/a.tif"
            "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=abc",
            "https://s3.eu-north-1.amazonaws.com/bucket/k/a.tif?versionId=v1",
            "https://bucket.s3.amazonaws.com/k/a.tif?versionId=v1",
            # Presigned on an endpoint the rewrite rejects: served as given.
            "https://bucket.s3.dualstack.us-east-1.amazonaws.com/k/a.tif?X-Amz-Signature=abc",
        ],
    )
    def test_a_query_string_is_read_as_given(self, uri: str):
        # Rewritten to s3://, a presigned URL lost its signature and 403'd, and
        # ?versionId= silently served the latest version instead.
        parsed = _parse_uri(uri)
        assert (parsed.kind, parsed.root, parsed.key) == ("http", uri, "")

    def test_a_fragment_on_an_s3_host_is_dropped_as_before(self):
        # Never sent to the server, so nothing is lost rewriting to s3://.
        parsed = _parse_uri("https://bucket.s3.eu-north-1.amazonaws.com/k/a.tif#f")
        assert (parsed.kind, parsed.root, parsed.key) == (
            "aws",
            "s3://bucket",
            "k/a.tif",
        )

    def test_region_is_not_read_from_the_path(self):
        # A substring search over the whole URI reads a region out of an
        # attacker-controlled path segment.
        parsed = _parse_uri("https://cdn.evil.com/a/.s3.us-east-1.amazonaws.com/x.tif")
        assert parsed.kind == "http"
        assert parsed.region is None


class TestParseUriOther:
    def test_gs_and_az(self):
        for uri, root in [("gs://b/k/a.tif", "gs://b"), ("az://c/k/a.tif", "az://c")]:
            parsed = _parse_uri(uri)
            assert parsed.kind == "cloud"
            assert (parsed.root, parsed.key) == (root, "k/a.tif")

    def test_generic_https_keeps_the_full_path(self):
        parsed = _parse_uri("https://cdn.example.com/2024/scene.tif")
        assert (parsed.kind, parsed.root) == ("http", "https://cdn.example.com")
        assert parsed.key == "2024/scene.tif"

    def test_generic_https_single_segment(self):
        assert _extract_key("https://cdn.example.com/scene.tif") == "scene.tif"

    def test_query_string_roots_at_the_full_uri(self):
        # Host-rooting would drop the token and turn a signed URL into a 403.
        uri = "https://cdn.example.com/k.tif?token=xyz"
        parsed = _parse_uri(uri)
        assert (parsed.root, parsed.key) == (uri, "")

    def test_percent_decodes_generic_https(self):
        assert _extract_key("https://cdn.example.com/my%20key/a.tif") == "my key/a.tif"

    def test_non_aws_s3_compatible_host_is_plain_http(self):
        parsed = _parse_uri("https://b.s3.wasabisys.com/k/a.tif")
        assert (parsed.kind, parsed.root) == ("http", "https://b.s3.wasabisys.com")
        assert parsed.key == "k/a.tif"

    def test_unknown_scheme_raises(self):
        with pytest.raises(ValueError, match="Unsupported URI scheme"):
            _parse_uri("ftp://host/file.tif")

    @pytest.mark.parametrize(
        "uri", ["s3://b/k.tif?versionId=v1", "gs://b/k.tif?generation=1", "az://c/k#x"]
    )
    def test_a_query_or_fragment_on_a_bucket_uri_raises(self, uri: str):
        """The key is the path alone, so ?versionId= read the current version."""
        with pytest.raises(ValueError, match="query or fragment"):
            _parse_uri(uri)


class TestParseUriLocal:
    def test_absolute_path(self, tmp_path: Path):
        f = tmp_path / "file.tif"
        f.write_bytes(b"")
        parsed = _parse_uri(str(f))
        assert parsed.kind == "local"
        assert parsed.local_path == f.resolve()
        anchor = Path(f.resolve().anchor)
        assert (parsed.root, parsed.key) == (
            anchor.as_uri(),
            f.resolve().relative_to(anchor).as_posix(),
        )

    def test_file_uri_is_percent_decoded(self, tmp_path: Path):
        d = tmp_path / "My Scenes"
        d.mkdir()
        f = d / "x.tif"
        f.write_bytes(b"")
        parsed = _parse_uri(f.as_uri())
        assert parsed.local_path == f.resolve()

    def test_dot_s3_in_filename_is_still_local(self, tmp_path: Path):
        # A substring match on ".s3." classifies this as remote.
        f = tmp_path / "scene.s3.tif"
        f.write_bytes(b"")
        assert _resolve_local_path(str(f)) == f.resolve()

    def test_remote_uris_are_not_local(self):
        for uri in ["s3://bucket/key", "https://example.com/f.tif"]:
            assert _resolve_local_path(uri) is None


# ── descriptor source resolution ─────────────────────────────────────────

VRT = "s3://bucket/delivery/scene.vrt"


class TestCheckSourceURI:
    @pytest.mark.parametrize(
        "source",
        [
            "s3://bucket/delivery/tile.tif",  # sibling of the descriptor
            "s3://bucket/delivery/sub/tile.tif",  # below it
            "s3://bucket/other-delivery/tile.tif",  # elsewhere in-bucket
        ],
    )
    def test_allows_the_descriptors_own_bucket(self, source: str):
        _check_source_uri(source, VRT)

    @pytest.mark.parametrize(
        ("source", "what"),
        [
            ("s3://other-bucket/tile.tif", "another bucket"),
            ("/mnt/data/tile.tif", "an absolute local path"),
            ("file:///mnt/data/tile.tif", "a file:// URL"),
            ("gs://other-bucket/tile.tif", "another cloud"),
            ("https://cdn.example.com/tile.tif", "an arbitrary host"),
            ("http://internal.example.local/tile.tif", "an internal host"),
        ],
    )
    def test_rejects_sources_outside_that_store(self, source: str, what: str):
        with pytest.raises(ValueError, match="outside its own store") as exc:
            _check_source_uri(source, VRT)
        # Both ends named, or whoever hits this on a legitimate descriptor
        # cannot tell which reference to go and fix.
        assert source in str(exc.value), what
        assert VRT in str(exc.value)

    def test_relative_segments_within_the_bucket_are_permitted(self):
        """``../`` segments leave the descriptor's prefix but not its bucket, and
        S3 keys are literal, so this is a key the delivery could have named
        outright."""
        joined = _join_relative_uri(VRT, "../../../../../../other.tif")
        assert joined == "s3://bucket/other.tif"
        _check_source_uri(joined, VRT)

    def test_http_descriptor_is_confined_to_its_host(self):
        descriptor = "https://data.example.com/a/x.vrt"
        _check_source_uri("https://data.example.com/b/tile.tif", descriptor)
        with pytest.raises(ValueError, match="outside its own store"):
            _check_source_uri("https://other.example.net/tile.tif", descriptor)

    @pytest.mark.parametrize(
        "source",
        ["", "s3://", "//other.example.com/f.tif", "HTTPS://OTHER.EXAMPLE.COM/f.tif"],
    )
    def test_odd_source_uris_are_rejected(self, source: str):
        with pytest.raises(ValueError):
            _check_source_uri(source, VRT)

    def test_remote_descriptor_cannot_reference_a_local_path(self):
        """Local descriptors are unconstrained, so the rule has to hold on the way
        in as well — otherwise one hop reaches a file whose own references are
        followed without it."""
        with pytest.raises(ValueError, match="outside its own store"):
            _check_source_uri("/tmp/other.vrt", VRT)


class TestCheckSourceURILocalDescriptor:
    """A local descriptor is unconstrained by design — see _check_source_uri."""

    def test_allows_remote_sources(self, tmp_path: Path):
        """The gdalbuildvrt case, which is why this is unconstrained."""
        _check_source_uri("s3://bucket/tile.tif", str(tmp_path / "scene.vrt"))

    def test_allows_a_path_outside_its_own_directory(self, tmp_path: Path):
        """Asserted so tightening this is a deliberate edit, not a silent break
        of local workflows."""
        vrt = tmp_path / "delivery" / "scene.vrt"
        vrt.parent.mkdir()
        _check_source_uri(str(tmp_path / "elsewhere.tif"), str(vrt))


# ── store root ───────────────────────────────────────────────────────────


class TestStoreRoot:
    def test_path_style_keeps_the_bucket(self):
        # Dropping it yields S3Store(bucket="") — constructs, reads nothing.
        uri = "https://s3.us-east-1.amazonaws.com/my-bucket/path/file.tif"
        assert _parse_uri(uri).root == "s3://my-bucket"

    def test_path_style_buckets_do_not_collide(self):
        a = "https://s3.us-east-1.amazonaws.com/bucket-a/x.tif"
        b = "https://s3.us-east-1.amazonaws.com/bucket-b/y.tif"
        assert _parse_uri(a).root != _parse_uri(b).root

    def test_local_files_share_a_root_across_folders(self, tmp_path: Path):
        """Rooted at each file's folder, files in two folders read as two
        buckets, and open(list), merge and build_index refused them."""
        (tmp_path / "A").mkdir()
        (tmp_path / "B").mkdir()
        a, b = tmp_path / "A" / "x.tif", tmp_path / "B" / "y.tif"
        a.write_bytes(b"")
        b.write_bytes(b"")
        root = Path(tmp_path.resolve().anchor).as_uri()
        assert _parse_uri(str(a)).root == _parse_uri(str(b)).root == root
        _require_same_bucket([str(a), str(b)], "testing")


# ── _require_same_bucket ─────────────────────────────────────────────────


class TestRequireSameBucket:
    def test_addressing_style_is_not_a_mismatch(self):
        """Both spellings of one bucket name the same objects."""
        _require_same_bucket(
            [
                "https://b.s3.eu-north-1.amazonaws.com/a.tif",
                "https://s3.eu-north-1.amazonaws.com/b/c.tif",
            ],
            "testing",
        )

    def test_different_buckets_still_raise(self):
        with pytest.raises(ValueError, match="same bucket/host"):
            _require_same_bucket(["s3://bucket-a/x.tif", "s3://bucket-b/x.tif"], "x")

    def test_region_only_some_uris_state_raises(self):
        """Matched on the region, not the bucket: they agree here, and a message
        naming only the bucket reads as a bucket mismatch."""
        with pytest.raises(ValueError, match="in region None"):
            _require_same_bucket(
                ["s3://b/a.tif", "https://b.s3.eu-north-1.amazonaws.com/c.tif"], "x"
            )


# ── _store_kwargs_for ────────────────────────────────────────────────────


def _kwargs(uri: str, **caller: Any) -> dict[str, Any]:
    return _store_kwargs_for(_parse_uri(uri), caller)


class TestStoreKwargs:
    def test_defaults_to_unsigned(self):
        assert _kwargs("s3://bucket/key")["skip_signature"] is True

    def test_fallback_region(self):
        # The autouse _clear_aws_region fixture already unset both vars.
        assert _kwargs("s3://bucket/key")["region"] == "us-west-2"

    def test_env_region_beats_the_fallback(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AWS_REGION", "eu-west-3")
        assert _kwargs("s3://bucket/key")["region"] == "eu-west-3"

    def test_host_region_beats_the_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AWS_REGION", "eu-west-3")
        uri = "https://b.s3.eu-north-1.amazonaws.com/k"
        assert _kwargs(uri)["region"] == "eu-north-1"

    def test_conflicting_explicit_region_raises(self):
        with pytest.raises(ValueError, match="conflicts with the region encoded"):
            _kwargs("https://b.s3.eu-north-1.amazonaws.com/k", region="ap-south-1")

    def test_matching_explicit_region_is_accepted(self):
        uri = "https://b.s3.eu-north-1.amazonaws.com/k"
        assert _kwargs(uri, region="eu-north-1")["region"] == "eu-north-1"

    def test_region_appears_once_when_also_given_in_config(self):
        # obstore rejects the same key twice with "Duplicate key aws_region".
        out = _kwargs("s3://bucket/key", config={"region": "eu-west-1"})
        assert out["region"] == "eu-west-1"
        assert "config" not in out

    def test_virtual_hosted_style_preserved(self):
        uri = "https://b.s3.us-east-1.amazonaws.com/k"
        assert _kwargs(uri)["virtual_hosted_style_request"] is True

    def test_path_style_not_flipped(self):
        uri = "https://s3.us-east-1.amazonaws.com/b/k"
        assert "virtual_hosted_style_request" not in _kwargs(uri)

    @pytest.mark.parametrize("skip_signature", [None, False])
    def test_custom_credential_provider_skips_defaults(
        self, skip_signature: bool | None
    ):
        """Under skip_signature=False the boto3 provider used to replace it."""
        provider = MagicMock()
        caller: dict[str, Any] = dict(credential_provider=provider)
        if skip_signature is not None:
            caller["skip_signature"] = skip_signature
        with patch("rastera.store._boto3_provider") as boto3:
            out = _kwargs("s3://bucket/key", **caller)
        assert not out.get("skip_signature")
        assert out["credential_provider"] is provider
        boto3.assert_not_called()

    def test_skip_signature_false_uses_boto3(self):
        provider = MagicMock()
        provider.config = {"region": "ca-central-1"}
        with patch("rastera.store._boto3_provider", return_value=provider):
            out = _kwargs("s3://bucket/key", skip_signature=False)
        assert "skip_signature" not in out
        assert out["credential_provider"] is provider
        assert out["region"] == "ca-central-1"

    def test_boto3_unavailable_falls_back_to_unsigned(self):
        with patch("rastera.store._boto3_provider", return_value=None):
            out = _kwargs("s3://bucket/key", skip_signature=False)
        assert out["skip_signature"] is True
        assert "credential_provider" not in out

    @pytest.mark.parametrize(
        "keys",
        [
            dict(access_key_id="AKIA", secret_access_key="s"),
            dict(config={"aws_access_key_id": "AKIA", "aws_secret_access_key": "s"}),
        ],
        ids=["kwargs", "config"],
    )
    @pytest.mark.parametrize("skip_signature", [None, False])
    def test_explicit_keys_sign_with_those_keys(
        self, keys: dict[str, Any], skip_signature: bool | None
    ):
        """They were ignored: unsigned by default, and swapped for the boto3
        provider under skip_signature=False, or for unsigned without a profile."""
        caller: dict[str, Any] = dict(keys)
        if skip_signature is not None:
            caller["skip_signature"] = skip_signature
        with patch("rastera.store._boto3_provider") as boto3:
            out = _kwargs("s3://bucket/key", **caller)
        assert not out.get("skip_signature")
        assert "credential_provider" not in out
        boto3.assert_not_called()
        assert {k: v for k, v in out.items() if k in keys} == keys

    def test_local_strips_skip_signature(self):
        assert "skip_signature" not in _kwargs("/tmp/foo.tif", skip_signature=False)

    def test_http_strips_s3_defaults(self):
        # HTTPStore rejects every config kwarg, region included.
        out = _kwargs("https://cdn.example.com/f.tif", skip_signature=True)
        assert out == {}

    def test_plain_http_is_allowed(self):
        """object_store refuses http:// unless told, as "builder error" on the
        first request."""
        out = _kwargs("http://localhost:8000/f.tif", client_options={"timeout": "5s"})
        assert out == {"client_options": {"allow_http": True, "timeout": "5s"}}

    def test_http_rejects_a_request_to_authenticate(self):
        # Silently stripping it would downgrade to an anonymous GET that only
        # fails on private objects.
        with pytest.raises(ValueError, match="cannot sign requests"):
            _kwargs("https://b.s3.wasabisys.com/k", skip_signature=False)

    def test_gs_keeps_caller_kwargs(self):
        assert "region" not in _kwargs("gs://b/k")


# ── _build_store_with ────────────────────────────────────────────────────


class TestBuildStoreWith:
    def test_roots_at_the_bucket(self):
        mock_from_url = MagicMock(return_value="store")
        assert (
            _build_store_with("s3://bucket/key", mock_from_url, MagicMock()) == "store"
        )
        assert mock_from_url.call_args[0][0] == "s3://bucket"

    @pytest.mark.parametrize(
        "uri",
        [
            "https://bucket.s3.eu-north-1.amazonaws.com/k/a.tif?versionId=v1",
            "https://s3.eu-north-1.amazonaws.com/bucket/k/a.tif?X-Amz-Signature=abc",
            "https://bucket.s3.dualstack.us-east-1.amazonaws.com/k/a.tif?X-Amz-Signature=abc",
        ],
    )
    @pytest.mark.parametrize(
        ("build", "http_store"),
        [
            (_build_store, HTTPStore),
            (
                lambda u: _build_store_with(u, obstore_from_url, ObstoreHTTPStore),
                ObstoreHTTPStore,
            ),
        ],
        ids=["async_tiff", "obstore"],
    )
    def test_a_query_string_url_gets_a_plain_http_store(
        self, uri: str, build: Any, http_store: Any
    ):
        """from_url picks the store by host: for an S3 host an S3Store, which
        dropped the query and signed with whatever credentials were around."""
        store = build(uri)
        assert isinstance(store, http_store)
        assert uri in repr(store)

    def test_signing_a_query_string_url_names_the_query(self):
        with pytest.raises(ValueError, match="has a query string"):
            _build_store(
                "https://bucket.s3.eu-north-1.amazonaws.com/k.tif?versionId=v1",
                skip_signature=False,
            )

    def test_local_roots_at_the_filesystem_root(self, tmp_path: Path):
        f = tmp_path / "foo.tif"
        f.write_bytes(b"")
        mock_from_url = MagicMock(return_value="store")
        _build_store_with(str(f), mock_from_url, MagicMock(), skip_signature=False)
        assert mock_from_url.call_args[0][0] == Path(f.resolve().anchor).as_uri()
        assert "skip_signature" not in mock_from_url.call_args[1]
