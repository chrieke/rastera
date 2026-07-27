"""Contract tests binding `_parse_uri` to the real `from_url`.

The rest of the store tests inject a fake ``from_url_fn``, which is what let a
``Duplicate key aws_region`` crash on every HTTPS-form S3 URI ship unnoticed.
These call the real thing. No network: ``repr(S3Store)`` reports the bucket and
prefix obstore parsed, so the split can be checked at construction time.
"""

from __future__ import annotations

import re

import pytest
from async_tiff.store import from_url  # type: ignore[reportMissingImports]

from rastera.store import _build_store, _parse_uri
from tests.test_store import AWS_URIS

OTHER_URIS = [
    "gs://bucket/path/file.tif",
    "https://cdn.example.com/2024/scene.tif",
    "https://cdn.example.com/k.tif?token=xyz",
    "https://b.s3.wasabisys.com/k/a.tif",
]

# Shapes obstore's own URL parser accepts, so its split is an oracle for ours.
NATIVE_URIS = [
    "s3://bucket/path/file.tif",
    "https://bucket.s3.us-east-1.amazonaws.com/path/file.tif",
    "https://s3.ap-southeast-1.amazonaws.com/bucket/path/file.tif",
    "https://bucket.s3.us-east-1.amazonaws.com/my%20key/a%2Bb.tif",
    "gs://bucket/path/file.tif",
]


def _prefix(store: object) -> str:
    match = re.search(r'prefix="([^"]*)"', repr(store))
    return match.group(1) if match else ""


@pytest.mark.parametrize("uri", [row[0] for row in AWS_URIS] + OTHER_URIS)
def test_store_constructs_for_every_supported_uri(uri: str):
    _build_store(uri)


@pytest.mark.parametrize("uri", NATIVE_URIS)
def test_key_matches_the_prefix_obstore_derives(uri: str):
    assert _parse_uri(uri).key == _prefix(from_url(uri))


class TestObstoreConstraints:
    """Why the normalisation exists. If obstore relaxes either, revisit it."""

    def test_region_kwarg_duplicates_a_region_in_the_host(self):
        with pytest.raises(Exception, match="Duplicate key aws_region"):
            from_url("https://b.s3.eu-north-1.amazonaws.com/k", region="eu-north-1")

    def test_http_store_rejects_every_kwarg(self):
        with pytest.raises(Exception, match="Cannot pass config or keyword"):
            from_url("https://cdn.example.com", skip_signature=True)

    @pytest.mark.parametrize(
        "uri",
        [
            "https://bucket.s3-us-west-2.amazonaws.com/k.tif",
            "https://s3-eu-west-1.amazonaws.com/bucket/k.tif",
            "https://s3.amazonaws.com/bucket/k.tif",
            "https://my.dotted.bucket.s3.us-east-1.amazonaws.com/k.tif",
        ],
    )
    def test_forms_obstore_cannot_parse_but_rastera_normalizes(self, uri: str):
        with pytest.raises(Exception, match="did not match any known pattern"):
            from_url(uri)
        _build_store(uri)
