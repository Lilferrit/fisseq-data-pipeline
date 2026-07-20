import pytest

from fisseq_data_pipeline.utils.variant import classify_variant, strip_variant_tag

# ---------------------------------------------------------------------------
# classify_variant
# ---------------------------------------------------------------------------


class TestClassifyVariant:
    @pytest.mark.parametrize(
        "v,expected",
        [
            ("WT", "WT"),
            ("A1A", "Synonymous"),
            ("A1B", "Single Missense"),
            ("A1fs", "Frameshift"),
            ("A1X", "Nonsense"),
            ("A1*", "Nonsense"),
            ("A5-|A6-", "3nt Deletion"),
            ("A5-|A9-", "Other"),
            ("A5-", "3nt Deletion"),
            ("garbage", "Other"),
        ],
    )
    def test_classify(self, v, expected):
        assert classify_variant(v) == expected


# ---------------------------------------------------------------------------
# strip_variant_tag
# ---------------------------------------------------------------------------


class TestStripVariantTag:
    def test_strips_tag(self):
        assert strip_variant_tag("M1K:downsampled-half") == "M1K"

    def test_untagged_unchanged(self):
        assert strip_variant_tag("M1K") == "M1K"

    def test_strips_only_first_colon(self):
        assert strip_variant_tag("M1K:tag:extra") == "M1K"
