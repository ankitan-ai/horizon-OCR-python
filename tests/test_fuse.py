"""Tests for rank-and-fuse logic."""

import pytest


class TestRankAndFuse:
    """Tests for field fusion logic."""
    
    @pytest.mark.unit
    def test_create_fuser(self):
        from docvision.kie.fuse import RankAndFuse, FusionStrategy
        
        fuser = RankAndFuse(strategy=FusionStrategy.HIGHEST_CONFIDENCE)
        
        assert fuser.strategy == FusionStrategy.HIGHEST_CONFIDENCE
    
    @pytest.mark.unit
    def test_fuse_identical_fields(self):
        from docvision.kie.fuse import RankAndFuse, FusionStrategy
        from docvision.types import Field, SourceEngine
        
        fuser = RankAndFuse(strategy=FusionStrategy.HIGHEST_CONFIDENCE)
        
        fields_list = [
            [Field(name="total", value="100.00", confidence=0.9, chosen_source=SourceEngine.DONUT)],
            [Field(name="total", value="100.00", confidence=0.85, chosen_source=SourceEngine.LAYOUTLMV3)],
        ]
        
        result = fuser.fuse_fields(fields_list)
        
        # Should have one fused field
        total_fields = [f for f in result if f.name == "total"]
        assert len(total_fields) == 1
        assert total_fields[0].value == "100.00"
    
    @pytest.mark.unit
    def test_fuse_conflicting_fields(self):
        from docvision.kie.fuse import RankAndFuse, FusionStrategy
        from docvision.types import Field, SourceEngine
        
        fuser = RankAndFuse(strategy=FusionStrategy.HIGHEST_CONFIDENCE)
        
        fields_list = [
            [Field(name="total", value="100.00", confidence=0.95, chosen_source=SourceEngine.DONUT)],
            [Field(name="total", value="200.00", confidence=0.85, chosen_source=SourceEngine.LAYOUTLMV3)],
        ]
        
        result = fuser.fuse_fields(fields_list)
        
        # Should pick highest confidence
        total_fields = [f for f in result if f.name == "total"]
        assert len(total_fields) == 1
        assert total_fields[0].value == "100.00"
        assert total_fields[0].confidence == 0.95
    
    @pytest.mark.unit
    def test_fuse_with_weighted_vote(self):
        from docvision.kie.fuse import RankAndFuse, FusionStrategy
        from docvision.types import Field, SourceEngine
        
        fuser = RankAndFuse(
            strategy=FusionStrategy.WEIGHTED_VOTE,
            source_weights={
                SourceEngine.DONUT: 0.6,
                SourceEngine.LAYOUTLMV3: 0.4,
            }
        )
        
        fields_list = [
            [Field(name="date", value="2024-01-15", confidence=0.8, chosen_source=SourceEngine.DONUT)],
            [Field(name="date", value="2024-01-15", confidence=0.9, chosen_source=SourceEngine.LAYOUTLMV3)],
        ]
        
        result = fuser.fuse_fields(fields_list)
        
        date_fields = [f for f in result if f.name == "date"]
        assert len(date_fields) == 1
    
    @pytest.mark.unit
    def test_fuse_preserves_candidates(self):
        from docvision.kie.fuse import RankAndFuse, FusionStrategy
        from docvision.types import Field, SourceEngine
        
        fuser = RankAndFuse(strategy=FusionStrategy.HIGHEST_CONFIDENCE)
        
        fields_list = [
            [Field(name="invoice_num", value="INV-001", confidence=0.9, chosen_source=SourceEngine.DONUT)],
            [Field(name="invoice_num", value="INV-002", confidence=0.7, chosen_source=SourceEngine.LAYOUTLMV3)],
        ]
        
        result = fuser.fuse_fields(fields_list)
        
        inv_fields = [f for f in result if f.name == "invoice_num"]
        assert len(inv_fields) == 1
        # Result should have the higher confidence value
        assert inv_fields[0].value == "INV-001"
    
    @pytest.mark.unit
    def test_fuse_empty_lists(self):
        from docvision.kie.fuse import RankAndFuse, FusionStrategy
        
        fuser = RankAndFuse(strategy=FusionStrategy.HIGHEST_CONFIDENCE)
        
        result = fuser.fuse_fields([])
        
        assert result == []
    
    @pytest.mark.unit
    def test_fuse_single_source(self):
        from docvision.kie.fuse import RankAndFuse, FusionStrategy
        from docvision.types import Field, SourceEngine
        
        fuser = RankAndFuse(strategy=FusionStrategy.HIGHEST_CONFIDENCE)
        
        fields_list = [
            [
                Field(name="total", value="100.00", confidence=0.9, chosen_source=SourceEngine.DONUT),
                Field(name="date", value="2024-01-15", confidence=0.85, chosen_source=SourceEngine.DONUT),
            ]
        ]
        
        result = fuser.fuse_fields(fields_list)
        
        assert len(result) == 2
    
    @pytest.mark.unit
    def test_consensus_strategy(self):
        from docvision.kie.fuse import RankAndFuse, FusionStrategy
        from docvision.types import Field, SourceEngine
        
        fuser = RankAndFuse(
            strategy=FusionStrategy.CONSENSUS
        )
        
        # Two sources agree
        fields_list = [
            [Field(name="total", value="100.00", confidence=0.9, chosen_source=SourceEngine.DONUT)],
            [Field(name="total", value="100.00", confidence=0.85, chosen_source=SourceEngine.LAYOUTLMV3)],
            [Field(name="total", value="200.00", confidence=0.7, chosen_source=SourceEngine.TROCR)],
        ]
        
        result = fuser.fuse_fields(fields_list)
        
        total_fields = [f for f in result if f.name == "total"]
        assert len(total_fields) == 1
        # Should pick the consensus value (100.00) since two sources agree
        assert total_fields[0].value == "100.00"


class TestFuserQualityFilter:
    """Tests for the post-fusion quality filter."""

    @pytest.mark.unit
    def test_filter_drops_empty_fields(self):
        """Empty-value fields from any source should be removed."""
        from docvision.kie.fuse import RankAndFuse, FusionStrategy
        from docvision.types import Field, SourceEngine

        fuser = RankAndFuse(strategy=FusionStrategy.HIGHEST_CONFIDENCE)

        fields_list = [
            [
                Field(name="store_name", value="", confidence=0.93, chosen_source=SourceEngine.DONUT),
                Field(name="total", value="100.00", confidence=0.93, chosen_source=SourceEngine.DONUT),
            ]
        ]
        result = fuser.fuse_fields(fields_list)
        names = [f.name for f in result]
        assert "store_name" not in names
        assert "total" in names

    @pytest.mark.unit
    def test_filter_drops_none_value_fields(self):
        """None-value fields should be removed."""
        from docvision.kie.fuse import RankAndFuse, FusionStrategy
        from docvision.types import Field, SourceEngine

        fuser = RankAndFuse(strategy=FusionStrategy.HIGHEST_CONFIDENCE)

        fields_list = [
            [
                Field(name="telephone", value=None, confidence=0.93, chosen_source=SourceEngine.DONUT),
                Field(name="vendor", value="Acme Corp", confidence=0.8, chosen_source=SourceEngine.DONUT),
            ]
        ]
        result = fuser.fuse_fields(fields_list)
        names = [f.name for f in result]
        assert "telephone" not in names
        assert "vendor" in names

    @pytest.mark.unit
    def test_filter_drops_low_confidence_single_source(self):
        """Ultra-low-confidence single-source fields (e.g. LayoutLMv3 ~6%) should be removed."""
        from docvision.kie.fuse import RankAndFuse, FusionStrategy
        from docvision.types import Field, SourceEngine

        fuser = RankAndFuse(strategy=FusionStrategy.HIGHEST_CONFIDENCE)

        fields_list = [
            [
                Field(name="invoice_number", value="TRAILER#:", confidence=0.06,
                      chosen_source=SourceEngine.LAYOUTLMV3),
                Field(name="total", value="250.00", confidence=0.90,
                      chosen_source=SourceEngine.DONUT),
            ]
        ]
        result = fuser.fuse_fields(fields_list)
        names = [f.name for f in result]
        assert "invoice_number" not in names
        assert "total" in names

    @pytest.mark.unit
    def test_filter_keeps_low_confidence_multi_source(self):
        """Low-confidence field corroborated by multiple sources should be kept."""
        from docvision.kie.fuse import RankAndFuse, FusionStrategy
        from docvision.types import Field, Candidate, SourceEngine

        fuser = RankAndFuse(strategy=FusionStrategy.HIGHEST_CONFIDENCE)

        field_a = Field(name="ref", value="ABC", confidence=0.10,
                        chosen_source=SourceEngine.LAYOUTLMV3,
                        candidates=[
                            Candidate(source=SourceEngine.LAYOUTLMV3, value="ABC", confidence=0.10),
                        ])
        field_b = Field(name="ref", value="ABC", confidence=0.12,
                        chosen_source=SourceEngine.DONUT,
                        candidates=[
                            Candidate(source=SourceEngine.DONUT, value="ABC", confidence=0.12),
                        ])

        result = fuser.fuse_fields([[field_a], [field_b]])
        names = [f.name for f in result]
        assert "ref" in names

    @pytest.mark.unit
    def test_filter_drops_currency_mismatch(self):
        """Currency-typed field whose value is not numeric should be removed."""
        from docvision.kie.fuse import RankAndFuse, FusionStrategy
        from docvision.types import Field, SourceEngine

        fuser = RankAndFuse(strategy=FusionStrategy.HIGHEST_CONFIDENCE)

        fields_list = [
            [
                Field(name="total", value="BATTERY RADIATOR", data_type="currency",
                      confidence=0.93, chosen_source=SourceEngine.DONUT),
                Field(name="subtotal", value="25.50", data_type="currency",
                      confidence=0.90, chosen_source=SourceEngine.DONUT),
            ]
        ]
        result = fuser.fuse_fields(fields_list)
        names = [f.name for f in result]
        assert "total" not in names
        assert "subtotal" in names

    @pytest.mark.unit
    def test_filter_drops_date_mismatch(self):
        """Date-typed field with no digits should be removed."""
        from docvision.kie.fuse import RankAndFuse, FusionStrategy
        from docvision.types import Field, SourceEngine

        fuser = RankAndFuse(strategy=FusionStrategy.HIGHEST_CONFIDENCE)

        fields_list = [
            [
                Field(name="date", value="UNKNOWN_TEXT", data_type="date",
                      confidence=0.93, chosen_source=SourceEngine.DONUT),
                Field(name="invoice_date", value="01/15/2024", data_type="date",
                      confidence=0.90, chosen_source=SourceEngine.DONUT),
            ]
        ]
        result = fuser.fuse_fields(fields_list)
        names = [f.name for f in result]
        assert "date" not in names
        assert "invoice_date" in names

    @pytest.mark.unit
    def test_filter_preserves_good_fields(self):
        """High-confidence, non-empty, correctly-typed fields remain untouched."""
        from docvision.kie.fuse import RankAndFuse, FusionStrategy
        from docvision.types import Field, SourceEngine

        fuser = RankAndFuse(strategy=FusionStrategy.HIGHEST_CONFIDENCE)

        fields_list = [
            [
                Field(name="vendor", value="Mansfield Oil", confidence=0.92,
                      chosen_source=SourceEngine.DONUT),
                Field(name="total", value="$1,234.56", data_type="currency",
                      confidence=0.95, chosen_source=SourceEngine.DONUT),
                Field(name="invoice_date", value="2024-03-15", data_type="date",
                      confidence=0.88, chosen_source=SourceEngine.DONUT),
            ]
        ]
        result = fuser.fuse_fields(fields_list)
        assert len(result) == 3
