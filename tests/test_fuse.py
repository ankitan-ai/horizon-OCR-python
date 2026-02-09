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
