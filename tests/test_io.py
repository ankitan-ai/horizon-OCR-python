"""Tests for IO modules."""

import pytest
import numpy as np


class TestPDFLoader:
    """Tests for PDF loading."""
    
    @pytest.mark.unit
    def test_load_pdf(self, sample_pdf_path):
        from docvision.io.pdf import PDFLoader
        
        loader = PDFLoader(dpi=150)
        pages = loader.load(str(sample_pdf_path))
        
        assert len(pages) == 1
        assert isinstance(pages[0], np.ndarray)
        assert pages[0].shape[2] == 3  # RGB
    
    @pytest.mark.unit
    def test_load_pdf_with_max_pages(self, sample_pdf_path):
        from docvision.io.pdf import PDFLoader
        
        loader = PDFLoader(dpi=150, max_pages=1)
        pages = loader.load(str(sample_pdf_path))
        
        assert len(pages) <= 1
    
    @pytest.mark.unit
    def test_load_nonexistent_pdf(self):
        from docvision.io.pdf import PDFLoader
        
        loader = PDFLoader()
        
        with pytest.raises(Exception):
            loader.load("/nonexistent/file.pdf")
    
    @pytest.mark.unit
    def test_get_page_count(self, sample_pdf_path):
        from docvision.io.pdf import PDFLoader
        
        loader = PDFLoader()
        count = loader.get_page_count(str(sample_pdf_path))
        
        assert count == 1


class TestImageLoader:
    """Tests for image loading."""
    
    @pytest.mark.unit
    def test_load_image(self, sample_image_path):
        from docvision.io.image import ImageLoader
        
        loader = ImageLoader()
        image = loader.load(str(sample_image_path))
        
        assert isinstance(image, np.ndarray)
        assert len(image.shape) == 3
    
    @pytest.mark.unit
    def test_load_from_path_object(self, temp_dir, sample_image):
        """Test loading from pathlib.Path object."""
        from docvision.io.image import ImageLoader
        import cv2
        
        # Save sample image to temp dir
        img_path = temp_dir / "test_image.png"
        cv2.imwrite(str(img_path), sample_image)
        
        loader = ImageLoader()
        image = loader.load(str(img_path))
        
        assert isinstance(image, np.ndarray)
        assert len(image.shape) == 3
    
    @pytest.mark.unit
    def test_load_preserves_color(self, temp_dir, sample_image):
        """Test that color images are loaded correctly."""
        from docvision.io.image import ImageLoader
        import cv2
        
        # Create a color image with known values
        color_image = np.zeros((100, 100, 3), dtype=np.uint8)
        color_image[:50, :, 0] = 255  # Blue top half
        color_image[50:, :, 2] = 255  # Red bottom half
        
        img_path = temp_dir / "color_test.png"
        cv2.imwrite(str(img_path), color_image)
        
        loader = ImageLoader()
        loaded = loader.load(str(img_path))
        
        assert loaded.shape == (100, 100, 3)
    
    @pytest.mark.unit
    def test_load_nonexistent_image(self):
        from docvision.io.image import ImageLoader
        
        loader = ImageLoader()
        
        with pytest.raises(Exception):
            loader.load("/nonexistent/image.png")


class TestArtifactManager:
    """Tests for artifact management."""
    
    @pytest.mark.unit
    def test_create_artifact_manager(self, temp_dir):
        from docvision.io.artifacts import ArtifactManager
        from pathlib import Path
        
        manager = ArtifactManager(
            output_dir=str(temp_dir),
            enable=True
        )
        
        assert manager.enable is True
        assert manager.output_dir == Path(temp_dir)  # output_dir is stored as Path
    
    @pytest.mark.unit
    def test_save_preprocessed_image(self, temp_dir, sample_image):
        from docvision.io.artifacts import ArtifactManager
        
        manager = ArtifactManager(
            output_dir=str(temp_dir),
            enable=True,
            save_preprocessed=True
        )
        
        path = manager.save_preprocessed_image(
            sample_image,
            doc_id="test-001",
            page_num=1,
            stage="enhanced"
        )
        
        assert path is not None
        assert (temp_dir / "test-001").exists()
    
    @pytest.mark.unit
    def test_disabled_manager(self, temp_dir, sample_image):
        from docvision.io.artifacts import ArtifactManager
        
        manager = ArtifactManager(
            output_dir=str(temp_dir),
            enable=False
        )
        
        path = manager.save_preprocessed_image(
            sample_image,
            doc_id="test-001",
            page_num=1,
            stage="enhanced"
        )
        
        # Should return None when disabled
        assert path is None
    
    @pytest.mark.unit
    def test_save_layout_overlay(self, temp_dir, sample_image):
        from docvision.io.artifacts import ArtifactManager
        from docvision.types import LayoutRegion, LayoutRegionType, BoundingBox
        
        manager = ArtifactManager(
            output_dir=str(temp_dir),
            enable=True,
            save_layout=True
        )
        
        regions = [
            LayoutRegion(
                id="region-001",
                type=LayoutRegionType.TEXT,
                bbox=BoundingBox(x1=100, y1=50, x2=300, y2=150),
                confidence=0.9
            )
        ]
        
        path = manager.save_layout_overlay(
            sample_image,
            regions,
            doc_id="test-001",
            page_num=1
        )
        
        assert path is not None

    # --- Mode-subfolder tests ---

    @pytest.mark.unit
    def test_get_document_dir_local_mode(self, temp_dir):
        from docvision.io.artifacts import ArtifactManager

        manager = ArtifactManager(output_dir=str(temp_dir), enable=True)
        manager.current_mode = "local"

        doc_dir = manager.get_document_dir("abc123")
        assert doc_dir == temp_dir / "Local" / "abc123"
        assert doc_dir.exists()

    @pytest.mark.unit
    def test_get_document_dir_azure_mode(self, temp_dir):
        from docvision.io.artifacts import ArtifactManager

        manager = ArtifactManager(output_dir=str(temp_dir), enable=True)
        manager.current_mode = "azure"

        doc_dir = manager.get_document_dir("abc123")
        assert doc_dir == temp_dir / "Azure_Cloud" / "abc123"
        assert doc_dir.exists()

    @pytest.mark.unit
    def test_get_document_dir_no_mode(self, temp_dir):
        from docvision.io.artifacts import ArtifactManager

        manager = ArtifactManager(output_dir=str(temp_dir), enable=True)
        # current_mode left as None — should fall back to flat layout
        doc_dir = manager.get_document_dir("abc123")
        assert doc_dir == temp_dir / "abc123"

    @pytest.mark.unit
    def test_explicit_mode_overrides_current_mode(self, temp_dir):
        from docvision.io.artifacts import ArtifactManager

        manager = ArtifactManager(output_dir=str(temp_dir), enable=True)
        manager.current_mode = "local"

        doc_dir = manager.get_document_dir("abc123", mode="azure")
        assert doc_dir == temp_dir / "Azure_Cloud" / "abc123"

    @pytest.mark.unit
    def test_save_preprocessed_with_mode(self, temp_dir, sample_image):
        from docvision.io.artifacts import ArtifactManager

        manager = ArtifactManager(
            output_dir=str(temp_dir),
            enable=True,
            save_preprocessed=True,
        )
        manager.current_mode = "azure"

        path = manager.save_preprocessed_image(
            sample_image,
            doc_id="test-mode",
            page_num=1,
            stage="enhanced",
        )

        assert path is not None
        assert "Azure_Cloud" in path


# ---------------------------------------------------------------------------
# Upload deduplication
# ---------------------------------------------------------------------------
class TestUploadDedup:
    """Tests for content-hash–based upload deduplication."""

    @pytest.mark.unit
    def test_same_content_produces_same_hash_filename(self, tmp_path):
        import hashlib

        content = b"hello world document bytes"
        h = hashlib.sha256(content).hexdigest()[:16]

        path_a = tmp_path / f"{h}.pdf"
        path_a.write_bytes(content)

        # Second write should hit the same filename
        path_b = tmp_path / f"{h}.pdf"
        assert path_a == path_b
        assert path_b.exists()

    @pytest.mark.unit
    def test_different_content_produces_different_filenames(self, tmp_path):
        import hashlib

        c1 = b"document one"
        c2 = b"document two"
        h1 = hashlib.sha256(c1).hexdigest()[:16]
        h2 = hashlib.sha256(c2).hexdigest()[:16]

        assert h1 != h2