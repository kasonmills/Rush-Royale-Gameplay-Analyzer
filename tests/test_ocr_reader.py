"""
Tests for OCRReader.

Covers:
  - _preprocess(): grayscale conversion, invert, upscaling, binarisation
  - _validate_range(): in-range pass-through, out-of-range → None
  - _find_tesseract(): path detection logic
  - OCRReader.crop_region(): fraction-to-pixel mapping
  - OCRReader.available: reflects whether Tesseract was found
  - OCRReader.read() / read_wave() / read_hp() / read_mana():
      graceful None returns when Tesseract is unavailable
  - OCRReader.read() / read_wave() / read_hp() / read_mana():
      correct dispatch and range validation when Tesseract is mocked

pytesseract is always mocked so tests run without a Tesseract installation.
"""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.recognition.ocr_reader import (
    HUDLayout,
    HUDReadings,
    OCRReader,
    _preprocess,
    _validate_range,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_frame(h: int = 200, w: int = 100, color: tuple = (128, 128, 128)) -> np.ndarray:
    """Return a solid-colour BGR frame of the given size."""
    frame = np.full((h, w, 3), color, dtype=np.uint8)
    return frame


def _make_reader_no_tesseract() -> OCRReader:
    """Return an OCRReader whose Tesseract path was not found."""
    with patch("src.recognition.ocr_reader._find_tesseract", return_value=None):
        reader = OCRReader()
    return reader


def _make_reader_with_tesseract(mock_path: str = r"C:\fake\tesseract.exe") -> OCRReader:
    """Return an OCRReader that believes Tesseract is at mock_path."""
    with patch("src.recognition.ocr_reader._find_tesseract", return_value=mock_path):
        reader = OCRReader()
    reader._tesseract_path = mock_path  # ensure available == True
    return reader


# ---------------------------------------------------------------------------
# _validate_range
# ---------------------------------------------------------------------------

class TestValidateRange:

    def test_value_in_range_returned(self):
        assert _validate_range(5, 0, 9) == 5

    def test_value_at_lower_bound_returned(self):
        assert _validate_range(0, 0, 3) == 0

    def test_value_at_upper_bound_returned(self):
        assert _validate_range(3, 0, 3) == 3

    def test_value_below_range_returns_none(self):
        assert _validate_range(-1, 0, 3) is None

    def test_value_above_range_returns_none(self):
        assert _validate_range(4, 0, 3) is None

    def test_none_input_returns_none(self):
        assert _validate_range(None, 0, 3) is None

    def test_wave_range_valid(self):
        assert _validate_range(15, 1, 99) == 15

    def test_wave_zero_rejected(self):
        assert _validate_range(0, 1, 99) is None

    def test_wave_100_rejected(self):
        assert _validate_range(100, 1, 99) is None


# ---------------------------------------------------------------------------
# _preprocess
# ---------------------------------------------------------------------------

class TestPreprocess:

    def test_returns_2d_binary_image(self):
        crop = _solid_frame(50, 60, (100, 150, 200))
        result = _preprocess(crop, invert=False)
        assert result.ndim == 2
        assert set(np.unique(result)).issubset({0, 255})

    def test_small_crop_upscaled(self):
        """A crop shorter than _MIN_OCR_HEIGHT (32) should be upscaled to 64."""
        crop = _solid_frame(20, 40)
        result = _preprocess(crop, invert=False)
        assert result.shape[0] == 64

    def test_tall_crop_not_downscaled(self):
        """A crop already ≥ 64 px tall should not be resized."""
        crop = _solid_frame(80, 40)
        result = _preprocess(crop, invert=False)
        assert result.shape[0] == 80

    def test_invert_flips_values(self):
        """A white crop inverted should become black after binarisation."""
        white_crop = _solid_frame(50, 50, (255, 255, 255))
        normal  = _preprocess(white_crop, invert=False)
        inverted = _preprocess(white_crop, invert=True)
        # After invert, white→black; binarisation may produce different dominant value
        assert not np.array_equal(normal, inverted)

    def test_output_shape_preserves_width_ratio(self):
        """Width scaling should be proportional to height scaling."""
        crop = _solid_frame(16, 80)  # height below MIN, width 80
        result = _preprocess(crop, invert=False)
        # height should be 64, width should be 80 * (64/16) = 320
        assert result.shape == (64, 320)


# ---------------------------------------------------------------------------
# OCRReader.crop_region
# ---------------------------------------------------------------------------

class TestCropRegion:

    def test_full_frame_region(self):
        frame = _solid_frame(100, 200)
        reader = _make_reader_no_tesseract()
        crop = reader.crop_region(frame, (0.0, 0.0, 1.0, 1.0))
        assert crop.shape[:2] == (100, 200)

    def test_half_region(self):
        frame = _solid_frame(100, 200)
        reader = _make_reader_no_tesseract()
        crop = reader.crop_region(frame, (0.0, 0.0, 0.5, 0.5))
        assert crop.shape[:2] == (50, 100)

    def test_returns_copy(self):
        frame = _solid_frame(100, 200)
        reader = _make_reader_no_tesseract()
        crop = reader.crop_region(frame, (0.0, 0.0, 1.0, 1.0))
        crop[:] = 0
        assert frame[0, 0, 0] != 0  # original unmodified

    def test_clamps_to_frame_bounds(self):
        """Fractions > 1 or producing negative sizes should not crash."""
        frame = _solid_frame(100, 200)
        reader = _make_reader_no_tesseract()
        crop = reader.crop_region(frame, (0.0, 0.0, 1.5, 1.5))
        assert crop.shape[:2] == (100, 200)


# ---------------------------------------------------------------------------
# OCRReader.available
# ---------------------------------------------------------------------------

class TestAvailable:

    def test_available_false_when_tesseract_not_found(self):
        reader = _make_reader_no_tesseract()
        assert reader.available is False

    def test_available_true_when_tesseract_found(self):
        reader = _make_reader_with_tesseract()
        assert reader.available is True


# ---------------------------------------------------------------------------
# Graceful degradation — all reads return None when Tesseract unavailable
# ---------------------------------------------------------------------------

class TestGracefulDegradation:

    def setup_method(self):
        self.reader = _make_reader_no_tesseract()
        self.frame  = _solid_frame(2340, 1080)

    def test_read_returns_all_none(self):
        result = self.reader.read(self.frame)
        assert isinstance(result, HUDReadings)
        assert result.wave_number  is None
        assert result.player_hp    is None
        assert result.opponent_hp  is None
        assert result.player_mana  is None

    def test_read_wave_returns_none(self):
        assert self.reader.read_wave(self.frame) is None

    def test_read_hp_returns_none_tuple(self):
        p, o = self.reader.read_hp(self.frame)
        assert p is None
        assert o is None

    def test_read_mana_returns_none(self):
        assert self.reader.read_mana(self.frame) is None


# ---------------------------------------------------------------------------
# OCR dispatch and range validation (pytesseract mocked)
# ---------------------------------------------------------------------------

class TestOCRDispatch:
    """
    Tests that verify OCRReader routes to the right region and applies
    the correct range validation, with pytesseract returning a fixed string.
    """

    def _make_reader_mocked_ocr(self, ocr_return: str) -> OCRReader:
        reader = _make_reader_with_tesseract()
        mock_pts = MagicMock()
        mock_pts.image_to_string.return_value = ocr_return
        reader._mock_pts = mock_pts
        return reader

    def _run_read(self, reader: OCRReader, ocr_return: str) -> HUDReadings:
        frame = _solid_frame(2340, 1080)
        with patch("src.recognition.ocr_reader._pytesseract_mod", reader._mock_pts):
            return reader.read(frame)

    def test_valid_wave_passed_through(self):
        reader = self._make_reader_mocked_ocr("12")
        result = self._run_read(reader, "12")
        assert result.wave_number == 12

    def test_wave_0_rejected(self):
        reader = self._make_reader_mocked_ocr("0")
        result = self._run_read(reader, "0")
        assert result.wave_number is None

    def test_wave_100_rejected(self):
        reader = self._make_reader_mocked_ocr("100")
        result = self._run_read(reader, "100")
        assert result.wave_number is None

    def test_hp_3_passed_through(self):
        reader = self._make_reader_mocked_ocr("3")
        result = self._run_read(reader, "3")
        assert result.player_hp == 3
        assert result.opponent_hp == 3

    def test_hp_4_rejected(self):
        reader = self._make_reader_mocked_ocr("4")
        result = self._run_read(reader, "4")
        assert result.player_hp is None
        assert result.opponent_hp is None

    def test_mana_9_passed_through(self):
        reader = self._make_reader_mocked_ocr("9")
        result = self._run_read(reader, "9")
        assert result.player_mana == 9

    def test_mana_10_rejected(self):
        reader = self._make_reader_mocked_ocr("10")
        result = self._run_read(reader, "10")
        assert result.player_mana is None

    def test_empty_ocr_returns_none(self):
        reader = self._make_reader_mocked_ocr("")
        result = self._run_read(reader, "")
        assert result.wave_number is None
        assert result.player_hp   is None
        assert result.player_mana is None

    def test_noisy_ocr_strips_non_digits(self):
        """'l2' (OCR noise) should parse as '2' → wave 2."""
        reader = self._make_reader_mocked_ocr("l2")
        result = self._run_read(reader, "l2")
        assert result.wave_number == 2


# ---------------------------------------------------------------------------
# HUDLayout customisation
# ---------------------------------------------------------------------------

class TestHUDLayout:

    def test_custom_layout_stored(self):
        layout = HUDLayout(wave=(0.4, 0.4, 0.6, 0.6))
        reader = _make_reader_no_tesseract()
        reader.layout = layout
        assert reader.layout.wave == (0.4, 0.4, 0.6, 0.6)

    def test_invert_flag_stored(self):
        reader = OCRReader.__new__(OCRReader)
        reader.invert = True
        assert reader.invert is True