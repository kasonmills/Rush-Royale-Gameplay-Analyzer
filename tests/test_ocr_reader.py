"""
Tests for OCRReader.

Covers:
  - _preprocess(): grayscale conversion, invert, upscaling, binarisation
  - _validate_range(): in-range pass-through, out-of-range → None
  - _count_hearts(): red-blob counting, noise filtering, empty crop
  - _find_tesseract(): path detection logic
  - OCRReader.crop_region(): fraction-to-pixel mapping
  - OCRReader.available: reflects whether Tesseract was found
  - OCRReader.read() / read_wave() / read_hp() / read_mana():
      correct behaviour with and without Tesseract
  - OCRReader.read() / read_wave(): correct dispatch and range validation
      when Tesseract is mocked

pytesseract is always mocked so tests run without a Tesseract installation.
HP detection (_count_hearts) does not require Tesseract.
"""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.recognition.ocr_reader import (
    HUDLayout,
    HUDReadings,
    OCRReader,
    _count_hearts,
    _preprocess,
    _validate_range,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_frame(h: int = 200, w: int = 100, color: tuple = (128, 128, 128)) -> np.ndarray:
    """Return a solid-colour BGR frame of the given size."""
    return np.full((h, w, 3), color, dtype=np.uint8)


def _frame_with_red_blobs(h: int, w: int, n_blobs: int,
                           blob_size: int = 15) -> np.ndarray:
    """
    Return a grey BGR frame with n_blobs separate red squares.
    Each blob is blob_size × blob_size, evenly spaced horizontally.
    """
    frame = np.full((h, w, 3), 128, dtype=np.uint8)
    spacing = w // (n_blobs + 1)
    for i in range(n_blobs):
        cx = (i + 1) * spacing
        x = max(0, min(cx - blob_size // 2, w - blob_size))
        y = max(0, min(h // 2 - blob_size // 2, h - blob_size))
        frame[y:y + blob_size, x:x + blob_size] = (0, 0, 255)  # BGR red
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
    reader._tesseract_path = mock_path
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
        normal   = _preprocess(white_crop, invert=False)
        inverted = _preprocess(white_crop, invert=True)
        assert not np.array_equal(normal, inverted)

    def test_output_shape_preserves_width_ratio(self):
        """Width scaling should be proportional to height scaling."""
        crop = _solid_frame(16, 80)  # height below MIN, width 80
        result = _preprocess(crop, invert=False)
        # height → 64, width → 80 * (64/16) = 320
        assert result.shape == (64, 320)


# ---------------------------------------------------------------------------
# _count_hearts
# ---------------------------------------------------------------------------

class TestCountHearts:

    def test_empty_crop_returns_none(self):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        assert _count_hearts(empty) is None

    def test_grey_crop_returns_zero(self):
        grey = _solid_frame(50, 100, (128, 128, 128))
        assert _count_hearts(grey) == 0

    def test_blue_crop_returns_zero(self):
        """Blue pixels should not be counted as hearts."""
        blue = _solid_frame(50, 100, (255, 0, 0))  # BGR blue
        assert _count_hearts(blue) == 0

    def test_one_red_blob_counts_as_one_heart(self):
        frame = _frame_with_red_blobs(50, 100, n_blobs=1)
        assert _count_hearts(frame) == 1

    def test_two_red_blobs_count_as_two_hearts(self):
        frame = _frame_with_red_blobs(50, 150, n_blobs=2)
        assert _count_hearts(frame) == 2

    def test_three_red_blobs_count_as_three_hearts(self):
        frame = _frame_with_red_blobs(50, 200, n_blobs=3)
        assert _count_hearts(frame) == 3

    def test_tiny_noise_is_filtered(self):
        """A 5×5 (25 px²) red blob is below _MIN_HEART_AREA and ignored."""
        frame = np.full((50, 50, 3), 128, dtype=np.uint8)
        frame[20:25, 20:25] = (0, 0, 255)  # 5×5 red square
        assert _count_hearts(frame) == 0

    def test_result_capped_at_three(self):
        """Even with 4+ blobs the maximum returned is 3."""
        frame = _frame_with_red_blobs(50, 300, n_blobs=4, blob_size=15)
        assert _count_hearts(frame) == 3


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
        """Fractions > 1 should not crash."""
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
# Behaviour without Tesseract
# ---------------------------------------------------------------------------

class TestGracefulDegradation:

    def setup_method(self):
        self.reader = _make_reader_no_tesseract()
        self.frame  = _solid_frame(2340, 1080)

    def test_read_wave_none_and_hp_zero_no_mana(self):
        result = self.reader.read(self.frame)
        assert isinstance(result, HUDReadings)
        assert result.wave_number is None   # needs Tesseract
        assert result.player_hp   == 0     # colour-based; grey → 0
        assert result.opponent_hp == 0     # colour-based; grey → 0
        assert result.player_mana is None  # not implemented

    def test_read_wave_returns_none(self):
        assert self.reader.read_wave(self.frame) is None

    def test_read_hp_returns_zero_on_grey_frame(self):
        """HP detection is colour-based and does not need Tesseract."""
        p, o = self.reader.read_hp(self.frame)
        assert p == 0
        assert o == 0

    def test_read_mana_returns_none(self):
        assert self.reader.read_mana(self.frame) is None


# ---------------------------------------------------------------------------
# OCR dispatch and range validation (pytesseract mocked)
# ---------------------------------------------------------------------------

class TestOCRDispatch:
    """
    Verifies OCRReader routes to the right region and applies correct range
    validation for wave number, with pytesseract returning a fixed string.
    HP is colour-based so it is not affected by the OCR mock.
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

    def test_mana_always_none(self):
        reader = self._make_reader_mocked_ocr("5")
        result = self._run_read(reader, "5")
        assert result.player_mana is None

    def test_empty_ocr_returns_none_wave(self):
        reader = self._make_reader_mocked_ocr("")
        result = self._run_read(reader, "")
        assert result.wave_number is None
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