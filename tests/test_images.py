"""
test_images.py — Round 3 image puzzles: transforms and the pictures they write.
"""

import numpy as np
from PIL import Image

from blackbox_game.images import (
    SOLARISE_THRESHOLD, channel_shuffle, fft_decode, fft_encode, load_curated,
    posterise, render_image_puzzle, repeated_overlay, solarise,
)
from blackbox_game.puzzles import PUZZLE_REGISTRY, get_puzzle


def _pixel(value):
    return np.full((2, 2, 3), value, dtype=np.uint8)


class TestTransforms:

    def test_solarise_inverts_bright_pixels_only(self):
        out = solarise(np.concatenate([_pixel(SOLARISE_THRESHOLD - 1), _pixel(200)]))
        assert np.all(out[:2] == SOLARISE_THRESHOLD - 1)
        assert np.all(out[2:] == 255 - 200)

    def test_posterise_has_four_levels(self):
        out = posterise(np.arange(256, dtype=np.uint8).reshape(16, 16, 1).repeat(3, axis=2))
        assert set(np.unique(out)) <= {0, 85, 170, 255}
        assert len(np.unique(out)) == 4

    def test_channel_shuffle_rgb_to_brg(self):
        img = np.array([[[10, 20, 30]]], dtype=np.uint8)
        assert channel_shuffle(img).tolist() == [[[30, 10, 20]]]

    def test_overlay_keeps_range_and_shape(self):
        img = load_curated("matrix")
        out = repeated_overlay(img)
        assert out.shape == img.shape and out.dtype == np.uint8

    def test_fft_round_trip_is_close(self):
        img = load_curated("istockphoto")
        gray = np.asarray(Image.fromarray(img).convert("L"), dtype=float)
        recon = fft_decode(fft_encode(img))[..., 0].astype(float)
        mse = np.mean((recon - gray) ** 2)
        psnr = 10 * np.log10(255.0 ** 2 / mse)
        assert psnr > 40

    def test_encoded_spectrum_is_256_square_rgb(self):
        encoded = fft_encode(load_curated("chessboard"))
        assert encoded.shape == (256, 256, 3)
        assert encoded[..., 2].max() == 0  # blue channel is unused


class TestImagePuzzles:

    def test_every_image_puzzle_writes_two_pictures(self, tmp_path):
        for pid, puzzle in PUZZLE_REGISTRY.items():
            if puzzle.function.type != "image":
                continue
            params = puzzle.function.parameters
            input_png, output_png = render_image_puzzle(
                params["image"], params["transform"], tmp_path / pid,
            )
            assert input_png.name == "input.png" and output_png.name == "output.png"
            assert Image.open(input_png).size == (256, 256)
            assert Image.open(output_png).size == (256, 256)

    def test_image_puzzles_are_difficulty_three(self):
        images = [p for p in PUZZLE_REGISTRY.values() if p.function.type == "image"]
        assert len(images) == 6
        assert all(p.difficulty == 3 for p in images)

    def test_decode_puzzle_input_is_an_encoded_spectrum(self, tmp_path):
        puzzle = get_puzzle("fft_decode")
        input_png, _ = render_image_puzzle(
            puzzle.function.parameters["image"], "fft_decode", tmp_path,
        )
        assert input_png.exists()
