"""Correct iris illumination and equalize local contrast as in Ma et al. (2003)."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from IrisLocalization import load_image, localize_iris
from IrisNormalization import normalize_iris


@dataclass(frozen=True)
class EnhancementConfig:
    illumination_block_size: int = 16  # Square blocks used to estimate background light.
    equalization_block_size: int = 32  # Square regions equalized independently.


@dataclass(frozen=True)
class EnhancementResult:
    enhanced: np.ndarray
    background: np.ndarray
    corrected: np.ndarray
    corrected_uint8: np.ndarray
    valid_mask: np.ndarray


def _estimate_background(image: np.ndarray, mask: np.ndarray, block_size: int) -> np.ndarray:
    height, width = image.shape
    grid_height, grid_width = height // block_size, width // block_size
    block_shape = (grid_height, block_size, grid_width, block_size)
    counts = mask.reshape(block_shape).sum(axis=(1, 3))
    sums = (image * mask).reshape(block_shape).sum(axis=(1, 3))
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    # Empty blocks borrow the nearest available block mean, without using padding zeros.
    missing = np.argwhere(counts == 0)
    available = np.argwhere(counts > 0)
    for row, col in missing:
        nearest = available[np.argmin(((available - [row, col])**2).sum(axis=1))]
        means[row, col] = means[tuple(nearest)]
    background = cv2.resize(means.astype(np.float32), (width, height),
                             interpolation=cv2.INTER_CUBIC)
    return np.clip(background, 0, 255).astype(np.float32)


def _rescale_corrected(corrected: np.ndarray, mask: np.ndarray) -> np.ndarray:
    low, high = float(corrected[mask].min()), float(corrected[mask].max())
    # Preserve signed subtraction first, then map valid residuals to 8-bit histogram bins.
    # A sub-0.001 gray-level range is interpolation roundoff, not useful contrast.
    if high - low <= 1e-3:
        output = np.full(corrected.shape, 128, dtype=np.uint8)
    else:
        scaled = (corrected - low) * (255.0 / (high - low))
        output = np.clip(np.rint(scaled), 0, 255).astype(np.uint8)
    output[~mask] = 0
    return output


def _equalize_blocks(image: np.ndarray, mask: np.ndarray, block_size: int) -> np.ndarray:
    output = np.zeros_like(image)
    height, width = image.shape
    # Each region has its own histogram; only valid pixels contribute to its CDF.
    for row in range(0, height, block_size):
        for col in range(0, width, block_size):
            tile = image[row:row + block_size, col:col + block_size]
            tile_mask = mask[row:row + block_size, col:col + block_size]
            values = tile[tile_mask]
            if values.size == 0:
                continue
            histogram = np.bincount(values, minlength=256)
            cdf = histogram.cumsum()
            cdf_min = cdf[np.flatnonzero(histogram)[0]]
            denominator = values.size - cdf_min
            target = output[row:row + block_size, col:col + block_size]
            if denominator == 0:
                target[tile_mask] = values
                continue
            lookup = np.clip(np.rint((cdf - cdf_min) * (255.0 / denominator)),
                             0, 255).astype(np.uint8)
            target[tile_mask] = lookup[values]
    return output


def enhance_iris(normalized: np.ndarray, valid_mask: np.ndarray | None = None,
                 config: EnhancementConfig | None = None) -> EnhancementResult:
    """Return a uint8 enhanced image plus intermediate float32 illumination data."""
    cfg = config or EnhancementConfig()
    if normalized.ndim != 2 or normalized.size == 0:
        raise ValueError("Expected a nonempty two-dimensional normalized image.")
    image = normalized.astype(np.float32)
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 255:
        raise ValueError("Input intensities must be finite and within 0-255.")
    mask = np.ones(image.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask)
    if mask.shape != image.shape or mask.dtype != np.bool_ or not mask.any():
        raise ValueError("The validity mask must be boolean, match the image, and contain valid points.")
    for size in (cfg.illumination_block_size, cfg.equalization_block_size):
        if not isinstance(size, int) or size < 1 or any(length % size for length in image.shape):
            raise ValueError("Block sizes must be positive integers dividing both image dimensions.")

    background = _estimate_background(image, mask, cfg.illumination_block_size)
    corrected = image - background
    corrected[~mask] = 0
    corrected_uint8 = _rescale_corrected(corrected, mask)
    enhanced = _equalize_blocks(corrected_uint8, mask, cfg.equalization_block_size)
    return EnhancementResult(enhanced, background, corrected, corrected_uint8, mask.copy())


def save_enhancement(normalized: np.ndarray, result: EnhancementResult,
                      output_dir: str | Path, stem: str) -> dict[str, Path]:
    """Save intermediate previews, a three-strip comparison, and reusable arrays."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    original = np.clip(np.rint(normalized), 0, 255).astype(np.uint8)
    background = np.clip(np.rint(result.background), 0, 255).astype(np.uint8)
    comparison = np.vstack([original, result.corrected_uint8, result.enhanced])
    images = {"illumination": background, "corrected": result.corrected_uint8,
              "enhanced": result.enhanced, "comparison": comparison}
    paths = {}
    for label, pixels in images.items():
        path = output_dir / f"{stem}_{label}.png"
        ok, encoded = cv2.imencode(".png", pixels)
        if not ok:
            raise ValueError(f"Cannot encode {label} preview.")
        encoded.tofile(path)
        paths[label] = path
    array_path = output_dir / f"{stem}_enhanced.npz"
    np.savez_compressed(array_path, enhanced=result.enhanced, background=result.background,
                        corrected=result.corrected, corrected_uint8=result.corrected_uint8,
                        valid_mask=result.valid_mask)
    paths["arrays"] = array_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--image", type=Path, help="Original iris image; localize and normalize first")
    inputs.add_argument("--normalized", type=Path, help="NPZ saved by IrisNormalization.py")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parent / "CASIA Iris Image Database (version 1.0)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "outputs" / "enhancement")
    args = parser.parse_args()
    cfg = EnhancementConfig()
    source = {}
    try:
        if args.normalized is not None:
            with np.load(args.normalized, allow_pickle=False) as saved:
                normalized, mask = saved["normalized"], saved["valid_mask"]
            stem = args.normalized.stem.removesuffix("_normalized")
            source["normalized_file"] = str(args.normalized.resolve())
        else:
            image_path = args.image
            if image_path is None:
                image_path = next(iter(sorted(args.data_dir.rglob("*.bmp"))), None)
                if image_path is None:
                    parser.error("No BMP images found. Set --image, --normalized, or --data-dir.")
            image = load_image(image_path)
            boundaries = localize_iris(image)
            norm = normalize_iris(image, boundaries)
            normalized, mask, stem = norm.normalized, norm.valid_mask, image_path.stem
            source = {"source_image": str(image_path.resolve()), "boundaries": boundaries.to_dict()}
        result = enhance_iris(normalized, mask, cfg)
        paths = save_enhancement(normalized, result, args.output_dir, stem)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Enhancement failed: {error}\n")
    metadata = {
        **source, "shape": list(result.enhanced.shape), "parameters": asdict(cfg),
        "valid_fraction": float(result.valid_mask.mean()),
        "corrected_valid_range": [float(result.corrected[mask].min()), float(result.corrected[mask].max())],
        "histogram_conversion": "Linear min-max mapping of valid signed residuals to 0-255; constants map to 128.",
        "equalization": "Independent local histogram equalization using valid-pixel CDFs.",
        "comparison_order": "Top: normalized; middle: rescaled illumination-corrected; bottom: enhanced.",
        "mask_meaning": "In-image coordinates only; not an eyelid or eyelash mask.",
    }
    metadata_path = args.output_dir / f"{stem}_enhancement.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Enhanced shape: {result.enhanced.shape}")
    print(f"Output intensity range: {result.enhanced.min()}-{result.enhanced.max()}")
    for label, path in paths.items():
        print(f"{label.capitalize()}: {path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
