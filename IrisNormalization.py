"""Unwrap an iris into a 64 x 512 image using Ma et al. (2003), Eq. (2)."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from IrisLocalization import LocalizationResult, load_image, localize_iris


@dataclass(frozen=True)
class NormalizationConfig:
    radial_samples: int = 64  # Rows, from the pupil toward the outer iris boundary.
    angular_samples: int = 512  # Columns, covering one counterclockwise revolution.
    start_angle_degrees: float = 0.0  # Starting angle for the seven matching templates.


@dataclass(frozen=True)
class NormalizationResult:
    normalized: np.ndarray
    valid_mask: np.ndarray

    @property
    def valid_fraction(self) -> float:
        return float(self.valid_mask.mean())


def normalize_iris(image: np.ndarray, boundaries: LocalizationResult,
                   config: NormalizationConfig | None = None) -> NormalizationResult:
    """Return float32 intensities (0-255) and a mask of in-image sampling points."""
    cfg = config or NormalizationConfig()
    if image.ndim != 2 or image.dtype != np.uint8 or image.size == 0:
        raise ValueError("Expected a nonempty two-dimensional uint8 grayscale image.")
    for count in (cfg.radial_samples, cfg.angular_samples):
        if not isinstance(count, int) or count < 2:
            raise ValueError("Radial and angular sample counts must be integers >= 2.")
    if not np.isfinite(cfg.start_angle_degrees):
        raise ValueError("The starting angle must be finite.")
    pupil, iris = boundaries.pupil, boundaries.iris
    values = [pupil.x, pupil.y, pupil.radius, iris.x, iris.y, iris.radius]
    if not np.isfinite(values).all() or min(pupil.radius, iris.radius) <= 0:
        raise ValueError("Circle coordinates must be finite and radii must be positive.")
    center_distance = np.hypot(iris.x - pupil.x, iris.y - pupil.y)
    if center_distance + pupil.radius >= iris.radius:
        raise ValueError("The outer iris circle must strictly contain the pupil circle.")

    # Exclude 2*pi to avoid repeating the first column; use Y/M as in Eq. (2).
    theta = (np.deg2rad(cfg.start_angle_degrees % 360)
             + 2 * np.pi * np.arange(cfg.angular_samples) / cfg.angular_samples)
    radial_fraction = np.arange(cfg.radial_samples)[:, None] / cfg.radial_samples
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    pupil_x = pupil.x + pupil.radius * cos_theta
    iris_x = iris.x + iris.radius * cos_theta
    # Image y increases downward, so minus sin(theta) unwraps counterclockwise.
    pupil_y = pupil.y - pupil.radius * sin_theta
    iris_y = iris.y - iris.radius * sin_theta
    map_x = (pupil_x + radial_fraction * (iris_x - pupil_x)).astype(np.float32)
    map_y = (pupil_y + radial_fraction * (iris_y - pupil_y)).astype(np.float32)

    height, width = image.shape
    valid_mask = ((map_x >= 0) & (map_x <= width - 1) &
                  (map_y >= 0) & (map_y <= height - 1))
    # Bilinear interpolation preserves fractional coordinates and intensity values.
    normalized = cv2.remap(image.astype(np.float32), map_x, map_y,
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=0)
    normalized[~valid_mask] = 0
    return NormalizationResult(normalized, valid_mask)


def save_normalization(result: NormalizationResult, output_dir: str | Path,
                       stem: str) -> dict[str, Path]:
    """Save a grayscale preview, geometric validity mask, and lossless arrays."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "preview": output_dir / f"{stem}_normalized.png",
        "mask": output_dir / f"{stem}_valid_mask.png",
        "arrays": output_dir / f"{stem}_normalized.npz",
    }
    preview = np.clip(np.rint(result.normalized), 0, 255).astype(np.uint8)
    mask_preview = result.valid_mask.astype(np.uint8) * 255
    for key, pixels in [("preview", preview), ("mask", mask_preview)]:
        ok, encoded = cv2.imencode(".png", pixels)
        if not ok:
            raise ValueError(f"Cannot encode {key} image.")
        encoded.tofile(paths[key])
    np.savez_compressed(paths["arrays"], normalized=result.normalized,
                        valid_mask=result.valid_mask)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Path to one iris image")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parent / "CASIA Iris Image Database (version 1.0)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "outputs" / "normalization")
    parser.add_argument("--angle", type=float, default=0.0, help="Starting angle in degrees")
    args = parser.parse_args()
    image_path = args.image
    if image_path is None:
        image_path = next(iter(sorted(args.data_dir.rglob("*.bmp"))), None)
        if image_path is None:
            parser.error("No BMP images found. Set --image or --data-dir.")
    config = NormalizationConfig(start_angle_degrees=args.angle)
    try:
        image = load_image(image_path)
        boundaries = localize_iris(image)
        result = normalize_iris(image, boundaries, config)
        paths = save_normalization(result, args.output_dir, image_path.stem)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Normalization failed: {error}\n")
    metadata = {
        "source_image": str(image_path.resolve()),
        "source_shape": list(image.shape),
        "normalized_shape": list(result.normalized.shape),
        "parameters": asdict(config),
        "valid_fraction": result.valid_fraction,
        "boundaries": boundaries.to_dict(),
        "angle_convention": "Counterclockwise from the right plus start_angle_degrees; default: right, up, left, down.",
        "radial_convention": "Y/M, Y=0,...,M-1; pupil boundary to just inside outer boundary.",
        "mask_meaning": "In-image coordinates only; not an eyelid or eyelash mask.",
    }
    metadata_path = args.output_dir / f"{image_path.stem}_normalization.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Normalized shape: {result.normalized.shape}")
    print(f"In-image sampling points: {100 * result.valid_fraction:.2f}%")
    for label, path in paths.items():
        print(f"{label.capitalize()}: {path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
