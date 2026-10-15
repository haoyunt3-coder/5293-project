"""Locate the pupil and outer iris boundary in a CASIA-IrisV1 image."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class LocalizationConfig:
    """Search settings for the supplied 320 x 280 grayscale images."""

    # Radius and offset values below are measured in pixels.
    pupil_radius_min: float = 15.0
    pupil_radius_max: float = 65.0
    pupil_threshold_max: int = 120  # Upper grayscale threshold for dark components.
    iris_radius_min: float = 65.0
    iris_radius_max: float = 135.0
    pupil_iris_gap: float = 20.0  # Minimum separation between the two radii.
    center_search: int = 16  # Maximum iris-center displacement on either axis.
    coarse_step: int = 4  # Center-grid spacing before one-pixel refinement.
    radial_offset: float = 3.0  # Compare intensities this far inside/outside a circle.
    sector_half_angle: float = 40.0  # Half-width of each side sector in degrees.
    angular_samples: int = 160  # Total sample points across both sectors.


@dataclass(frozen=True)
class Circle:
    x: float
    y: float
    radius: float


@dataclass(frozen=True)
class LocalizationResult:
    pupil: Circle
    iris: Circle
    pupil_circularity: float
    iris_edge_contrast: float

    def to_dict(self) -> dict:
        return asdict(self)


def load_image(path: str | Path) -> np.ndarray:
    """Read a grayscale image, including paths that contain non-ASCII text."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def _find_pupil(image: np.ndarray, cfg: LocalizationConfig) -> tuple[Circle, float]:
    height, width = image.shape
    # Several dark thresholds handle differences in pupil brightness between images.
    levels = np.unique(np.r_[np.arange(25, cfg.pupil_threshold_max + 1, 10),
                             np.percentile(image, [2, 5, 8, 12, 18])])
    best = None
    best_score = -np.inf
    for threshold in levels[levels <= cfg.pupil_threshold_max]:
        mask = (image <= threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if not np.pi * cfg.pupil_radius_min**2 <= area <= np.pi * cfg.pupil_radius_max**2:
                continue
            # Ignore one-pixel stair steps when measuring a contour's roundness.
            smooth_contour = cv2.approxPolyDP(contour, 1.5, True)
            perimeter = cv2.arcLength(smooth_contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * cv2.contourArea(smooth_contour) / perimeter**2
            (x, y), radius = cv2.minEnclosingCircle(contour)
            fill_ratio = area / (np.pi * radius**2)
            if circularity < 0.65 or fill_ratio < 0.75:
                continue
            if not (0.1 * width < x < 0.9 * width and 0.1 * height < y < 0.9 * height):
                continue
            if not cfg.pupil_radius_min <= radius <= cfg.pupil_radius_max:
                continue
            center_distance = np.hypot((x - width / 2) / width,
                                       (y - height / 2) / height)
            score = circularity + fill_ratio - 0.15 * center_distance
            if score > best_score:
                best_score = score
                best = (Circle(float(x), float(y), float(radius)), float(circularity))
    if best is None:
        raise ValueError("No circular dark pupil found. Check the image or pupil search settings.")
    return best


def _find_iris(image: np.ndarray, pupil: Circle,
               cfg: LocalizationConfig) -> tuple[Circle, float]:
    # Side sectors reduce interference from the upper and lower eyelids.
    half = np.deg2rad(cfg.sector_half_angle)
    angles = np.r_[np.linspace(-half, half, cfg.angular_samples // 2),
                   np.linspace(np.pi - half, np.pi + half, cfg.angular_samples // 2)]
    cos_theta, sin_theta = np.cos(angles), np.sin(angles)
    radius_min = max(cfg.iris_radius_min, pupil.radius + cfg.pupil_iris_gap)
    radii = np.arange(np.ceil(radius_min), np.floor(cfg.iris_radius_max) + 1)
    if radii.size == 0:
        raise ValueError("The iris radius range is empty.")
    blurred = cv2.GaussianBlur(image, (5, 5), 1.0).astype(np.float32)
    height, width = image.shape

    def evaluate_center(cx: float, cy: float) -> tuple[float, float]:
        inner_r = radii[:, None] - cfg.radial_offset
        outer_r = radii[:, None] + cfg.radial_offset
        inner_x, inner_y = cx + inner_r * cos_theta, cy + inner_r * sin_theta
        outer_x, outer_y = cx + outer_r * cos_theta, cy + outer_r * sin_theta
        valid = ((inner_x >= 0) & (inner_x < width - 1) &
                 (outer_x >= 0) & (outer_x < width - 1) &
                 (inner_y >= 0) & (inner_y < height - 1) &
                 (outer_y >= 0) & (outer_y < height - 1))
        inner = cv2.remap(blurred, inner_x.astype(np.float32), inner_y.astype(np.float32),
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        outer = cv2.remap(blurred, outer_x.astype(np.float32), outer_y.astype(np.float32),
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # A limbus changes from darker iris to brighter sclera as radius increases.
        contrast = np.sum((outer - inner) * valid, axis=1) / np.maximum(valid.sum(axis=1), 1)
        contrast[valid.mean(axis=1) < 0.75] = -np.inf
        separation = pupil.radius + np.hypot(cx - pupil.x, cy - pupil.y) + cfg.pupil_iris_gap
        contrast[radii < separation] = -np.inf
        index = int(np.argmax(contrast))
        return float(contrast[index]), float(radii[index])

    best_score, best_circle = -np.inf, None
    offsets = range(-cfg.center_search, cfg.center_search + 1, cfg.coarse_step)
    # Search center and radius independently; pupil and iris need not be concentric.
    for dx in offsets:
        for dy in offsets:
            score, radius = evaluate_center(pupil.x + dx, pupil.y + dy)
            if score > best_score:
                best_score = score
                best_circle = Circle(pupil.x + dx, pupil.y + dy, radius)
    if best_circle is None:
        raise ValueError("No valid outer iris boundary found within the search region.")
    coarse_circle = best_circle
    for dx in range(-cfg.coarse_step + 1, cfg.coarse_step):
        for dy in range(-cfg.coarse_step + 1, cfg.coarse_step):
            cx, cy = coarse_circle.x + dx, coarse_circle.y + dy
            if abs(cx - pupil.x) > cfg.center_search or abs(cy - pupil.y) > cfg.center_search:
                continue
            score, radius = evaluate_center(cx, cy)
            if score > best_score:
                best_score, best_circle = score, Circle(cx, cy, radius)
    if best_score <= 0:
        raise ValueError("The outer boundary has no positive iris-to-sclera contrast.")
    return best_circle, best_score


def localize_iris(image: np.ndarray,
                  config: LocalizationConfig | None = None) -> LocalizationResult:
    """Return two circles in original image coordinates; x is column and y is row."""
    cfg = config or LocalizationConfig()
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("Expected a two-dimensional uint8 grayscale image.")
    if image.size == 0:
        raise ValueError("The image is empty.")
    if cfg.coarse_step < 1 or cfg.center_search < 0 or cfg.angular_samples < 4:
        raise ValueError("Invalid center search or angular sampling settings.")
    pupil, circularity = _find_pupil(image, cfg)
    iris, contrast = _find_iris(image, pupil, cfg)
    return LocalizationResult(pupil, iris, circularity, contrast)


def save_overlay(image: np.ndarray, result: LocalizationResult, path: str | Path) -> None:
    """Save the original image with the pupil in green and outer boundary in red."""
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for circle, color in [(result.pupil, (0, 200, 0)), (result.iris, (0, 0, 255))]:
        center = (round(circle.x), round(circle.y))
        cv2.circle(canvas, center, round(circle.radius), color, 1, cv2.LINE_AA)
        cv2.drawMarker(canvas, center, color, cv2.MARKER_CROSS, 7, 1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise ValueError("Cannot encode the localization preview.")
    encoded.tofile(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Path to one iris image")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parent / "CASIA Iris Image Database (version 1.0)",
                        help="Dataset directory; used to choose the first BMP if --image is omitted")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "outputs" / "localization")
    args = parser.parse_args()
    image_path = args.image
    if image_path is None:
        image_path = next(iter(sorted(args.data_dir.rglob("*.bmp"))), None)
        if image_path is None:
            parser.error("No BMP images found. Set --image or --data-dir.")
    config = LocalizationConfig()
    try:
        image = load_image(image_path)
        result = localize_iris(image, config)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Localization failed: {error}\n")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = args.output_dir / f"{image_path.stem}_localization.png"
    json_path = args.output_dir / f"{image_path.stem}_localization.json"
    save_overlay(image, result, preview_path)
    payload = {"source_image": str(image_path.resolve()), "image_shape": list(image.shape),
               "parameters": asdict(config), **result.to_dict()}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(result.to_dict(), indent=2))
    print(f"Preview: {preview_path}\nParameters: {json_path}")


if __name__ == "__main__":
    main()
