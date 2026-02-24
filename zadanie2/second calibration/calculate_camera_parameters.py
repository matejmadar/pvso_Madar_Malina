from __future__ import annotations

import json
from pathlib import Path

import cv2 as cv
import numpy as np


CHESSBOARD_SIZE = (5, 7)  # inner corners (columns, rows)
SQUARE_SIZE = 1.0  # real square size unit (e.g., mm). Keep 1.0 if unknown.
INPUT_DIR = Path(__file__).resolve().parent / "images_for_calibration"
OUTPUT_DIR = Path(__file__).resolve().parent / "camera_calibration"
CALIBRATION_CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def read_image_unicode_safe(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv.imdecode(data, cv.IMREAD_COLOR)


def collect_image_points(image_paths: list[Path]) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int], list[str]]:
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : CHESSBOARD_SIZE[0], 0 : CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    used_images: list[str] = []
    image_size: tuple[int, int] | None = None

    for image_path in image_paths:
        image = read_image_unicode_safe(image_path)
        if image is None:
            print(f"Skipping unreadable image: {image_path.name}")
            continue

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])

        found, corners = cv.findChessboardCornersSB(gray, CHESSBOARD_SIZE)
        if not found:
            found, corners = cv.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        if not found:
            print(f"Pattern not found: {image_path.name}")
            continue

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), CALIBRATION_CRITERIA)
        objpoints.append(objp.copy())
        imgpoints.append(corners2)
        used_images.append(image_path.name)
        print(f"Pattern found: {image_path.name}")

    if image_size is None:
        raise RuntimeError("No readable images found.")

    return objpoints, imgpoints, image_size, used_images


def compute_reprojection_error(
    objpoints: list[np.ndarray],
    imgpoints: list[np.ndarray],
    rvecs: tuple[np.ndarray, ...],
    tvecs: tuple[np.ndarray, ...],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    total_error = 0.0
    total_points = 0

    for i, objp in enumerate(objpoints):
        projected, _ = cv.projectPoints(objp, rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv.norm(imgpoints[i], projected, cv.NORM_L2)
        total_error += error * error
        total_points += len(projected)

    return float(np.sqrt(total_error / total_points)) if total_points else float("nan")


def main() -> None:
    image_paths = sorted(INPUT_DIR.glob("*.png")) + sorted(INPUT_DIR.glob("*.jpg")) + sorted(INPUT_DIR.glob("*.jpeg"))
    if not image_paths:
        raise FileNotFoundError(f"No calibration images found in: {INPUT_DIR}")

    print(f"Found {len(image_paths)} candidate images.")
    objpoints, imgpoints, image_size, used_images = collect_image_points(image_paths)

    if len(objpoints) < 3:
        raise RuntimeError(
            "Not enough valid images for calibration. At least 3 images with detected chessboard are required."
        )

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )

    reprojection_error = compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = OUTPUT_DIR / "camera_parameters.npz"
    json_path = OUTPUT_DIR / "camera_parameters.json"

    np.savez(
        npz_path,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_width=image_size[0],
        image_height=image_size[1],
        chessboard_cols=CHESSBOARD_SIZE[0],
        chessboard_rows=CHESSBOARD_SIZE[1],
        square_size=SQUARE_SIZE,
        rms=rms,
        reprojection_error=reprojection_error,
        used_images=np.array(used_images),
    )

    payload = {
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "chessboard_size": {"cols": CHESSBOARD_SIZE[0], "rows": CHESSBOARD_SIZE[1]},
        "square_size": SQUARE_SIZE,
        "rms": float(rms),
        "reprojection_error": reprojection_error,
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "used_images": used_images,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nCalibration completed.")
    print(f"Valid images used: {len(used_images)}")
    print(f"RMS error: {rms:.6f}")
    print(f"Mean reprojection error: {reprojection_error:.6f}")
    print(f"Saved: {npz_path.name}")
    print(f"Saved: {json_path.name}")


if __name__ == "__main__":
    main()