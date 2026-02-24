from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from ximea import xiapi


CALIBRATION_JSON = Path(__file__).resolve().parent / "camera_calibration" / "camera_parameters.json"
PREVIEW_SCALE = 0.30
SNAPSHOT_DIR = Path(__file__).resolve().parent / "undistortion_snapshots"


def load_calibration(json_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not json_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {json_path}")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)

    if dist_coeffs.ndim == 1:
        dist_coeffs = dist_coeffs.reshape(1, -1)

    return camera_matrix, dist_coeffs


def save_image_unicode_safe(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image before saving: {path.name}")
    path.write_bytes(encoded.tobytes())


def crop_to_roi(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        return image

    x = max(0, x)
    y = max(0, y)
    x2 = min(image.shape[1], x + w)
    y2 = min(image.shape[0], y + h)

    if x2 <= x or y2 <= y:
        return image

    return image[y:y2, x:x2]


def main() -> None:
    camera_matrix, dist_coeffs = load_calibration(CALIBRATION_JSON)

    cam = xiapi.Camera()
    xi_img = xiapi.Image()
    device_opened = False
    acquisition_started = False
    shot_idx = 1

    try:
        print("Opening first XIMEA camera...")
        cam.open_device()
        device_opened = True

        cam.set_exposure(50000)
        cam.set_param("imgdataformat", "XI_RGB32")
        cam.set_param("auto_wb", 1)

        max_width = cam.get_width_maximum()
        max_height = cam.get_height_maximum()
        cam.set_offsetX(0)
        cam.set_offsetY(0)
        cam.set_width(max_width)
        cam.set_height(max_height)

        width = cam.get_width()
        height = cam.get_height()
        image_size = (width, height)

        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, image_size, 1, image_size)

        print(f"Using camera resolution: {width} x {height}")
        print(f"Undistort crop ROI: {roi}")
        print("Press SPACE to save snapshot, Q to quit.")

        cam.start_acquisition()
        acquisition_started = True

        while True:
            cam.get_image(xi_img)
            frame = xi_img.get_image_data_numpy()

            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
            original_cropped = crop_to_roi(frame, roi)
            undistorted_cropped = crop_to_roi(undistorted, roi)

            combined = np.hstack((original_cropped, undistorted_cropped))
            cv2.putText(combined, "Original", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(
                combined,
                "Undistorted",
                (combined.shape[1] // 2 + 20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.6,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )

            preview = cv2.resize(combined, None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE)
            cv2.imshow("XIMEA: Original vs Undistorted", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                output_path = SNAPSHOT_DIR / f"original_vs_undistorted_{shot_idx:03d}.png"
                save_image_unicode_safe(output_path, combined)
                print(f"Saved snapshot: {output_path.name}")
                shot_idx += 1

    finally:
        cv2.destroyAllWindows()
        if acquisition_started:
            cam.stop_acquisition()
        if device_opened:
            cam.close_device()
        print("Camera closed.")


if __name__ == "__main__":
    main()