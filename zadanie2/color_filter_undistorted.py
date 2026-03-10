from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from ximea import xiapi


CALIBRATION_JSON = Path(__file__).resolve().parent / "second calibration" / "camera_calibration" / "camera_parameters.json"
PREVIEW_SCALE = 0.35

# Source color range in HSV (two ranges because hue wraps around)
LOWER_SOURCE_1 = np.array([0, 90, 70], dtype=np.uint8)
UPPER_SOURCE_1 = np.array([10, 255, 255], dtype=np.uint8)
LOWER_SOURCE_2 = np.array([170, 90, 70], dtype=np.uint8)
UPPER_SOURCE_2 = np.array([180, 255, 255], dtype=np.uint8)

# Replacement color range in HSV (blue range)
LOWER_REPLACEMENT_HSV = np.array([100, 120, 80], dtype=np.uint8)
UPPER_REPLACEMENT_HSV = np.array([130, 255, 255], dtype=np.uint8)


def load_calibration(json_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not json_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {json_path}")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)

    if dist_coeffs.ndim == 1:
        dist_coeffs = dist_coeffs.reshape(1, -1)

    return camera_matrix, dist_coeffs


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


def replace_source_with_blue_range(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, LOWER_SOURCE_1, UPPER_SOURCE_1)
    mask2 = cv2.inRange(hsv, LOWER_SOURCE_2, UPPER_SOURCE_2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Remove isolated noise and fill small gaps.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Use midpoint color of replacement HSV range so resulting color is guaranteed in-range.
    replacement_hsv = ((LOWER_REPLACEMENT_HSV.astype(np.uint16) + UPPER_REPLACEMENT_HSV.astype(np.uint16)) // 2).astype(
        np.uint8
    )
    replacement_bgr = cv2.cvtColor(replacement_hsv.reshape(1, 1, 3), cv2.COLOR_HSV2BGR).reshape(3)

    result = frame_bgr.copy()
    result[mask > 0] = replacement_bgr
    return result, mask


def main() -> None:
    camera_matrix, dist_coeffs = load_calibration(CALIBRATION_JSON)

    cam = xiapi.Camera()
    xi_img = xiapi.Image()
    device_opened = False
    acquisition_started = False

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

        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            dist_coeffs,
            image_size,
            1,
            image_size,
        )

        print(f"Using camera resolution: {width} x {height}")
        print(f"Using calibration from: {CALIBRATION_JSON}")
        print("Color filter: source range -> blue range")
        print("Press Q to quit.")

        cam.start_acquisition()
        acquisition_started = True

        while True:
            cam.get_image(xi_img)
            frame = xi_img.get_image_data_numpy()

            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
            undistorted = crop_to_roi(undistorted, roi)

            filtered, mask = replace_source_with_blue_range(undistorted)

            # Side-by-side preview: mask stream | filtered stream
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            combined = np.hstack((mask_bgr, filtered))
            cv2.putText(
                combined,
                "Mask (source range)",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                combined,
                "Color filter (source -> blue range)",
                (combined.shape[1] // 2 + 20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            preview = cv2.resize(combined, None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE)

            cv2.imshow("XIMEA Undistorted Color Filter", preview)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    finally:
        cv2.destroyAllWindows()
        if acquisition_started:
            cam.stop_acquisition()
        if device_opened:
            cam.close_device()
        print("Camera closed.")


if __name__ == "__main__":
    main()
