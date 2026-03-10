from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from ximea import xiapi


CALIBRATION_JSON = Path(__file__).resolve().parent / "second calibration" / "camera_calibration" / "camera_parameters.json"
PREVIEW_SCALE = 0.25

# Target color: green in HSV
GREEN_HUE_MIN: float = 35.0
GREEN_HUE_MAX: float = 85.0
GREEN_SAT_MIN: float = 50.0
GREEN_VAL_MIN: float = 50.0

# Destination hue range: red in HSV (0–10 avoids the wrap-around artefact for mapping)
RED_HUE_MIN: float = 0.0
RED_HUE_MAX: float = 10.0


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


def replace_green_with_scaled_red(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replace green pixels with red pixels whose hue is linearly scaled
    according to the original green hue position within [GREEN_HUE_MIN, GREEN_HUE_MAX].
    Saturation and value are preserved so that dark/light shades are retained."""

    hsv_f = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h = hsv_f[:, :, 0]
    s = hsv_f[:, :, 1]
    v = hsv_f[:, :, 2]

    # Build the green mask
    bool_mask = (
        (h >= GREEN_HUE_MIN) & (h <= GREEN_HUE_MAX) &
        (s >= GREEN_SAT_MIN) &
        (v >= GREEN_VAL_MIN)
    )
    mask = bool_mask.astype(np.uint8) * 255

    # Morphological cleanup – remove noise and fill gaps
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    bool_mask = mask > 0

    # Map hue: green [GREEN_HUE_MIN, GREEN_HUE_MAX] → red [RED_HUE_MIN, RED_HUE_MAX]
    # t=0 → yellowish-green maps to RED_HUE_MIN (warm red)
    # t=1 → cyan-green maps to RED_HUE_MAX (slightly cooler red)
    result_hsv = hsv_f.copy()
    green_hues = h[bool_mask]
    t = np.clip(
        (green_hues - GREEN_HUE_MIN) / (GREEN_HUE_MAX - GREEN_HUE_MIN),
        0.0, 1.0,
    )
    result_hsv[bool_mask, 0] = RED_HUE_MIN + t * (RED_HUE_MAX - RED_HUE_MIN)
    # S and V are intentionally left unchanged → shading is preserved

    result_bgr = cv2.cvtColor(
        np.clip(result_hsv, 0, 255).astype(np.uint8),
        cv2.COLOR_HSV2BGR,
    )

    output = frame_bgr.copy()
    output[bool_mask] = result_bgr[bool_mask]

    return output, mask


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
        print("Color filter: green -> red (hue-scaled)")
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

            filtered, mask = replace_green_with_scaled_red(undistorted)

            # Side-by-side preview: original undistorted | filtered
            combined = np.hstack((undistorted, filtered))
            cv2.putText(combined, "Undistorted", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(
                combined,
                "Color filter (green -> red, scaled)",
                (combined.shape[1] // 2 + 20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            preview = cv2.resize(combined, None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE)
            mask_preview = cv2.resize(mask, None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE)

            cv2.imshow("XIMEA Undistorted Color Filter", preview)
            cv2.imshow("Green Mask", mask_preview)

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