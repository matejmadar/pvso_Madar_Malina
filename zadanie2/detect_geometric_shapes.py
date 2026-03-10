from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from ximea import xiapi


CALIBRATION_JSON = Path(__file__).resolve().parent / "second calibration" / "camera_calibration" / "camera_parameters.json"
MIN_CONTOUR_AREA = 1200
PREVIEW_SCALE = 0.35

# Hough circle parameters (tune for your scene if needed)
HOUGH_DP = 1.2
HOUGH_MIN_DIST = 70
HOUGH_PARAM1 = 120
HOUGH_PARAM2 = 35
HOUGH_MIN_RADIUS = 20
HOUGH_MAX_RADIUS = 0


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


def classify_polygon(contour: np.ndarray) -> str | None:
    area = cv2.contourArea(contour)
    if area < MIN_CONTOUR_AREA:
        return None

    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return None

    # Ignore near-circular contours, circles are handled by Hough transform.
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    if circularity > 0.82:
        return None

    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    vertices = len(approx)

    if vertices == 3:
        return "triangle"

    if vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        if h == 0:
            return None
        aspect_ratio = w / float(h)
        if 0.92 <= aspect_ratio <= 1.08:
            return "square"
        return "rectangle"

    return None


def draw_label_with_center(frame: np.ndarray, label: str, cx: int, cy: int, color: tuple[int, int, int]) -> None:
    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
    cv2.putText(
        frame,
        label,
        (cx - 55, cy - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        color,
        2,
        cv2.LINE_AA,
    )


def detect_and_draw_circles(frame: np.ndarray, gray_blurred: np.ndarray) -> None:
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_RADIUS,
        maxRadius=HOUGH_MAX_RADIUS,
    )

    if circles is None:
        return

    circles = np.round(circles[0, :]).astype(int)
    for x, y, r in circles:
        cv2.circle(frame, (x, y), r, (255, 100, 0), 3)
        draw_label_with_center(frame, "circle", x, y, (255, 255, 0))


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

        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, image_size, 1, image_size)

        print(f"Using maximum resolution: {width} x {height}")
        print(f"Using calibration from: {CALIBRATION_JSON}")
        print("Pipeline: undistortion -> circle detection (Hough) -> polygon detection (contours)")
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

            gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 1.2)

            # 1) Circle detection by Hough transform
            detect_and_draw_circles(undistorted, blurred)

            # 2) Polygon detection by contour approximation
            edges = cv2.Canny(blurred, 50, 150)
            edges = cv2.dilate(edges, None, iterations=1)
            edges = cv2.erode(edges, None, iterations=1)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                shape_name = classify_polygon(contour)
                if shape_name is None:
                    continue

                moments = cv2.moments(contour)
                if moments["m00"] == 0:
                    continue

                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])

                cv2.drawContours(undistorted, [contour], -1, (0, 255, 0), 3)
                draw_label_with_center(undistorted, shape_name, cx, cy, (0, 255, 255))

            preview = cv2.resize(undistorted, None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE)
            cv2.imshow("XIMEA Undistorted Shape Detection", preview)

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
