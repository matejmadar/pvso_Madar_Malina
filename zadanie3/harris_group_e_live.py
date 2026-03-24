from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from ximea import xiapi


# Processing scale keeps real-time speed on higher camera resolutions.
PROCESS_SCALE = 0.5

# Harris parameters.
BLOCK_SIZE = 3
SOBEL_KSIZE = 3
HARRIS_K = 0.04
THRESHOLD_REL = 0.01

# Stability evaluation settings.
LIGHT_FACTOR = 0.65
NOISE_SIGMA = 8.0
STABILITY_UPDATE_EVERY = 10
WINDOW_TITLE = "Group E - Harris corners (manual vs OpenCV)"
THRESHOLD_TRACKBAR = "Threshold x1e4"
THRESHOLD_MIN = 1e-4
THRESHOLD_MAX = 0.2
THRESHOLD_TRACKBAR_MAX = 5000
CAMERA_WIDTH = 2464
CAMERA_HEIGHT = 2056
CALIBRATION_JSON_CANDIDATES = [
    Path(__file__).resolve().parent.parent / "zadanie2" / "second calibration" / "camera_calibration" / "camera_parameters.json",
    Path(__file__).resolve().parent.parent / "zadanie2" / "second_calibration" / "camera_calibration" / "camera_parameters.json",
]


def prepare_gray(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, None, fx=PROCESS_SCALE, fy=PROCESS_SCALE, interpolation=cv2.INTER_AREA)
    return gray, small


def find_calibration_file() -> Path:
    for p in CALIBRATION_JSON_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("Calibration JSON not found in expected zadanie2/second calibration paths.")


def load_calibration(json_path: Path) -> tuple[np.ndarray, np.ndarray]:
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


def harris_manual_response(gray_u8: np.ndarray, block_size: int, sobel_ksize: int, k: float) -> np.ndarray:
    gray = gray_u8.astype(np.float32) / 255.0

    ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=sobel_ksize)
    iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=sobel_ksize)

    ixx = ix * ix
    iyy = iy * iy
    ixy = ix * iy

    sxx = cv2.GaussianBlur(ixx, (block_size, block_size), 0)
    syy = cv2.GaussianBlur(iyy, (block_size, block_size), 0)
    sxy = cv2.GaussianBlur(ixy, (block_size, block_size), 0)

    det_m = (sxx * syy) - (sxy * sxy)
    trace_m = sxx + syy
    response = det_m - k * (trace_m * trace_m)
    return response


def harris_opencv_response(gray_u8: np.ndarray, block_size: int, sobel_ksize: int, k: float) -> np.ndarray:
    gray = gray_u8.astype(np.float32) / 255.0
    return cv2.cornerHarris(gray, block_size, sobel_ksize, k)


def corners_from_response(response: np.ndarray, threshold_rel: float) -> np.ndarray:
    max_val = float(response.max())
    if max_val <= 0:
        return np.empty((0, 2), dtype=np.int32)

    threshold = threshold_rel * max_val
    strong = response > threshold
    local_max = response == cv2.dilate(response, np.ones((3, 3), np.uint8))
    mask = strong & local_max
    yx = np.argwhere(mask)
    if yx.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    return yx[:, ::-1].astype(np.int32)


def detect_corners_manual(gray_small: np.ndarray, threshold_rel: float) -> np.ndarray:
    r = harris_manual_response(gray_small, BLOCK_SIZE, SOBEL_KSIZE, HARRIS_K)
    return corners_from_response(r, threshold_rel)


def detect_corners_opencv(gray_small: np.ndarray, threshold_rel: float) -> np.ndarray:
    r = harris_opencv_response(gray_small, BLOCK_SIZE, SOBEL_KSIZE, HARRIS_K)
    return corners_from_response(r, threshold_rel)


def upscale_points(points_small: np.ndarray) -> np.ndarray:
    if points_small.size == 0:
        return points_small
    scale = 1.0 / PROCESS_SCALE
    return np.round(points_small.astype(np.float32) * scale).astype(np.int32)


def draw_points(frame: np.ndarray, points: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = frame.copy()
    for x, y in points:
        cv2.circle(out, (int(x), int(y)), 2, color, -1, lineType=cv2.LINE_AA)
    return out


def make_dark(gray: np.ndarray) -> np.ndarray:
    return np.clip(gray.astype(np.float32) * LIGHT_FACTOR, 0, 255).astype(np.uint8)


def make_noisy(gray: np.ndarray) -> np.ndarray:
    noise = np.random.normal(0.0, NOISE_SIGMA, gray.shape).astype(np.float32)
    noisy = gray.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def count_for_both(gray_small: np.ndarray, threshold_rel: float) -> tuple[int, int]:
    c_manual = detect_corners_manual(gray_small, threshold_rel).shape[0]
    c_cv = detect_corners_opencv(gray_small, threshold_rel).shape[0]
    return c_manual, c_cv


def draw_text_styled(
    img: np.ndarray, text: str, x: int, y: int, font_scale: float, thickness: int, color: tuple[int, int, int]
) -> None:
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def get_screen_size() -> tuple[int, int]:
    try:
        import ctypes

        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:
        return 1920, 1080


def fit_to_screen(image: np.ndarray, margin: int = 120) -> np.ndarray:
    screen_w, screen_h = get_screen_size()
    target_w = max(640, screen_w - margin)
    target_h = max(480, screen_h - margin)

    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h, 1.0)
    if scale >= 0.999:
        return image

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def draw_preview_hud(
    preview: np.ndarray,
    manual_count: int,
    cv_count: int,
    threshold_rel: float,
    m_dark: int,
    c_dark: int,
    m_noise: int,
    c_noise: int,
) -> None:
    h, w = preview.shape[:2]
    half_w = w // 2

    # Full-width top panel for consistent contrast on any scene.
    panel_h = max(130, int(0.2 * h))
    overlay = preview.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.7, preview, 0.3, 0, dst=preview)

    # Larger titles for each stream (drawn on final preview, not source frame).
    title_scale = max(0.8, h / 950.0)
    title_thickness = max(2, int(round(h / 420.0)))
    draw_text_styled(preview, "Manual Harris", 16, 38, title_scale, title_thickness, (0, 70, 255))
    draw_text_styled(preview, "OpenCV cornerHarris", half_w + 16, 38, title_scale, title_thickness, (0, 255, 255))

    text_scale = max(0.72, h / 1200.0)
    text_thickness = max(2, int(round(h / 650.0)))
    line_h = max(26, int(round(34 * text_scale)))
    y = 72

    # Left (manual) metadata.
    draw_text_styled(preview, f"corners: {manual_count}", 16, y, text_scale, text_thickness, (0, 70, 255))
    draw_text_styled(preview, f"dark: {m_dark}", 16, y + line_h, text_scale, text_thickness, (0, 70, 255))
    draw_text_styled(preview, f"noise: {m_noise}", 16, y + 2 * line_h, text_scale, text_thickness, (0, 70, 255))

    # Right (opencv) metadata.
    draw_text_styled(preview, f"corners: {cv_count}", half_w + 16, y, text_scale, text_thickness, (0, 255, 255))
    draw_text_styled(preview, f"dark: {c_dark}", half_w + 16, y + line_h, text_scale, text_thickness, (0, 255, 255))
    draw_text_styled(preview, f"noise: {c_noise}", half_w + 16, y + 2 * line_h, text_scale, text_thickness, (0, 255, 255))

    # Shared parameter shown once in center.
    center_x = max(16, half_w - 120)
    draw_text_styled(preview, f"threshold_rel={threshold_rel:.4f}", center_x, y + 3 * line_h, text_scale, text_thickness, (90, 255, 90))


def threshold_to_trackbar(v: float) -> int:
    v = float(np.clip(v, THRESHOLD_MIN, THRESHOLD_MAX))
    ratio = (v - THRESHOLD_MIN) / (THRESHOLD_MAX - THRESHOLD_MIN)
    return int(round(ratio * THRESHOLD_TRACKBAR_MAX))


def trackbar_to_threshold(pos: int) -> float:
    p = int(np.clip(pos, 0, THRESHOLD_TRACKBAR_MAX))
    ratio = p / float(THRESHOLD_TRACKBAR_MAX)
    return THRESHOLD_MIN + ratio * (THRESHOLD_MAX - THRESHOLD_MIN)


def main() -> None:
    cam = xiapi.Camera()
    xi_img = xiapi.Image()
    opened = False
    started = False
    calibration_path: Path | None = None
    camera_matrix: np.ndarray | None = None
    dist_coeffs: np.ndarray | None = None
    new_camera_matrix: np.ndarray | None = None
    roi: tuple[int, int, int, int] | None = None

    threshold_rel = THRESHOLD_REL
    frame_idx = 0
    m_dark = 0
    c_dark = 0
    m_noise = 0
    c_noise = 0

    try:
        calibration_path = find_calibration_file()
        camera_matrix, dist_coeffs = load_calibration(calibration_path)
        print(f"Using calibration: {calibration_path}")

        print("Opening first XIMEA camera...")
        cam.open_device()
        opened = True

        cam.set_exposure(20000)
        cam.set_param("imgdataformat", "XI_RGB32")
        cam.set_param("auto_wb", 1)
        cam.set_offsetX(0)
        cam.set_offsetY(0)
        cam.set_width(CAMERA_WIDTH)
        cam.set_height(CAMERA_HEIGHT)

        print(f"Using camera resolution: {cam.get_width()} x {cam.get_height()}")
        image_size = (cam.get_width(), cam.get_height())
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            dist_coeffs,
            image_size,
            1.0,
            image_size,
        )

        print("Starting acquisition...")
        cam.start_acquisition()
        started = True

        print("Keys: q=quit, +=higher threshold, -=lower threshold")
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.createTrackbar(THRESHOLD_TRACKBAR, WINDOW_TITLE, threshold_to_trackbar(threshold_rel), THRESHOLD_TRACKBAR_MAX, lambda _: None)
        while True:
            threshold_pos = cv2.getTrackbarPos(THRESHOLD_TRACKBAR, WINDOW_TITLE)
            threshold_rel = trackbar_to_threshold(threshold_pos)

            cam.get_image(xi_img)
            frame = xi_img.get_image_data_numpy()

            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Use undistorted frame for all streams and all metrics.
            undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
            undistorted = crop_to_roi(undistorted, roi)
            _, gray_small = prepare_gray(undistorted)

            pts_manual_small = detect_corners_manual(gray_small, threshold_rel)
            pts_cv_small = detect_corners_opencv(gray_small, threshold_rel)

            pts_manual = upscale_points(pts_manual_small)
            pts_cv = upscale_points(pts_cv_small)

            manual_vis = draw_points(undistorted, pts_manual, (0, 0, 255))
            cv_vis = draw_points(undistorted, pts_cv, (0, 255, 255))

            combined = np.hstack((manual_vis, cv_vis))

            if frame_idx % STABILITY_UPDATE_EVERY == 0:
                dark = make_dark(gray_small)
                noisy = make_noisy(gray_small)

                m_dark, c_dark = count_for_both(dark, threshold_rel)
                m_noise, c_noise = count_for_both(noisy, threshold_rel)

            preview = fit_to_screen(combined)
            draw_preview_hud(
                preview,
                pts_manual.shape[0],
                pts_cv.shape[0],
                threshold_rel,
                m_dark,
                c_dark,
                m_noise,
                c_noise,
            )

            cv2.imshow(WINDOW_TITLE, preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (ord("+"), ord("=")):
                threshold_rel = min(THRESHOLD_MAX, threshold_rel * 1.2)
                cv2.setTrackbarPos(THRESHOLD_TRACKBAR, WINDOW_TITLE, threshold_to_trackbar(threshold_rel))
            if key in (ord("-"), ord("_")):
                threshold_rel = max(THRESHOLD_MIN, threshold_rel / 1.2)
                cv2.setTrackbarPos(THRESHOLD_TRACKBAR, WINDOW_TITLE, threshold_to_trackbar(threshold_rel))

            frame_idx += 1

    finally:
        cv2.destroyAllWindows()
        if started:
            cam.stop_acquisition()
        if opened:
            cam.close_device()
        print("Camera closed.")


if __name__ == "__main__":
    main()
