from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from ximea import xiapi

CALIBRATION_JSON = Path(__file__).resolve().parent / "second calibration" / "camera_calibration" / "camera_parameters.json"
PREVIEW_SCALE = 0.35
WINDOW_NAME = "Harris Corner Detection"


def load_calibration(json_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not json_path.exists():
        raise FileNotFoundError(f"calibracne data nenajdene: {json_path}")
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
    x, y = max(0, x), max(0, y)
    x2 = min(image.shape[1], x + w)
    y2 = min(image.shape[0], y + h)
    if x2 <= x or y2 <= y:
        return image
    return image[y:y2, x:x2]


def convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode='reflect')
    shape = (img.shape[0], img.shape[1], kh, kw)
    strides = padded.strides + padded.strides
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return np.einsum('ijkl,kl->ij', windows, kernel)


def gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return k / k.sum()


def harris_response(gray: np.ndarray, k: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    smooth = convolve2d(gray, gaussian_kernel(5, 1.0))
    Ix = convolve2d(smooth, Kx)
    Iy = convolve2d(smooth, Ky)

    g = gaussian_kernel(3, 1.0)
    Ixx = convolve2d(Ix * Ix, g)
    Ixy = convolve2d(Ix * Iy, g)
    Iyy = convolve2d(Iy * Iy, g)

    det   = Ixx * Iyy - Ixy**2
    trace = Ixx + Iyy
    R = det - k * trace**2
    return R, smooth  # vraciame aj smooth na odhad sumu


def draw_corners(frame_bgr: np.ndarray, R: np.ndarray, threshold_ratio: float) -> tuple[np.ndarray, int]:
    result = frame_bgr.copy()
    ys, xs = np.where(R > threshold_ratio * R.max())
    for y, x in zip(ys, xs):
        cv2.circle(result, (x, y), 3, (0, 0, 255), -1)
    return result, len(xs)


def compute_stats(gray: np.ndarray, smooth: np.ndarray) -> dict:
    brightness = float(np.mean(gray))
    noise = float(np.std(gray - smooth))
    return {
        "jas": round(brightness, 1),
        "sum":      round(noise, 3),
        "tmavy":       brightness < 60,
    }


def put_stats(img: np.ndarray, title: str, corners: int, stats: dict, color=(0, 255, 0)) -> None:
    lines = [
        title,
        f"rohy: {corners}",
        f"jas: {stats['brightness']:.1f}",
        f"sum cca: {stats['noise']:.2f}",
        f"tmavy: {'YES' if stats['dark'] else 'no'}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(img, line, (15, 35 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def nothing(_): pass


def main() -> None:
    camera_matrix, dist_coeffs = load_calibration(CALIBRATION_JSON)

    cv2.namedWindow(WINDOW_NAME)
    cv2.createTrackbar("Prah %", WINDOW_NAME, 1, 20, nothing)

    cam = xiapi.Camera()
    xi_img = xiapi.Image()
    device_opened = False
    acquisition_started = False

    try:
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
            camera_matrix, dist_coeffs, image_size, 1, image_size
        )

        print(f"rozlisenie: {width} x {height}")
        print("q - vypnut")

        cam.start_acquisition()
        acquisition_started = True

        while True:
            cam.get_image(xi_img)
            frame = xi_img.get_image_data_numpy()

            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
            undistorted = crop_to_roi(undistorted, roi)

            # grayscale manualne
            gray = (0.114 * undistorted[:, :, 0] +
                    0.587 * undistorted[:, :, 1] +
                    0.299 * undistorted[:, :, 2]).astype(np.float32)

            # threshold zo slidera (1-20 %)
            thresh_pct = max(1, cv2.getTrackbarPos("prah %", WINDOW_NAME)) / 100.0

            # manualny Harris
            R_manual, smooth = harris_response(gray)
            result_manual, cnt_manual = draw_corners(undistorted, R_manual, thresh_pct)

            # OpenCV Harris
            R_cv = cv2.cornerHarris(gray, blockSize=3, ksize=3, k=0.05)
            result_opencv = undistorted.copy()
            mask_cv = R_cv > thresh_pct * R_cv.max()
            result_opencv[mask_cv] = [0, 0, 255]
            cnt_opencv = int(np.sum(mask_cv))

            stats = compute_stats(gray, smooth)

            put_stats(result_manual, "manualny Harris", cnt_manual, stats)
            put_stats(result_opencv, "openCV Harris", cnt_opencv, stats)

            combined = np.hstack((result_manual, result_opencv))
            preview = cv2.resize(combined, None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE)

            cv2.imshow(WINDOW_NAME, preview)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    finally:
        cv2.destroyAllWindows()
        if acquisition_started:
            cam.stop_acquisition()
        if device_opened:
            cam.close_device()
        print("kamera vypnuta")


if __name__ == "__main__":
    main()