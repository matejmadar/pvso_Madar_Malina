import cv2
import numpy as np
import json
from ximea import xiapi

CALIBRATION_PATH = "camera_parameters.json"
PREVIEW_SCALE = 0.35
WINDOW_NAME = "Harris Corner Detection"


def load_calibration(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    cam_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)
    if dist_coeffs.ndim == 1:
        dist_coeffs = dist_coeffs.reshape(1, -1)
    return cam_matrix, dist_coeffs


def crop_to_roi(image, roi):
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        return image
    x, y = max(0, x), max(0, y)
    x2 = min(image.shape[1], x + w)
    y2 = min(image.shape[0], y + h)
    return image[y:y2, x:x2] if x2 > x and y2 > y else image


def convolve2d(img, kernel):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode='reflect')
    shape = (img.shape[0], img.shape[1], kh, kw)
    strides = padded.strides + padded.strides
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return np.einsum('ijkl,kl->ij', windows, kernel)


def gaussian_kernel(size=5, sigma=1.0):
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return k / k.sum()


def harris_response(gray, k=0.05):
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
    return det - k * trace**2, smooth


def nms(R, min_dist=5):
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(R, size=min_dist)
    return (R == local_max) & (R > 0)


def draw_corners(frame, R, thresh):
    result = frame.copy()
    mask = nms(R) & (R > thresh * R.max())
    ys, xs = np.where(mask)
    for y, x in zip(ys, xs):
        cv2.circle(result, (x, y), 2, (0, 0, 255), -1)
    return result, len(xs)


def put_stats(img, title, corners, brightness, noise, dark, color):
    lines = [title, f"Corners: {corners}", f"Brightness: {brightness:.1f}",
             f"Noise: {noise:.3f}", f"Dark: {'YES' if dark else 'no'}"]
    for i, line in enumerate(lines):
        cv2.putText(img, line, (15, 35 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def nothing(_): pass


cam_matrix, dist_coeffs = load_calibration(CALIBRATION_PATH)

cv2.namedWindow(WINDOW_NAME)
cv2.createTrackbar("Threshold %", WINDOW_NAME, 1, 20, nothing)

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
    cam.set_offsetX(0)
    cam.set_offsetY(0)
    cam.set_width(cam.get_width_maximum())
    cam.set_height(cam.get_height_maximum())

    w, h = cam.get_width(), cam.get_height()
    new_cam_matrix, roi = cv2.getOptimalNewCameraMatrix(
        cam_matrix, dist_coeffs, (w, h), 1, (w, h)
    )

    cam.start_acquisition()
    acquisition_started = True
    print(f"Running at {w}x{h} – press Q to quit.")

    while True:
        cam.get_image(xi_img)
        frame = xi_img.get_image_data_numpy()

        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        undistorted = cv2.undistort(frame, cam_matrix, dist_coeffs, None, new_cam_matrix)
        undistorted = crop_to_roi(undistorted, roi)

        gray = (0.114 * undistorted[:, :, 0] +
                0.587 * undistorted[:, :, 1] +
                0.299 * undistorted[:, :, 2]).astype(np.float32)

        thresh = max(1, cv2.getTrackbarPos("Threshold %", WINDOW_NAME)) / 100.0

        R_manual, smooth = harris_response(gray)
        result_manual, cnt_manual = draw_corners(undistorted, R_manual, thresh)

        R_cv = cv2.cornerHarris(gray, blockSize=3, ksize=3, k=0.05)
        result_opencv, cnt_opencv = draw_corners(undistorted, R_cv, thresh)

        brightness = float(np.mean(gray))
        noise = float(np.std(gray - smooth))
        dark = brightness < 60

        put_stats(result_manual, "Manual Harris", cnt_manual, brightness, noise, dark, (0, 200, 255))
        put_stats(result_opencv, "OpenCV Harris", cnt_opencv, brightness, noise, dark, (0, 255, 0))

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
    print("Camera closed.")