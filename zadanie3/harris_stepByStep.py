from pathlib import Path

import cv2
import numpy as np


PROCESS_RESIZE_RATIO = 1.0
UPDATE_EVERY_ROWS = 100
UPDATE_DELAY_MS = 1
SCREEN_W, SCREEN_H = 1900, 900
WINDOW_NAME = "Convolution Build (2x2)"
HARRIS_K = 0.04
THRESHOLD_RATIO = 0.01
NMS_RADIUS = 1
OUTPUT_DIR = Path(__file__).resolve().parent / "harris_outputs"

_REF_H = 300


def _font_scale(img):
    h = img.shape[0]
    ratio = max(h / _REF_H, 1.0)
    scale = 0.55 * ratio
    thickness = max(2, int(2 * ratio))
    y1 = int(35 * ratio)
    y2 = int(70 * ratio)
    x = int(15 * ratio)
    return scale, thickness, x, y1, y2


def normalize_for_display(arr):
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-9:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255.0).astype(np.uint8)


def fit_to_screen(image, screen_w, screen_h, margin=120):
    h, w = image.shape[:2]
    max_w = max(200, screen_w - margin)
    max_h = max(200, screen_h - margin)
    scale = min(max_w / w, max_h / h, 1.0)
    if scale >= 1.0:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def label(img, text1, text2=""):
    out = img.copy()
    sc, th, x, y1, y2 = _font_scale(out)
    cv2.putText(out, text1, (x, y1), cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 255, 0), th, cv2.LINE_AA)
    if text2:
        cv2.putText(out, text2, (x, y2), cv2.FONT_HERSHEY_SIMPLEX, sc * 0.8, (255, 255, 0), th, cv2.LINE_AA)
    return out


def convolve2d(image, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))
    output = np.einsum("ijkl,kl->ij", windows, kernel, optimize=True)
    return output.astype(np.float32, copy=False)


def to_bgr(arr):
    return cv2.cvtColor(normalize_for_display(arr), cv2.COLOR_GRAY2BGR)


def show_2x2(title, p1, p2, p3, p4):
    canvas = np.vstack((np.hstack((p1, p2)), np.hstack((p3, p4))))
    canvas = fit_to_screen(canvas, SCREEN_W, SCREEN_H)
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(title, canvas)
    cv2.waitKey(1)
    return canvas


def nonmax_suppression(response, threshold, radius=1):
    h, w = response.shape
    corners = np.zeros((h, w), dtype=np.uint8)

    for y in range(radius, h - radius):
        for x in range(radius, w - radius):
            val = response[y, x]
            if val <= threshold:
                continue
            region = response[y - radius : y + radius + 1, x - radius : x + radius + 1]
            if val == np.max(region):
                corners[y, x] = 255

    return corners


def save_image_unicode_safe(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    path.write_bytes(encoded.tobytes())


def main():
    image_path = Path(__file__).resolve().parent / "20260321_182138.jpg"
    snapshots = {}

    data = np.fromfile(str(image_path), dtype=np.uint8)
    image_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    image_bgr = cv2.resize(
        image_bgr,
        None,
        fx=PROCESS_RESIZE_RATIO,
        fy=PROCESS_RESIZE_RATIO,
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    print("[STEP] 1/5 Manual gradients (Sobel kernels)")
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]], dtype=np.float32)

    h, w = gray.shape
    gx_full = convolve2d(gray, kx)
    gy_full = convolve2d(gray, ky)
    gx = np.zeros_like(gx_full, dtype=np.float32)
    gy = np.zeros_like(gy_full, dtype=np.float32)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    for y in range(h):
        gx[y, :] = gx_full[y, :]
        gy[y, :] = gy_full[y, :]

        if (y % UPDATE_EVERY_ROWS == 0) or (y == h - 1):
            mag = np.sqrt(gx * gx + gy * gy)
            rows_text = f"Rows: {y + 1}/{h}"

            p1 = label(image_bgr, "1) Original")
            p2 = label(to_bgr(gx), "2) Sobel X", rows_text)
            p3 = label(to_bgr(gy), "3) Sobel Y", rows_text)
            p4 = label(to_bgr(mag), "4) Gradient magnitude", rows_text)

            canvas = np.vstack((np.hstack((p1, p4)), np.hstack((p2, p3))))
            canvas = fit_to_screen(canvas, SCREEN_W, SCREEN_H)
            cv2.imshow(WINDOW_NAME, canvas)
            snapshots["01_progressive_convolution"] = canvas.copy()

            if (cv2.waitKey(UPDATE_DELAY_MS) & 0xFF) == ord("q"):
                break

    print("[STEP] 2/5 Structure tensor products: Ixx, Iyy, Ixy")
    ixx = gx * gx
    iyy = gy * gy
    ixy = gx * gy

    snapshots["02_products"] = show_2x2(
        "Harris - Products",
        label(to_bgr(ixy), "Ixy = Ix*Iy"),
        label(to_bgr(np.sqrt(gx * gx + gy * gy)), "Gradient magnitude"),
        label(to_bgr(ixx), "Ixx = Ix*Ix"),
        label(to_bgr(iyy), "Iyy = Iy*Iy")
    )

    print("[STEP] 3/5 Smoothing tensor terms (manual convolution)")
    gauss = np.array([[1, 2, 1],
                      [2, 4, 2],
                      [1, 2, 1]], dtype=np.float32) / 16.0
    sxx = convolve2d(ixx, gauss)
    syy = convolve2d(iyy, gauss)
    sxy = convolve2d(ixy, gauss)

    snapshots["03_smoothed_tensor"] = show_2x2(
        "Harris - Smoothed Tensor",
        label(to_bgr(sxy), "Sxy"),
        label(to_bgr(sxx + syy), "Trace = Sxx + Syy"),
        label(to_bgr(sxx), "Sxx"),
        label(to_bgr(syy), "Syy")
    )

    print("[STEP] 4/5 Harris response R = det(M) - k*trace(M)^2")
    det_m = sxx * syy - sxy * sxy
    trace_m = sxx + syy
    response = det_m - HARRIS_K * (trace_m * trace_m)

    r_max = float(np.max(response))
    threshold = THRESHOLD_RATIO * r_max

    snapshots["04_response"] = show_2x2(
        "Harris - Response",
        label(to_bgr(det_m), "det(M)"),
        label(to_bgr(trace_m), "trace(M)"),
        label(to_bgr(response), "R response"),
        label(to_bgr(np.where(response > threshold, response, 0)), f"R > {THRESHOLD_RATIO:.2f}*Rmax"),
    )

    print("[STEP] 5/5 Non-maximum suppression + final corners")
    corners_mask = nonmax_suppression(response, threshold, radius=NMS_RADIUS)
    corner_points = np.column_stack(np.where(corners_mask > 0))

    final = image_bgr.copy()
    for y, x in corner_points:
        cv2.circle(final, (int(x), int(y)), 2, (0, 0, 255), -1)

    opencv_response = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=HARRIS_K)
    opencv_threshold = THRESHOLD_RATIO * float(np.max(opencv_response))
    opencv_mask = nonmax_suppression(opencv_response, opencv_threshold, radius=NMS_RADIUS)
    opencv_points = np.column_stack(np.where(opencv_mask > 0))

    opencv_final = image_bgr.copy()
    for y, x in opencv_points:
        cv2.circle(opencv_final, (int(x), int(y)), 2, (255, 0, 0), -1)

    print(f"[INFO] Manual corners detected: {len(corner_points)}")
    print(f"[INFO] OpenCV corners detected: {len(opencv_points)}")

    snapshots["05_final"] = show_2x2(
        "Harris - Final",
        label(image_bgr, "Original"),
        label(final, f"Manual corners: {len(corner_points)}"),
        label(cv2.cvtColor(corners_mask, cv2.COLOR_GRAY2BGR), "NMS corners mask"),
        label(opencv_final, f"OpenCV Harris corners: {len(opencv_points)}")
    )

    cv2.waitKey(0)

    for name, img in snapshots.items():
        out_path = OUTPUT_DIR / f"{name}.png"
        save_image_unicode_safe(out_path, img)
    print(f"[INFO] Saved {len(snapshots)} window images to: {OUTPUT_DIR}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()