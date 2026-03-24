from __future__ import annotations

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = "test.jpg"   # zmen na svoju cestu


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
    return R, smooth


def draw_corners_bgr(frame: np.ndarray, R: np.ndarray, thresh: float) -> tuple[np.ndarray, int]:
    result = frame.copy()
    ys, xs = np.where(R > thresh * R.max())
    for y, x in zip(ys, xs):
        cv2.circle(result, (x, y), 3, (0, 0, 255), -1)
    return result, len(xs)


def compute_stats(gray: np.ndarray, smooth: np.ndarray) -> dict:
    brightness = float(np.mean(gray))
    noise = float(np.std(gray - smooth))
    return {
        "brightness": round(brightness, 2),
        "noise_est":  round(noise, 3),
        "dark":       brightness < 60,
    }


def main() -> None:
    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        sys.exit(f"[ERROR] Subor '{IMAGE_PATH}' nebol najdeny.")

    # grayscale manualne
    gray = (0.114 * img_bgr[:, :, 0] +
            0.587 * img_bgr[:, :, 1] +
            0.299 * img_bgr[:, :, 2]).astype(np.float32)

    THRESH = 0.01   # 1 % z maxima – zmen podla potreby

    # manuálny Harris
    R_manual, smooth = harris_response(gray)
    result_manual, cnt_manual = draw_corners_bgr(img_bgr, R_manual, THRESH)
    stats = compute_stats(gray, smooth)

    # OpenCV Harris
    R_cv = cv2.cornerHarris(gray, blockSize=3, ksize=3, k=0.05)
    result_opencv = img_bgr.copy()
    mask_cv = R_cv > THRESH * R_cv.max()
    result_opencv[mask_cv] = [0, 0, 255]
    cnt_opencv = int(np.sum(mask_cv))

    # výpis štatistík
    print(f"Jas:   {stats['brightness']}")
    print(f"Sum cca:   {stats['noise_est']}")
    print(f"Tmavy frame:   {stats['dark']}")
    print(f"Manualne rohy:  {cnt_manual}")
    print(f"OpenCV rohy:  {cnt_opencv}")

    # vizualizácia
    orig_rgb    = cv2.cvtColor(img_bgr,      cv2.COLOR_BGR2RGB)
    manual_rgb  = cv2.cvtColor(result_manual, cv2.COLOR_BGR2RGB)
    opencv_rgb  = cv2.cvtColor(result_opencv, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, img, title in zip(axes,
                               [orig_rgb, manual_rgb, opencv_rgb],
                               ["Originál",
                                f"Manualny Harris  (rohov: {cnt_manual})",
                                f"OpenCV Harris  (rohov: {cnt_opencv})"]):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    plt.suptitle(
        f"Prah: {int(THRESH*100)} %  |  Jas: {stats['brightness']}  "
        f"|  Sum: {stats['noise_est']}  |  Tmavy: {stats['dark']}",
        fontsize=10, color='gray'
    )
    plt.tight_layout()
    plt.savefig("harris_porovnanie.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()