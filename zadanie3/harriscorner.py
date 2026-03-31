import cv2
import numpy as np

IMAGE_PATH = "test6.jpg"
THRESH = 0.01  # 1 % z maxima


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
    smooth = convolve2d(gray, gaussian_kernel(4, 0.45))
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
    # zachova len lokalne maximuma v okoli min_dist x min_dist
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


img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Subor '{IMAGE_PATH}' nebol najdeny.")

gray = (0.114 * img[:, :, 0] +
        0.587 * img[:, :, 1] +
        0.299 * img[:, :, 2]).astype(np.float32)

R_manual, smooth = harris_response(gray)
result_manual, cnt_manual = draw_corners(img, R_manual, THRESH)

R_cv = cv2.cornerHarris(gray, blockSize=3, ksize=3, k=0.05)
result_opencv, cnt_opencv = draw_corners(img, R_cv, THRESH)

brightness = round(float(np.mean(gray)), 2)
noise      = round(float(np.std(gray - smooth)), 3)
dark       = brightness < 60

print(f"Manual corners: {cnt_manual}")
print(f"OpenCV corners: {cnt_opencv}")
print(f"Brightness:     {brightness}")
print(f"Noise est.:     {noise}")
print(f"Dark:           {dark}")

def add_label(img, text):
    out = img.copy()
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    return out

panel_orig   = add_label(img, "Original")
panel_manual = add_label(result_manual, f"Manual Harris  corners:{cnt_manual}")
panel_opencv = add_label(result_opencv, f"OpenCV Harris  corners:{cnt_opencv}")

stats_line = f"Brightness:{brightness}  Noise:{noise}  Dark:{'YES' if dark else 'no'}"
combined = np.hstack((panel_orig, panel_manual, panel_opencv))
cv2.putText(combined, stats_line, (10, combined.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)

cv2.imwrite("harris_porovnanie.png", combined)
max_width = 1900
if combined.shape[1] > max_width:
    scale = max_width / combined.shape[1]
    preview = cv2.resize(combined, None, fx=scale, fy=scale)
else:
    preview = combined
cv2.imshow("Original | Manual | OpenCV", preview)
cv2.waitKey(0)
cv2.destroyAllWindows()