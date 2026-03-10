from __future__ import annotations

import math
import json
from pathlib import Path
import numpy as np
import cv2
from ximea import xiapi

CALIBRATION_JSON = Path(__file__).resolve().parent / "camera_parameters.json"
PREVIEW_SCALE = 0.35


def classify_shape(contour, min_area, circularity_thresh) -> str | None:
    area = cv2.contourArea(contour)
    if area < min_area:
        return None

    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return None

    approx   = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    vertices = len(approx)

    if vertices == 3:
        return "triangle"

    if vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        if h == 0:
            return None
        if 0.92 <= w / float(h) <= 1.08:
            return "square"
        return "rectangle"

    circularity = 4 * math.pi * area / (perimeter * perimeter)
    if circularity > circularity_thresh / 100:
        return "circle"

    return None


def px_to_cm(px, fx, dist_cm):
    return round((px / fx) * dist_cm, 2)


def nothing(_): pass


def load_calibration(json_path: Path):
    if not json_path.exists():
        return None, None, None
    data          = json.loads(json_path.read_text(encoding="utf-8"))
    camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    dist_coeffs   = np.array(data["dist_coeffs"],   dtype=np.float64)
    if dist_coeffs.ndim == 1:
        dist_coeffs = dist_coeffs.reshape(1, -1)
    fx = camera_matrix[0, 0]
    return camera_matrix, dist_coeffs, fx


def main() -> None:
    camera_matrix, dist_coeffs, fx = load_calibration(CALIBRATION_JSON)
    CALIB = camera_matrix is not None

    cam = xiapi.Camera()
    xi_img = xiapi.Image()

    cv2.namedWindow("Nastavenia")
    cv2.createTrackbar("Canny min",       "Nastavenia", 50,   500,  nothing)
    cv2.createTrackbar("Canny max",       "Nastavenia", 150,  500,  nothing)
    cv2.createTrackbar("Min plocha",      "Nastavenia", 1200, 5000, nothing)
    cv2.createTrackbar("Circularity",     "Nastavenia", 85,   100,  nothing)
    cv2.createTrackbar("Vzdialenost cm",  "Nastavenia", 30,   200,  nothing)

    try:
        cam.open_device()
        cam.set_exposure(50000)
        cam.set_param("imgdataformat", "XI_RGB32")
        cam.set_param("auto_wb", 1)
        cam.start_acquisition()

        new_camera_matrix, roi = None, None
        if CALIB:
            w, h = cam.get_width(), cam.get_height()
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix, dist_coeffs, (w, h), 1, (w, h)
            )
            print(f"Kalibracia nacitana, fx={round(fx, 2)}")
        else:
            print("Kalibracny subor nenajdeny, bezi bez undistorcie")
        print("Q = koniec")

        while True:
            canny_min   = cv2.getTrackbarPos("Canny min",      "Nastavenia")
            canny_max   = cv2.getTrackbarPos("Canny max",      "Nastavenia")
            min_area    = cv2.getTrackbarPos("Min plocha",     "Nastavenia")
            circularity = cv2.getTrackbarPos("Circularity",    "Nastavenia")
            dist_cm     = cv2.getTrackbarPos("Vzdialenost cm", "Nastavenia")

            cam.get_image(xi_img)
            frame = xi_img.get_image_data_numpy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            if CALIB:
                frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
                x, y, w, h = roi
                frame = frame[y:y+h, x:x+w]

            gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges   = cv2.Canny(blurred, canny_min, canny_max)
            edges   = cv2.dilate(edges, None, iterations=1)
            edges   = cv2.erode(edges,  None, iterations=1)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                shape_name = classify_shape(contour, min_area, circularity)
                if shape_name is None:
                    continue

                moments = cv2.moments(contour)
                if moments["m00"] == 0:
                    continue
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])

                # kreslenie tvaru
                if shape_name == "circle":
                    # minEnclosingCircle nakresli okruhly kruh (nie polygón)
                    (ex, ey), er = cv2.minEnclosingCircle(contour)
                    cv2.circle(frame, (int(ex), int(ey)), int(er), (0, 255, 0), 3)
                else:
                    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)

                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

                # vypocet rozmerov v cm
                size_label = ""
                if CALIB and dist_cm > 0:
                    if shape_name == "circle":
                        area_px    = cv2.contourArea(contour)
                        r_px       = math.sqrt(area_px / math.pi)
                        d_cm       = px_to_cm(2 * r_px, fx, dist_cm)
                        size_label = f"d={d_cm}cm"
                    else:
                        bx, by, bw, bh = cv2.boundingRect(contour)
                        w_cm       = px_to_cm(bw, fx, dist_cm)
                        h_cm       = px_to_cm(bh, fx, dist_cm)
                        if shape_name == "square":
                            size_label = f"a={w_cm}cm"
                        else:
                            size_label = f"{w_cm}x{h_cm}cm"

                label = f"{shape_name} {size_label}"
                cv2.putText(frame, label, (cx - 60, cy - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

            preview       = cv2.resize(frame, None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE)
            edges_preview = cv2.resize(edges, None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE)
            cv2.imshow("Shape Detection", preview)
            cv2.imshow("Canny edges", edges_preview)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    finally:
        cv2.destroyAllWindows()
        cam.stop_acquisition()
        cam.close_device()


if __name__ == "__main__":
    main()