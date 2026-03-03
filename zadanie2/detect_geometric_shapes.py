from __future__ import annotations

import math
import json
from pathlib import Path
import numpy as np
import cv2
from ximea import xiapi

CALIBRATION_JSON = Path(__file__).resolve().parent / "camera_parameters.json"

PREVIEW_SCALE = 0.4


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


def nothing(_): pass


def load_calibration(json_path: Path):
    # nacitanie kalibracnych parametrov
    if not json_path.exists():
        return None, None
    data          = json.loads(json_path.read_text(encoding="utf-8"))
    camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    dist_coeffs   = np.array(data["dist_coeffs"],   dtype=np.float64)
    if dist_coeffs.ndim == 1:
        dist_coeffs = dist_coeffs.reshape(1, -1)
    return camera_matrix, dist_coeffs


def main() -> None:
    camera_matrix, dist_coeffs = load_calibration(CALIBRATION_JSON)
    CALIB = camera_matrix is not None

    cam = xiapi.Camera()
    xi_img = xiapi.Image()

    # okno so slidermi
    cv2.namedWindow("Nastavenia")
    cv2.createTrackbar("Canny min",    "Nastavenia", 50,   500, nothing)
    cv2.createTrackbar("Canny max",    "Nastavenia", 150,  500, nothing)
    cv2.createTrackbar("Min plocha",   "Nastavenia", 1200, 5000, nothing)
    cv2.createTrackbar("Circularity",  "Nastavenia", 85,   100, nothing)

    try:
        cam.open_device()
        cam.set_exposure(50000)
        cam.set_param("imgdataformat", "XI_RGB32")
        cam.set_param("auto_wb", 1)
        cam.start_acquisition()

        # vypocet undistort matice raz pred sluckou
        new_camera_matrix, roi = None, None
        if CALIB:
            w, h = cam.get_width(), cam.get_height()
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix, dist_coeffs, (w, h), 1, (w, h)
            )
            print("Kalibracia nacitana, undistorcia zapnuta")
        else:
            print("Kalibracny subor nenajdeny, bezi bez undistorcie")
        print("Q = koniec")

        while True:
            # nacitanie hodnot zo slidermi
            canny_min  = cv2.getTrackbarPos("Canny min",   "Nastavenia")
            canny_max  = cv2.getTrackbarPos("Canny max",   "Nastavenia")
            min_area   = cv2.getTrackbarPos("Min plocha",  "Nastavenia")
            circularity = cv2.getTrackbarPos("Circularity", "Nastavenia")

            cam.get_image(xi_img)
            frame = xi_img.get_image_data_numpy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # odstranenie skreslenia objektívu
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

                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                cv2.putText(frame, shape_name, (cx - 40, cy - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)

            preview = cv2.resize(frame, None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE)
            cv2.imshow("Shape Detection", preview)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    finally:
        cv2.destroyAllWindows()
        cam.stop_acquisition()
        cam.close_device()


if __name__ == "__main__":
    main()