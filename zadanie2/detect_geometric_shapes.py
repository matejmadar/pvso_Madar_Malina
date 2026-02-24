from __future__ import annotations

import math

import cv2
from ximea import xiapi


MIN_CONTOUR_AREA = 1200
PREVIEW_SCALE = 0.35


def classify_shape(contour) -> str | None:
    area = cv2.contourArea(contour)
    if area < MIN_CONTOUR_AREA:
        return None

    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
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

    circularity = 4 * math.pi * area / (perimeter * perimeter)
    if circularity > 0.80:
        return "circle"

    return None


def main() -> None:
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

        print(f"Using maximum resolution: {cam.get_width()} x {cam.get_height()}")
        print("Press Q to quit.")

        cam.start_acquisition()
        acquisition_started = True

        while True:
            cam.get_image(xi_img)
            frame = xi_img.get_image_data_numpy()

            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            edges = cv2.dilate(edges, None, iterations=1)
            edges = cv2.erode(edges, None, iterations=1)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                shape_name = classify_shape(contour)
                if shape_name is None:
                    continue

                moments = cv2.moments(contour)
                if moments["m00"] == 0:
                    continue

                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])

                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    shape_name,
                    (cx - 40, cy - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            preview = cv2.resize(frame, None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE)
            cv2.imshow("XIMEA Shape Detection", preview)

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