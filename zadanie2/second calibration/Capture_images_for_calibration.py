from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np
from ximea import xiapi


CHESSBOARD_SIZE = (5, 7)  # inner corners (columns, rows)
CALIBRATION_CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
OUTPUT_DIR = Path(__file__).resolve().parent / "images_for_calibration"
PREVIEW_SCALE = 0.25


def save_image_unicode_safe(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image before saving: {path.name}")
    path.write_bytes(encoded.tobytes())


def main() -> None:
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : CHESSBOARD_SIZE[0], 0 : CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    cam = xiapi.Camera()
    xi_img = xiapi.Image()
    acquisition_started = False

    try:
        print("Opening first XIMEA camera...")
        cam.open_device()

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

        cam.start_acquisition()
        acquisition_started = True

        print("Controls: SPACE = validate and save if good, Q = quit")

        while True:
            cam.get_image(xi_img)
            frame = xi_img.get_image_data_numpy()

            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)

            preview = cv.resize(frame, None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE)
            cv.imshow("XIMEA Calibration Preview", preview)
            key = cv.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key != ord(" "):
                continue

            print("\nCaptured frame, finding chessboard corners...")
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            found, corners = cv.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
            if not found:
                print("... No chessboard corners found, discarding image ...")
                continue

            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), CALIBRATION_CRITERIA)
            objpoints.append(objp.copy())
            imgpoints.append(corners2)

            frame_with_corners = frame.copy()
            cv.drawChessboardCorners(frame_with_corners, CHESSBOARD_SIZE, corners2, found)

            output_path = OUTPUT_DIR / f"calib{len(objpoints)}.png"
            save_image_unicode_safe(output_path, frame_with_corners)
            print(f"*** Chessboard corners found, saved: {output_path.name} ***")

        print(f"Finished. Saved valid images: {len(objpoints)}")

    finally:
        cv.destroyAllWindows()
        if acquisition_started:
            cam.stop_acquisition()
        cam.close_device()
        print("Camera closed.")


if __name__ == "__main__":
    main()