from __future__ import annotations

from pathlib import Path

import cv2
from ximea import xiapi


def save_image_unicode_safe(output_path: Path, frame) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(output_path.suffix, frame)
    if not success:
        raise RuntimeError("Failed to encode image before saving.")
    output_path.write_bytes(encoded.tobytes())


def main() -> None:
    cam = xiapi.Camera()
    img = xiapi.Image()
    acquisition_started = False

    try:
        print("Opening first XIMEA camera...")
        cam.open_device()

        cam.set_exposure(50000)
        cam.set_param("imgdataformat", "XI_RGB32")
        cam.set_param("auto_wb", 1)

        max_width = cam.get_width_maximum()
        max_height = cam.get_height_maximum()
        print(f"Maximum supported resolution: {max_width} x {max_height}")

        cam.set_offsetX(0)
        cam.set_offsetY(0)
        cam.set_width(max_width)
        cam.set_height(max_height)

        current_width = cam.get_width()
        current_height = cam.get_height()
        print(f"Configured resolution: {current_width} x {current_height}")

        cam.start_acquisition()
        acquisition_started = True

        cam.get_image(img)
        frame = img.get_image_data_numpy()

        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        output_dir = Path(__file__).resolve().parent / "captured"
        output_path = output_dir / f"ximea_max_{img.width}x{img.height}.png"

        save_image_unicode_safe(output_path, frame)
        print(f"Image captured and saved to: zadanie2/captured/{output_path.name}")

    finally:
        if acquisition_started:
            cam.stop_acquisition()
        cam.close_device()
        print("Camera closed.")


if __name__ == "__main__":
    main()
