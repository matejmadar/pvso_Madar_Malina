from ximea import xiapi
import cv2
import os
import sys

# NASTAVENIA KAMERY – uprav podľa podmienok osvetlenia

OUTPUT_DIR    = "/home/matej/gaussian_data/moja_scena/input"  # kam sa ukladajú fotky

EXPOSURE_US   = 15000       # expozícia v µs - znížit pri jasnom svetle
GAIN_DB       = 0.0        # gain (pridáva šum)

WB_RED        = 1.6        # vyváženie bielej
WB_GREEN      = 1.0
WB_BLUE       = 1.8

IMAGE_FORMAT  = "XI_RGB24" # farebný obraz
SAVE_FORMAT   = ".png"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cam = xiapi.Camera()
    print("Otváranie kamery...")
    cam.open_device()
    print(f"Kamera: {cam.get_device_name()}  |  SN: {cam.get_device_sn()}")

    # Pevné nastavenia
    cam.disable_aeag()
    cam.set_exposure(EXPOSURE_US)
    cam.set_gain(GAIN_DB)
    cam.disable_auto_wb()
    cam.set_wb_kr(WB_RED)
    cam.set_wb_kg(WB_GREEN)
    cam.set_wb_kb(WB_BLUE)
    cam.set_imgdataformat(IMAGE_FORMAT)

    # Maximálne rozlíšenie
    cam.set_width(cam.get_width_maximum())
    cam.set_height(cam.get_height_maximum())
    cam.set_offsetX(0)
    cam.set_offsetY(0)

    print(f"Rozlíšenie: {cam.get_width()} x {cam.get_height()}")
    print(f"Expozícia: {cam.get_exposure()} µs  |  Gain: {cam.get_gain()} dB")
    print(f"Ukladanie do: {OUTPUT_DIR}")
    print()
    print("OVLÁDANIE:")
    print("  MEDZERNÍK  – ulož snímku")
    print("  S          – ulož snímku (alternatíva)")
    print("  Q / ESC    – ukonči")
    print()

    img = xiapi.Image()
    cam.start_acquisition()

    count = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(SAVE_FORMAT)])
    print(f"Existujúce snímky v priečinku: {count}")

    while True:
        cam.get_image(img)
        frame = img.get_image_data_numpy()

        # XIMEA na tejto kamere vracia BGR priamo, žiadna konverzia
        display = frame.copy()

        info = f"Snimky: {count}  |  MEDZERNIK = uloz  |  Q = koniec"
        cv2.putText(display, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        h, w = display.shape[:2]
        preview = cv2.resize(display, (w // 2, h // 2))
        cv2.imshow("XIMEA – MEDZERNIK = uloz, Q = koniec", preview)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord(' '), ord('s')):
            filename = os.path.join(OUTPUT_DIR, f"img_{count:04d}{SAVE_FORMAT}")

            save_frame = frame.copy()

            if SAVE_FORMAT == ".png":
                cv2.imwrite(filename, save_frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            else:
                cv2.imwrite(filename, save_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            count += 1
            print(f"  Uložená: {filename}  ({count} celkom)")

        elif key in (ord('q'), 27):
            break

    cam.stop_acquisition()
    cam.close_device()
    cv2.destroyAllWindows()
    print(f"\nHotovo. Celkom uložených snímok: {count}")
    print(f"Priečinok: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()