from ximea import xiapi
import cv2
import os

SIZE = (1280, 640)
os.makedirs("snimky", exist_ok=True)

cam = xiapi.Camera()
cam.open_device()
cam.set_exposure(50000)
cam.set_param("imgdataformat", "XI_RGB32")
cam.set_param("auto_wb", 1)

img = xiapi.Image()
cam.start_acquisition()
print("medzernik - snimanie")
print("q - koniec")

snimky = []
while len(snimky) < 4:
    cam.get_image(img)
    frame = cv2.resize(cv2.cvtColor(img.get_image_data_numpy(), cv2.COLOR_BGRA2BGR), SIZE)
    cv2.imshow("nahlad", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        path = f"snimky/snimok{len(snimky)+1}.png"
        cv2.imwrite(path, frame)
        snimky.append(frame.copy())
        print(f"snimka {len(snimky)}")
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
cam.stop_acquisition()
cam.close_device()