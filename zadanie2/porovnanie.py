from ximea import xiapi
import cv2
import numpy as np
import pickle
import os

RES = (640, 480)
os.makedirs("porovnanie", exist_ok=True)

# nacitanie kalibracnych parametrov
with open("cam_params.pkl", "rb") as f:
    p = pickle.load(f)
K, dist = p["K"], p["dist"]
K_new, roi = cv2.getOptimalNewCameraMatrix(K, dist, RES, alpha=0)

# inicializacia kamery
cam = xiapi.Camera()
cam.open_device()
cam.set_exposure(50000)
cam.set_param("imgdataformat", "XI_RGB32")
cam.set_param("auto_wb", 1)
xi_img = xiapi.Image()
cam.start_acquisition()

pocet = 0
print("medzernik = uloz fotku, q = koniec")

while True:
    cam.get_image(xi_img)
    frame = cv2.resize(
        cv2.cvtColor(xi_img.get_image_data_numpy(), cv2.COLOR_BGRA2BGR), RES
    )

    # undistorcia
    undist = cv2.undistort(frame, K, dist, None, K_new)
    x, y, w, h = roi
    undist = cv2.resize(undist[y:y+h, x:x+w], RES)

    # popisky
    cv2.putText(frame, "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(undist, "KALIBROVANA", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

    # zobrazenie vedla seba
    oba = np.hstack([frame, undist])
    cv2.imshow("porovnanie (L=original, P=kalibrovana)", oba)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        pocet += 1
        cv2.imwrite(f"porovnanie/foto_{pocet}.png", oba)
        print(f"ulozene -> porovnanie/foto_{pocet}.png")
    elif key == ord('q'):
        break

cam.stop_acquisition()
cam.close_device()
cv2.destroyAllWindows()