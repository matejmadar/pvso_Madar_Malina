from ximea import xiapi
import cv2
import numpy as np
import pickle
import os

CHESS  = (7, 5)
SQ_MM  = 30.0
N_MIN  = 15
RES    = (640, 480)

os.makedirs("snimky_kalibracie", exist_ok=True)

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((CHESS[0] * CHESS[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESS[0], 0:CHESS[1]].T.reshape(-1, 2) * SQ_MM

obj_pts, img_pts = [], []

cam = xiapi.Camera()
cam.open_device()
cam.set_exposure(50000)
cam.set_param("imgdataformat", "XI_RGB32")
cam.set_param("auto_wb", 1)
xi_img = xiapi.Image()
cam.start_acquisition()

last_corners = None
last_found   = False

while True:
    cam.get_image(xi_img)
    frame = cv2.resize(
        cv2.cvtColor(xi_img.get_image_data_numpy(), cv2.COLOR_BGRA2BGR), RES
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        last_found, last_corners = cv2.findChessboardCorners(
            gray, CHESS,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if last_found:
            last_corners = cv2.cornerSubPix(gray, last_corners, (11, 11), (-1, -1), CRITERIA)
            obj_pts.append(objp)
            img_pts.append(last_corners)
            # ulozenie snimky so sachovnicou
            path = f"snimky_kalibracie/snimka_{len(obj_pts):02d}.png"
            cv2.imwrite(path, frame)
            print(f"Snimka {len(obj_pts)}/{N_MIN} zachytena -> {path}")
        else:
            print("Sachovnica nenajdena, skus inak")

    vis = frame.copy()
    if last_found and last_corners is not None:
        cv2.drawChessboardCorners(vis, CHESS, last_corners, last_found)

    label = f"[SPACE] capture {len(obj_pts)}/{N_MIN}  [C] calibrate  [Q] quit"
    cv2.putText(vis, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 0) if last_found else (0, 0, 200), 1)
    cv2.imshow("calibration", vis)

    if key == ord('c') and len(obj_pts) >= N_MIN:
        break
    elif key == ord('q'):
        cam.stop_acquisition(); cam.close_device(); cv2.destroyAllWindows()
        exit()

cam.stop_acquisition()
cam.close_device()
cv2.destroyAllWindows()

# vypocet vnutornych parametrov kamery
rms, K, dist, *_ = cv2.calibrateCamera(obj_pts, img_pts, RES, None, None)

# zadanie - vypiste maticu vnutornych parametrov kamery
print(f"\nMatica vnutornych parametrov K:\n{K}")

# zadanie - urcte hodnoty fx, fy, cx, cy
print(f"\nfx={K[0,0]:.2f}  fy={K[1,1]:.2f}  cx={K[0,2]:.2f}  cy={K[1,2]:.2f}")
print(f"RMS reprojection error: {rms:.4f} px")
print(f"Distorzne koeficienty: {dist.ravel()}")

# zadanie - ulozenie matice kamery a distorznych koeficientov
with open("cam_params.pkl", "wb") as f:
    pickle.dump({"K": K, "dist": dist}, f)
print("\nParametre ulozene do cam_params.pkl")

# zadanie - demonstracia odstranenia skreslenia na realnom obraze
K_new, roi = cv2.getOptimalNewCameraMatrix(K, dist, RES, alpha=0)
undist = cv2.undistort(frame, K, dist, None, K_new)
x, y, w, h = roi
undist = undist[y:y+h, x:x+w]

# ulozenie porovnania original vs kalibrована
porovnanie = np.hstack([cv2.resize(frame, (undist.shape[1], undist.shape[0])), undist])
cv2.imwrite("undistort_result.png", porovnanie)
print("Porovnanie ulozene -> undistort_result.png")

cv2.imshow("original | undistorted", porovnanie)
cv2.waitKey(0)
cv2.destroyAllWindows()