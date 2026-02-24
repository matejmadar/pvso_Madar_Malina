from ximea import xiapi
import cv2
import numpy as np
import pickle

# nacitanie kalibracnych parametrov z kalibracneho programu
try:
    with open("cam_params.pkl", "rb") as f:
        p = pickle.load(f)
    K, dist = p["K"], p["dist"]
    CALIB = True
except FileNotFoundError:
    CALIB = False   # kamera nekalibrоvana, bezi bez undistorcie

RES = (640, 480)

# mapa tvarov: pocet vrcholov -> (nazov, farba)
SHAPES = {
    3: ("trojuholnik", (255, 80,  0  )),
    4: ("stvorec",     (0,   200, 0  )),
    5: ("pentagon",    (200, 0,   200)),
    6: ("sestuholnik", (0,   200, 200)),
}

def classify(approx):
    n = len(approx)
    if n not in SHAPES:
        return "polygon", (160, 160, 160)
    name, color = SHAPES[n]
    if n == 4:
        # rozlisenie stvorec vs obdlznik podla pomeru stran
        _, _, w, h = cv2.boundingRect(approx)
        name = "stvorec" if 0.85 <= w / h <= 1.15 else "obdlznik"
    return name, color

def process(frame):
    # odstranenie skreslenia objektívu pomocou kalibracnych parametrov
    if CALIB:
        K_new, roi = cv2.getOptimalNewCameraMatrix(K, dist, RES, alpha=0)
        frame = cv2.undistort(frame, K, dist, None, K_new)
        x, y, w, h = roi
        frame = cv2.resize(frame[y:y+h, x:x+w], RES)

    out  = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # gaussov blur – potlacenie sumu pred detekciou
    blur = cv2.GaussianBlur(gray, (7, 7), 1.5)

    # detekcia kruznic – Houghova transformacia hlasuje v priestore (x, y, r)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1,
                                minDist=50, param1=60, param2=35,
                                minRadius=15, maxRadius=180)
    if circles is not None:
        for cx, cy, r in np.uint16(np.around(circles[0])):
            cv2.circle(out, (cx, cy), r, (0, 220, 255), 2)
            cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(out, "kruznica", (cx - r, cy - r - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)

    # detekcia hran Cannyho detektorom + dilatacia pre spojenie kontur
    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, None, iterations=1)

    # najdenie vonkajsich kontur
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in cnts:
        # ignoruj male objekty (sum, artefakty)
        if cv2.contourArea(cnt) < 600:
            continue

        # aproximacia kontury na polygon (Douglas-Peucker, eps = 2% obvodu)
        eps    = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)

        # tazisko kontury pre umiestnenie popisku
        M  = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        name, color = classify(approx)
        cv2.drawContours(out, [approx], -1, color, 2)
        cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(out, name, (cx - 35, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return out

# inicializacia kamery
cam = xiapi.Camera()
cam.open_device()
cam.set_exposure(50000)
cam.set_param("imgdataformat", "XI_RGB32")
cam.set_param("auto_wb", 1)
xi_img = xiapi.Image()
cam.start_acquisition()

# hlavna slucka – spracovanie a zobrazenie kazdeho framu
while True:
    cam.get_image(xi_img)
    frame = cv2.resize(
        cv2.cvtColor(xi_img.get_image_data_numpy(), cv2.COLOR_BGRA2BGR), RES
    )
    cv2.imshow("detekcia tvarov", process(frame))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.stop_acquisition()
cam.close_device()
cv2.destroyAllWindows()