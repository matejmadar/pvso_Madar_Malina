import numpy as np
from ximea import xiapi
import cv2
import os

SIZE = (320, 320)
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


# ak sa neodfotili 4 snimky

if len(snimky) < 4:
    exit()

h, w = SIZE[1], SIZE[0]

# bod 2
mozaika = np.zeros((h*2, w*2, 3), dtype=np.uint8)
mozaika[0:h,   0:w]   = snimky[0]  # vlavo hore
mozaika[0:h,   w:w*2] = snimky[1]  # vpravo hore
mozaika[h:h*2, 0:w]   = snimky[2]  # vlavo dole
mozaika[h:h*2, w:w*2] = snimky[3]  # vpravo dole

# bod 3 - cast 1: laplace kernel (detekcia hran)
kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
mozaika[0:h, 0:w] = np.clip(cv2.filter2D(mozaika[0:h, 0:w], -1, kernel, borderType=cv2.BORDER_DEFAULT), 0, 255)

# bod 4
c2 = mozaika[0:h, w:w*2].copy()
rot = np.zeros_like(c2)
for i in range(h):
    for j in range(w):
        rot[j][h-1-i] = c2[i][j]
mozaika[0:h, w:w*2] = rot

# bod 5 – iba cerveny kanal v 3. casti mozaiky
c3 = mozaika[h:h*2, 0:w]
c3[:, :, 0] = 0   # B
c3[:, :, 1] = 0   # G

# bod 6 – informacie o obraze
print("\n   Basic info:")
print("dtype:", mozaika.dtype)
print("rozmer (shape):", mozaika.shape)
print("velkost (pocet prvkov):", mozaika.size)

cv2.imwrite("mozaika.png", mozaika)
cv2.imshow("mozaika", mozaika)
cv2.waitKey(0)
cv2.destroyAllWindows()