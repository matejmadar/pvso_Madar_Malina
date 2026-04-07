from ximea import xiapi
import cv2

# =============================================================================
# NASTAVENIA - tu môžete meniť parametre kamery
# =============================================================================

EXPOSURE_US      = 10000   # čas expozície v mikrosekundách (napr. 1000 = 1ms)
GAIN_DB          = 0.0     # zosilnenie v dB (0 = žiadne, vyššie = jasnejší obraz)

AUTO_EXPOSURE    = False   # True = automatická expozícia a gain
AUTO_WB          = False   # True = automatické vyváženie bielej

WB_RED           = 1.0     # koeficient červeného kanála (ručné vyváženie bielej)
WB_GREEN         = 1.0     # koeficient zeleného kanála
WB_BLUE          = 1.0     # koeficient modrého kanála

IMAGE_FORMAT     = "XI_RGB24"   # formát obrazu: "XI_RGB24", "XI_MONO8", "XI_RAW8"

# =============================================================================

cam = xiapi.Camera()
print("Otváranie kamery...")
cam.open_device()

print(f"Kamera: {cam.get_device_name()}  |  SN: {cam.get_device_sn()}")

# Expozícia a gain
if AUTO_EXPOSURE:
    cam.enable_aeag()
    print("Auto expozícia: ZAP")
else:
    cam.disable_aeag()
    cam.set_exposure(EXPOSURE_US)
    cam.set_gain(GAIN_DB)
    print(f"Expozícia: {cam.get_exposure()} µs  |  Gain: {cam.get_gain()} dB")

# Vyváženie bielej
if AUTO_WB:
    cam.enable_auto_wb()
    print("Auto WB: ZAP")
else:
    cam.disable_auto_wb()
    cam.set_wb_kr(WB_RED)
    cam.set_wb_kg(WB_GREEN)
    cam.set_wb_kb(WB_BLUE)

# Formát obrazu
cam.set_imgdataformat(IMAGE_FORMAT)

img = xiapi.Image()
cam.start_acquisition()
print("Snímaní spustené. Stlačte 'q' pre ukončenie.")

while True:
    cam.get_image(img)
    frame = img.get_image_data_numpy()

    # XIMEA vracia RGB, OpenCV očakáva BGR
    if IMAGE_FORMAT == "XI_RGB24":
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow("XIMEA kamera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.stop_acquisition()
cam.close_device()
cv2.destroyAllWindows()
print("Hotovo.")
