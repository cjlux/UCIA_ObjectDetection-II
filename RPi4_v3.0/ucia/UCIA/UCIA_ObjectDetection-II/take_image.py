######################################
#   Jean-Luc.Charles@mailo.com
#   2024/11/21 - v1.0
######################################

from picamera2 import Picamera2, Preview
import sys, time

picam2 = Picamera2()
picam2.preview_configuration.main.size = (800, 600)
picam2.configure("preview")
picam2.start_preview(Preview.QTGL, width=800, height=600)
picam2.start()

n = 1
rep = input("numéro image pour démarrer [Q:quit] ? ")

if rep.lower() == 'q':
    picam2.stop()
    sys.exit()
else:
    n = int(rep)

while True:
    rep = input(f"ENTER -> image suivante {n:03d} [Q:quit] ...")
    if rep.lower() == 'q': 
        break
	
    picam2.capture_file(f"objets3D-{n:03d}.jpg")
    time.sleep(1)
    n += 1

picam2.stop()
