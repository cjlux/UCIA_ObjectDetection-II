##########################################################################
# Program inspired from 
# https://docs.ultralytics.com/fr/guides/raspberry-pi/#inference-with-camera
#   
# Modified by Jean-Luc CHARLES (Jean-Luc.Charles@mailo.com)
# 
#   2025/04/28 - v1.2
#
##########################################################################

import cv2
from picamera2 import Picamera2

from ultralytics import YOLO

def do_infer(yolo_trained, confidence=0.6, maxdetect=6):
	''' 
		Make YOLO inferences with the given weights binary file
	'''
	
	# Initialize the Picamera2
	picam2 = Picamera2()
	picam2.preview_configuration.main.size = (640, 640)
	picam2.preview_configuration.main.format = "RGB888"
	picam2.preview_configuration.align()
	picam2.configure("preview")
	picam2.start()

	# Load the YOLO11 model
	model = YOLO(yolo_trained, task='detect')

	while True:

		# Capture frame-by-frame
		#frame = picam2.capture_array()
		img = picam2.capture_image()
		img = img.resize((640, 640))
		img_g  = img.convert("L")
		
		# Run YOLO inference on the frame
		results = model.predict(img_g, imgsz=640, conf=confidence, max_det=maxdetect)

		# Visualize the results on the frame
		annotated_frame = results[0].plot()

		# Display the resulting frame
		cv2.imshow(str(yolo_trained), annotated_frame)

		# Break the loop if 'q' is pressed
		if cv2.waitKey(1) == ord("q"):
			break

# Release resources and close windows
cv2.destroyAllWindows()

if __name__ == '__main__':
	
	from pathlib import Path
	import argparse, sys

	# Set options to configure program execution:
	parser = argparse.ArgumentParser()
	parser.add_argument('-v', '--version', action="store", dest='version', 
		required=False, default='1.8', help="1.6, 1.7, v1.8")
	parser.add_argument('-b', '--batch', action="store", dest='batch', 
		required=False, type=int, default='30', help="4, 8, 16, 20, 30 ou 40")
	parser.add_argument('-e', '--epochs', action="store", dest='epochs', 
		required=False, type=int, default='300', help="40, 80, 120, 160, 200, 240, 300")
	parser.add_argument('-m', '--maxdetect', action="store", dest='maxdetect', 
		required=False, type=int, default='15', help="Nombre max d'objets à détecter.")
	parser.add_argument('-c', '--confidence', action="store", dest='conf', 
		required=False, type=float, default='0.5', help="Seuil de confiance pour afficher une détection.")
	
	args = parser.parse_args()
	version   = f'v{args.version}'
	batch     = args.batch
	epochs    = args.epochs
	maxdetect = args.maxdetect
	conf      = args.conf
	
	yolo_ver = 'v8n'
	yolo_weights_path  = f'Training/YOLO-trained-{version}/UCIA-II-YOLO{yolo_ver}/'
	yolo_weights_path += f'batch-{batch:02d}_epo-{epochs:03d}/weights/best_ncnn_model'

	yolo_weights = Path(yolo_weights_path)
	if not yolo_weights.exists():
		print(f'fichier inexitant: <{yolo_weights}>, désolé.')
	else:
		do_infer(yolo_weights, conf, maxdetect)
	
