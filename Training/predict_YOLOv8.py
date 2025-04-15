######################################
#   Jean-Luc.Charles@mailo.com
#   2025/04/13 - v1.1
######################################

from pathlib import Path
from ultralytics import YOLO
from time import sleep
import os, sys, cv2
from PIL import Image

def do_infer(yolo_trained, image_path, confidence=0.6, maxdetect=6, grey=True):
	''' 
		Make YOLO inferences with the given weights binary file
	'''

	# Load the YOLO11 model
	model = YOLO(yolo_trained, task='detect')
	
	if image_path.is_dir():
		list_image = [ image_path / f for f in os.listdir(image_path) if f.endswith('.jpg')]
	else:
		list_image = [image_path]
	list_image.sort()

	for image in list_image:

		img   = Image.open(image)
		img   = img.resize((640, 640))
		
		if grey:
			img = img.convert("L")
		
		# Run YOLO inference on the frame
		results = model.predict(img, imgsz=640, conf=confidence, max_det=maxdetect)

		# Visualize the results on the frame
		annotated_frame = results[0].plot()

		print(f'{yolo_trained=}')
		print(f'{image_path=}')
		
		yolo_name = '/'.join(yolo_trained.parts[2:4]).replace('/','-')
		# Display the resulting frame
		cv2.imshow(f'{yolo_name}___{image.name}', annotated_frame)

		while True:	
			# Break the loop if 'q' is pressed
			if cv2.waitKey(1) == ord("q"):
				break
				
		cv2.destroyAllWindows()


# Release resources and close windows


if __name__ == '__main__':
	
	from pathlib import Path
	import argparse, sys

	# Set options to configure program execution:
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--image', action="store", dest='image', 
		required=True, help="image path")
	parser.add_argument('-v', '--version', action="store", dest='version', 
		required=False, default='v1.0', help="'v1.0' ou 'v1.1...'")
	parser.add_argument('-b', '--batch', action="store", dest='batch', 
		required=False, type=int, default='4', help="2, 4, 8, 16, ou 32")
	parser.add_argument('-e', '--epochs', action="store", dest='epochs', 
		required=False, type=int, default='100', help="20, 40, 60, 80 ou 100")
	parser.add_argument('-m', '--maxdetect', action="store", dest='maxdetect', 
		required=False, type=int, default='30', help="Nombre max d'objets à détecter.")
	parser.add_argument('-c', '--confidence', action="store", dest='conf', 
		required=False, type=float, default='0.2', help="Seuil de confiance pour afficher une détection.")
	parser.add_argument('-color', '--colorScale', action="store_true", dest='color', 
		required=False, help="Whether to convert images to grescale or not")
	
	args = parser.parse_args()
	image     = args.image
	version   = args.version
	batch     = args.batch
	epochs    = args.epochs
	maxdetect = args.maxdetect
	conf      = args.conf
	color      = args.color
	
	
	grey = not color
	print(f'{grey=}')
	
	yolo_weights_path  = f'Training/YOLO-trained-{version}/UCIA-II-YOLOv8s/'
	yolo_weights_path += f'batch-{batch:02d}_epo-{epochs:03d}/weights/best.pt'
	#rep = input(f'Utiliser le réseau {yolo_weights_path} [Oui]/Non ? ')
	rep = 'o'
	
	if rep.lower() in ('', 'o', 'oui', 'y', 'yes'):
		yolo_weights = Path(yolo_weights_path)
		if not yolo_weights.exists():
			print(f'fichier inexitant: <{yolo_weights}>, désolé.')
		else:
			do_infer(yolo_weights, Path(image), conf, maxdetect, grey)
	
