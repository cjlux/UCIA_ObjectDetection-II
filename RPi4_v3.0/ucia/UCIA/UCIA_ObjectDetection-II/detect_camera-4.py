######################################
#   Jean-Luc.Charles@mailo.com
#   2025/04/28 - v1.0
######################################

from flask import Flask, Response
from picamera2 import Picamera2
import cv2, os, errno
import numpy as np

from ultralytics import YOLO

from PIL import ImageFont, ImageDraw, Image

font1 = ImageFont.truetype("/home/ucia/.config/Ultralytics/Arial.ttf", 12)
font2 = ImageFont.truetype("/home/ucia/.config/Ultralytics/Arial.ttf", 14)

# Classe selection:
class_nb   = 15
classe_sel = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

class_name_en  = ('Park', 'Ball','Cock','Cube','Cyl','Hexa',  'Home',   'laneL',   'laneR',     'lineS','Nest',  'Star','Stop','Tri','Cross')
class_name_fr  = ('Park','Balle','Coca','Cube','Cyl','Hexa','Maison','virage G','virage D','Tout Droit', 'Nid','Etoile','Stop','Tri','Clous')
label_width    = (    55,     59,    59,    62,   55,    62,      68,        75,        75,          75,    58,      70,    57,   50,     60)
box_hue        = (     0,     25,    51,    70,   95,   140,     170,       190,       210,         230,   260,     282,   308,  334,    360)
box_sat        = (   100,    100,   100,   100,  100,   100,     100,       100,       100,         110,   100,     100,   100,  100,     50)

# text HSL: (0, 0%, 0%) -> black and (0, 0%, 100%) -> white
text_hue = [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]
text_sat = [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]
text_lum = [100, 100,   0,   0,   0,   0,   0,   0, 100, 100, 100, 100, 100, 100, 100]

# Pipe for the communication with Thymio
FIFO = '/tmp/RPi4_pipe'
try:
	os.mkfifo(FIFO)
except OSError as oe:
	if oe.errno != errno.EEXIST:
		raise

print("Trying to open the FIFO to communicate with Thymio...", end ='')
fifo = open(FIFO, 'w')
print( " done.")

app = Flask(__name__)

def generate_frames():
	
	class_name = class_name_en if not french else class_name_fr
	
	while True:

		# Capture frame-by-frame
		img = picam2.capture_image()
		img = img.resize((640, 640))
		img_g  = img.convert("L")

		# Run YOLO inference on the frame
		results = model.predict(img_g, imgsz=640, classes=classe_sel, conf=confidence, max_det=maxdetect)
		boxes   = results[0].boxes

		draw = ImageDraw.Draw(img, 'RGBA')
		# Draw the networh path
		draw.rectangle([(10, 30),(630, 10)], fill=(150, 100, 90))
		draw.text((10, 13), yolo_weights_path, font=font2, fill=(250, 250, 250), size=14)

		for class_id, conf, xyxy, in zip(boxes.cls, boxes.conf, boxes.xyxy):
			class_id = int(class_id)
			name  = class_name[class_id]
			conf_str = f'{conf:.2f}'
			label = f' {name:5s} {conf_str}'
			box_hsl  = f'hsl({box_hue[class_id]}, {box_sat[class_id]}%, 50%)'
			text_hsl = f"hsl({text_hue[class_id]}, {text_sat[class_id]}%, {text_lum[class_id]}%)" 
			# The bounding box coordinates:
			x1, y1, x2, y2  = xyxy.numpy().astype(int)
			# object center coordinates:
			cx, cy = (x1+x2)//2, (y1+y2)//2
			# mean object color around the center:
			color = np.array(img)[cy-7:cy+7, cx-7:cx++7].mean(axis=0).mean(axis=0).astype(int)
			# send data to the fifo
			message = f'{class_id},{int(conf*100)},{x1},{y2},{x2},{y1},{color[0]},{color[1]},{color[2]}\n'
			fifo.write(message)
			fifo.flush()
			print(message, end='')
			
			# The bounding box
			draw.rectangle([(x1, y1), (x2, y2)], outline=box_hsl, width=2)
			# The label inside its white rectangle:
			draw.rectangle(((x1, y1), (x1 + label_width[class_id], y1-12)), fill=box_hsl)
			draw.text((x1, y1-11), label, font=font1, fill=text_hsl)
			
		# end of detection: send en empty string to Thymio 
		fifo.write('-1\n')
		fifo.flush()
			
		np_img = np.array(img)[::,::,::-1]
		ret, buffer = cv2.imencode('.jpg', np_img)
		frame = buffer.tobytes()
		yield (b'--frame\r\n'
		b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video_feed():
	return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/halt')
def halt():
    print('Shuting down the RPi4...')
    os.system('shutdown -h now')

@app.route('/quit')
def quit():
	print('Quit the application...')
	fifo.close()
	os.kill(os.getpid(), 9)
	sys.exit(0)
    
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
	parser.add_argument('-E', '--english', action="store_true", dest='english', default=False)
		
	args = parser.parse_args()
	version   = f'v{args.version}'
	batch     = args.batch
	epochs    = args.epochs
	maxdetect = args.maxdetect
	conf      = args.conf
	french    = not args.english
	
	yolo_ver  = 'v8n'
	yolo_weights_path  = f'Training/YOLO-trained-{version}/UCIA-II-YOLO{yolo_ver}/'
	yolo_weights_path += f'batch-{batch:02d}_epo-{epochs:03d}/weights/best_ncnn_model'
	print(f'Inférences du réseau {yolo_weights}')

	if not yolo_weights.exists():
		print(f'fichier inexitant: <{yolo_weights}>, désolé.')
	else:
		# Initialize the Picamera2
		picam2 = Picamera2()
		picam2.preview_configuration.main.size = (640, 640)
		picam2.preview_configuration.main.format = "RGB888"
		picam2.preview_configuration.align()
		picam2.configure("preview")
		picam2.start()
		
		model = YOLO(yolo_weights, task='detect')
		
		app.run(host='0.0.0.0', port=5000)
		
