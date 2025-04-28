VER = 'v1.6'
weights_path = YOLO-trained-${VER}/UCIA-II-YOLOv8n/

weights: ncnn

onnx: ./Training/YOLO-trained-${VER}-onnx.tgz
	
ncnn: ./Training/YOLO-trained-${VER}-ncnn.tgz

Training/YOLO-trained-${VER}-onnx.tgz:
	@echo Getting ONNX weights in "<${weights_path}>"
	cd `dirname $@` && tar cvzf `basename $@` `find ${weights_path}. -name "best.onnx"`
	
Training/YOLO-trained-${VER}-ncnn.tgz:
	@echo Getting NCNN weights in "<${weights_path}>"
	cd `dirname $@` && tar cvzf `basename $@` `find ${weights_path}. -name "best_ncnn_model"`

clean:
	rm -rf runs yolo11n.pt

clean_onnx:
	find . -name best.onnx -exec rm -f {} \;

clean_ncnn:
	find . -name best_ncnn_model -exec rm -rf {} \;