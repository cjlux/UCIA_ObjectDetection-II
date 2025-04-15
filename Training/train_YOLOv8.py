######################################
#   Jean-Luc.Charles@mailo.com
#   2025/04/14 - v1.1
######################################

from pathlib import Path
from ultralytics import YOLO
from time import sleep

#
# this programm must be run from the UCIA_ObjectDetection directory
#

def main(VER):

    BATCH = (4, 8, 16, 32)
    EPOCH = (40, 80, 120, 160, 180, 200)

    model_dir = Path('./Training/YOLO-pretrained')

    data_path = {'v1.0': "./datasets/V1.0_yolo8_48train-8val-4test/data.yaml",
			     'v1.1': "./datasets/V1.0_yolo8_97train-12val-4test/data.yaml",
			     'v1.2': "./datasets/V1.0_yolo8_166train-24val-13test/data.yaml",}

    yolo = 'YOLOv8s'
    yolo_weights = model_dir / f'{yolo.lower()}.pt'

    for batch in BATCH:
        for epoch in EPOCH:
            project = f'Training/YOLO-trained-{VER}/UCIA-II-{yolo}' 
            name = f'batch-{batch:02d}_epo-{epoch:03d}'
            best = Path(project, name, 'weights', 'best.pt')
            print(f'{best}')

            if not best.exists(): 	
                model = YOLO(model_dir / yolo_weights)  # load a pretrained model 
                model.train(data=data_path[VER], 
                            epochs=epoch, 
                            imgsz=640, 
                            batch=batch, 
                            patience=100,
                            cache=False,
                            workers=0,			# no parallesiation for loading data
                            project=project, 
                            name=name, 
                            exist_ok=True, 
                            pretrained=True,
                            optimizer='auto', 
                            seed=1234,
                            deterministic=True, # force using deterministic algorithms
                            overlap_mask=False) # whether object masks should be merged into a single mask for training 
												# or kept separate for each object.

            '''
            print(f'looking for <best.onnx>... ')            
            best_onnx = Path(project, name, 'weights', 'best.onnx')
            if not best_onnx.exists():
                print('\t exporting <best.pt> to <best.onnx>...', end="")
                model = YOLO(best)  # load a custom trained model
                model.export(format="onnx", int8=True, data=data_path[VER])
                print(" done.")

            print(f'looking for <best_ncnn_model>... ')            
            best_ncnn_model = Path(project, name, 'weights', 'best_ncnn_model')
            if not best_ncnn_model.exists():
                print('\t exporting <best.pt> to <best_ncnn_model>...', end="")
                model = YOLO(best)  # load a custom trained model
                model.export(format="ncnn", int8=True, data=data_path[VER])
                print(" done.")
                '''
                
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-V', '--version', action="store", dest='version', 
                        required=True, help="dataset version: 'v1.0', 'v1.1' or 'v2.0...'")
    
    args = parser.parse_args()
    version = args.version

    main(version)
