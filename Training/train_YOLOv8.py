######################################
#   Jean-Luc.Charles@mailo.com
#   2024/12/18 - v1.1
######################################

from pathlib import Path
from ultralytics import YOLO
from time import sleep

#
# this programm must be run from the UCIA_ObjectDetection directory
#

def main(VER):

    BATCH = {'V1': (4, 8, 16, 32)}
    EPOCH = (40, 80, 120)

    model_dir = Path('./Training/YOLO-pretrained')

    data_path = {'V1': "./yolo_split_dataset/data.yaml" }

    yolo = 'YOLOv8s'
    yolo_weights = yolo.lower() + '.pt'

    for batch in BATCH[VER]:
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
                            workers=0,
                            project=project, 
                            name=name, 
                            exist_ok=True, 
                            pretrained=True,
                            optimizer='auto', 
                            seed=1234)

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
    parser.add_argument('-v', '--version', action="store", dest='version', 
                        required=True, help="Work version: 'V1', 'V2' or 'V2.1'")
    
    args = parser.parse_args()
    version = args.version

    main(version)
