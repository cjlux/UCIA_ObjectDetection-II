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

def main(VER:str, force_NCNN=False):

    BATCH = {'v1.3': (4, 8, 16, 32),
             'v1.4': (4, 8, 16, 20, 30, 40),
             'v1.5': (4, 8, 16, 20, 30, 40),
             'v1.6': (4, 8, 16, 20, 30, 40),
             'v1.7': (4, 8, 16, 20, 30, 40),
             'v1.8': (4, 8, 16, 20, 30, 40),
             }
    
    EPOCH = {'v1.3': (40, 80, 120, 160, 200),
             'v1.4': (80, 120, 160, 200, 240, 300),
             'v1.5': (80, 120, 160, 200, 240, 300, 400, 500),
             'v1.6': (80, 120, 160, 200, 240, 300),
             'v1.7': (80, 120, 160, 200, 240, 300),
             'v1.8': (80, 120, 160, 200, 240, 300),
             }
    
    PATIENCE = {'v1.3': 100, 'v1.4': 100, 'v1.5': 100, 'v1.6': 40, 'v1.7': 40, 'v1.8': 50,}

    model_dir = Path('./Training/YOLO-pretrained')

    data_path = {'v1.3': "./datasets/V1.1_yolo8_166train-24val-13test/data.yaml",
                 'v1.4': "./datasets/V1.0_yolo8_162train-27val-13test/data.yaml",
                 'v1.5': "./datasets/V1.1_yolo8_247train-85val-0test/data.yaml",
                 'v1.6': "./datasets/V1.0_yolo8_248train-84val-0test/data.yaml",
                 'v1.7': "./datasets/V1.0_yolo8_280train-72val-15test/data.yaml",
                 'v1.8': "./datasets/V1.0_yolo8_296train-67val-23test/data.yaml",
                 }

    yolo = 'YOLOv8n'
    yolo_weights = f'{yolo.lower()}.pt'

    for batch in BATCH[VER]:
        for epoch in EPOCH[VER]:
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
                            patience=PATIENCE[VER],
                            cache=False,
                            workers=10,			# no parallelisation for loading data
                            project=project,
                            name=name,
                            exist_ok=True,
                            pretrained=True,
                            optimizer='auto',
                            seed=1234,
                            deterministic=True, # force using deterministic algorithms
                            overlap_mask=False) # whether object masks should be merged into a single mask for training 
												# or kept separate for each object.

            print(f'looking for <best_ncnn_model>... ')            
            best_ncnn = Path(project, name, 'weights', 'best_ncnn_model')
            if not best_ncnn.exists() or force_NCNN:
                print('\t exporting <best.pt> to <best_ncnn_model>...', end="")
                model = YOLO(best)  # load the best custom trained model
                model.export(format="ncnn", imgsz=640, half=True, data=data_path[VER])
                print(" done.")
           
                
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', action="store", dest='version', 
                        required=True, help="dataset version: 1.0, 1.1...'")
    parser.add_argument('--NCNN', action='store_true', dest='NCNN', default=False)
    args = parser.parse_args()
    version = f'v{args.version}'
    NCNN    = args.NCNN

    main(version, force_NCNN=NCNN)
