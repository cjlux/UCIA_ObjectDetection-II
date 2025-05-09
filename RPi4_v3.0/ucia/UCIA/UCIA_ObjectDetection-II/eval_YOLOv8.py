######################################
#   Jean-Luc.Charles@mailo.com
#   2025/04/28 - v1.2
######################################

from pathlib import Path
from ultralytics import YOLO
from time import sleep
import sys

def main(VER, NNformat='ncnn'):

    if NNformat == 'ncnn':
        weights_name = 'best_ncnn_model'
    else:
        weights_name = 'best.onnx'

    BATCH = {'v1.3': (4, 8, 16, 32),
             'v1.4': (4, 8, 16, 20, 30, 40),
             'v1.5': (4, 8, 16, 20, 30, 40),
             'v1.6': (4, 8, 16, 20, 30, 40),
             'v1.7': (4, 8, 16, 20, 30, 40),
             'v1.8': (4, 8, 16, 20, 30, 40),
            }
    
    EPOCH = {'v1.3': (40, 80, 120, 160, 180, 200),
             'v1.4': (40, 80, 120, 160, 200, 240),
             'v1.5': (40, 80, 120, 160, 200, 240, 300, 400, 500),
             'v1.6': (40, 80, 120, 160, 200, 240, 300),
             'v1.7': (40, 80, 120, 160, 200, 240, 300),
             'v1.8': (40, 80, 120, 160, 200, 240, 300),
            }

    model_dir = Path('./Training/YOLO-pretrained')

    data_path = {'v1.3': "./datasets/V1.1_yolo8_166train-24val-13test/data.yaml",
                 'v1.4': "./datasets/V1.0_yolo8_162train-27val-13test/data.yaml",
                 'v1.5': "./datasets/V1.1_yolo8_247train-85val-0test/data.yaml",
                 'v1.6': "./datasets/V1.0_yolo8_248train-84val-0test/data.yaml",
                 'v1.7': "./datasets/V1.0_yolo8_280train-72val-15test/data.yaml",
                 'v1.8': "./datasets/V1.0_yolo8_296train-67val-23test/data.yaml",
                }

    yolo = 'YOLOv8n'

    header  = '#meta-params\tpre[ms]\tinf[ms]\tloss[ms]\tpost[ms]\t'
    header += 'prec\trecall\tmAP50\tmAP50-95\tfitness\n'
    header += '#pre:preprocessing; inf:inference; post:postprocessing; prec:precision\n'

    results_dir = Path('./Training/Results')

    if not results_dir.exists():
        results_dir.mkdir()
        
    results_file = Path(results_dir, f"results_yolov8n-{VER}_{NNformat}_batch-1.txt")
    F_out = open(results_file, "w", encoding="utf8")
    F_out.write(header)
        
    for batch in BATCH[VER]:
        for epoch in EPOCH[VER]:
            project = f'Training/YOLO-trained-{VER}/UCIA-II-{yolo}' 
            name = f'batch-{batch:02d}_epo-{epoch:03d}'

            best = Path(project, name, 'weights', weights_name)
            print(best)

            if best.exists(): 
                model = YOLO(best)  # load a pretrained model 
                # Validate the model
                metrics = model.val(batch=batch,
                                    imgsz=640,
                                    data=data_path[VER],
                                    workers=0,
                                    conf=0.3,
                                    max_det=15,
                                    save_crop=True,
                                    plots=True,
                                    name=name)
                
                F_out.write(f'{name}')
                for key in metrics.speed:
                    F_out.write(f'\t{metrics.speed[key]:.2f}')
                for key in metrics.results_dict:
                    F_out.write(f'\t{metrics.results_dict[key]:.3f}')

                F_out.write('\n')

                del model

    F_out.close()

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', action="store", dest='version', 
                        required=True, help="dataset version: 1.0, v1.1, ...")
    parser.add_argument('-f', '--format', action="store", dest='format', default='ncnn', 
                        required=False, help="Format of the weights file: 'onnx' or 'ncnn'")
    
    args = parser.parse_args()
    version  = f'v{args.version}'
    NNformat = args.format

    main(version, NNformat=NNformat)
