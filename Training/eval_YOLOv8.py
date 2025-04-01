######################################
#   Jean-Luc.Charles@mailo.com
#   2024/11/18 - v1.1
######################################

from pathlib import Path
from ultralytics import YOLO
from time import sleep
import sys

def main(VER):

    BATCH = {'v1.0': (4, 8, 16, 32),}
    EPOCH = (40, 80, 120, 160)

    model_dir = Path('./Training/YOLO-pretrained')

    data_path = {'v1.0': "./datasets/V1.0_yolo8_48train-8val-4test/data.yaml",}

    yolo = 'YOLOv8s'
    yolo_weights = model_dir / f'{yolo.lower()}.pt'

    header  = '#meta-params\tpre[ms]\tinf[ms]\tloss[ms]\tpost[ms]\t'
    header += 'prec\trecall\tmAP50\tmAP50-95\tfitness\n'
    header += '#pre:preprocessing; inf:inference; post:postprocessing; prec:precision\n'

    results_dir = Path('./Training/Results')

    if not results_dir.exists():
        results_dir.mkdir()
        
    # load any network for the first time:, because there is some overhead in computing the first time
    model = YOLO(f"Training/YOLO-trained-{VER}/UCIA-II-YOLOv8s/batch-32_epo-160/weights/best.pt")
    metrics = model.val(batch=1, imgsz=640, data=data_path[VER], workers=0)  

    results_file = Path(results_dir, f"results_yolov8s-{VER}.txt")
    F_out = open(results_file, "w", encoding="utf8")
    F_out.write(header)
        
    for batch in BATCH[VER]:
        for epoch in EPOCH:
            project = f'Training/YOLO-trained-{VER}/UCIA-II-{yolo}' 
            name = f'batch-{batch:02d}_epo-{epoch:03d}'

            best = Path(project, name, 'weights', 'best.pt')
            print(best)

            if best.exists(): 
                model = YOLO(best)  # load a pretrained model 
                # Validate the model
                metrics = model.val(batch=batch, imgsz=640, data=data_path[VER], workers=0)  
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
    parser.add_argument('-V', '--version', action="store", dest='version', 
                        required=True, help="dataset version: 'v1.0', 'v1.1' or 'v2.0...'")
    
    args = parser.parse_args()
    version = args.version

    main(version)
