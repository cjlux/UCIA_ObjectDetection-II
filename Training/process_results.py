######################################
#   Jean-Luc.Charles@mailo.com
#   2025/04/13 - v1.1
######################################

import pandas as pd
from pathlib import Path

def main(VER):

    results_dir = Path('./Training/Results')
    out_file = Path(results_dir, f"processed_res-{VER}.txt")
    
    with open(out_file, "w", encoding="utf8") as stream_out:

        mess = 50*'*' + "\n* Sort by 'recall' & 'mAP50-95'\n" + 50*'*' 
        print(mess)
        stream_out.write(mess+'\n')   

        txt_file = Path(results_dir, f'results_yolov8s-{VER}.txt')
            
        mess = f'\nFile <{txt_file}>'
        print(mess, end="")
        stream_out.write(mess)
            
        # read CSV file with panda:
        df = pd.read_csv(txt_file, sep='\t', header=0, skiprows=[1])
        # sort rows by descending order of columns "recall","mAP50-95":
        df = df.sort_values(by=["recall","mAP50-95", ], ascending=False)
        # the first values in columns "recall" and "mAP50-95  are the max values:
        max_mAP50_90 = df['mAP50-95'].values[0]
        max_recall   = df['recall'].values[0]

        mess = f'\n\tMax values -> "max_recall": {max_recall}, "max_mAP50-90": {max_mAP50_90}' 
        print(mess)
        stream_out.write(mess+'\n')
        
        # selected  significant columns
        df1 = df[['#meta-params', 'recall', 'mAP50-95', 'fitness']]
        
        # print the first 4 rows:
        mess = df1.head(4)
        print(mess)
        stream_out.write(str(mess)+'\n')
            

        mess = '\n' + 50*'*' + "\n* Sort by 'fitness'\n" + 50*'*'
        print(mess)
        stream_out.write(mess+'\n')   



        txt_file = Path(results_dir, f'results_yolov8s-{VER}.txt')
        mess = f'\nFile <{txt_file}>'
        print(mess, end="")
        stream_out.write(mess)
        
        # read CSV file with panda:
        df = pd.read_csv(txt_file, sep='\t', header=0, skiprows=[1])            
        # now sort rows by descending order of column "fitnes":
        df = df.sort_values(by=["fitness"], ascending=False)
        # the first values in column "fitness" is the max values
        max_fitness = df['fitness'].values[0]

        mess = f'\n\tMax values -> "fitness": {max_fitness}'
        print(mess)
        stream_out.write(mess+'\n')
        
        # selected  significant columns
        df2 = df[['#meta-params', 'recall', 'mAP50-95', 'fitness']]
        
        # print the first 4 rows:
        mess = df2.head(4)
        print(mess)
        stream_out.write(str(mess)+'\n')

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-V', '--version', action="store", dest='version', 
                        required=True, help="dataset version: 'v1.0', 'v1.1' or 'v2.0...'")
    
    args = parser.parse_args()
    version = args.version
    
    main(version)
