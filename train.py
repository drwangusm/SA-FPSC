import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

#bash: nohup python train.py > train.log 2>&1 & 

if __name__ == '__main__':
    # model = YOLO('ultralytics/cfg/models/v10/yolov10n.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    # ['yolov8m','yolov8n','yolov8s','yolov5','yolov6','yolov9s','yolov9m','yolov10m','yolov10n','yolov10s']
    #['sa','s','sa-fpsc-n','sa-fpsc-m','sa-fpsc-b','sa-fpsc-s']

  
    for yaml_name in ['sa-fpsc-m','sa-fpsc-b','sa-fpsc-s']:
        model = YOLO(f'/final/SA-FPSC/models/innovations/{yaml_name}.yaml')
        model.train(data='/final/datasets/WUDD/WUDD.yaml',
                    cache=False,
                    imgsz=640,
                    epochs=300,
                    batch=16, #32
                    close_mosaic=0,
                    workers=4,
                    # device='0',
                    optimizer='SGD', # using SGD
                    # patience=0, # close earlystop
                    # resume=True, # 断点续训,YOLO初始化时选择last.pt
                    amp=False, # close amp
                    # fraction=0.2,
                    project='runs/train/WUDD',
                    name=yaml_name,
                    )

# if __name__ == '__main__':
#     model = YOLO('/final/YOLO-SA-FPSC/models/innovations/yolov10n-FPSC.yaml')
#     # model.load('yolov8n.pt') # loading pretrain weights
#     model.train(data='/final/datasets/DUO/DUO.yaml',
#                 cache=False,
#                 imgsz=640,
#                 epochs=300,
#                 batch=16, #32
#                 close_mosaic=0,
#                 workers=4,
#                 # device='0',
#                 optimizer='SGD', # using SGD
#                 # patience=0, # close earlystop
#                 # resume=True, # 断点续训,YOLO初始化时选择last.pt
#                 # amp=False, # close amp
#                 # fraction=0.2,
#                 project='runs/train',
#                 name="yolov10n-FPSC",
#                 )