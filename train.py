import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# BILIBILI UP 魔傀面具
# 训练参数官方详解链接：https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings:~:text=a%20training%20run.-,Train%20Settings,-The%20training%20settings

# 指定显卡和多卡训练问题 统一都在<YOLOV8V10配置文件.md>下方常见错误和解决方案。
# 训练过程中loss出现nan，可以尝试关闭AMP，就是把下方amp=False的注释去掉。

#bash: nohup python train.py > train.log 2>&1 & 

# if __name__ == '__main__':
#     # model = YOLO('ultralytics/cfg/models/v10/yolov10n.yaml')
#     # model.load('yolov8n.pt') # loading pretrain weights
#     # ['yolov8m','yolov8n','yolov8s','yolov5','yolov6','yolov9s','yolov9m','yolov10m','yolov10n','yolov10s']
#     #['sa','s','sa-fpsc-n','sa-fpsc-m','sa-fpsc-b','sa-fpsc-s']
#     for yaml_name in ['yolov8m','yolov8n','yolov8s','yolov5','yolov6','yolov9s','yolov9m','yolov10m','yolov10n','yolov10s']:
#         model = YOLO(f'/final/YOLO-SA-FPSC/models/base/{yaml_name}.yaml')
#         model.train(data='/final/datasets/DUO_Gamma_Clahe/DUO_GC.yaml',
#                     cache=False,
#                     imgsz=640,
#                     epochs=300,
#                     batch=16, #32
#                     close_mosaic=0,
#                     workers=4,
#                     # device='0',
#                     optimizer='SGD', # using SGD
#                     # patience=0, # close earlystop
#                     # resume=True, # 断点续训,YOLO初始化时选择last.pt
#                     # amp=False, # close amp
#                     # fraction=0.2,
#                     project='runs/train',
#                     name=yaml_name+"-DUO_GC",
#                     )

if __name__ == '__main__':
    model = YOLO('/final/YOLO-SA-FPSC/models/innovations/yolov10n-FPSC.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='/final/datasets/DUO/DUO.yaml',
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
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name="yolov10n-FPSC",
                )