from ultralytics import YOLO
if __name__ == '__main__':
    # 加载一个模型权重
    # model = YOLO('yolov8n.pt')  # 加载官方的模型权重作评估
    model = YOLO('/final/SA-FPSC/runs/train/UDO/yolo11/weights/best.pt')  # 加载自定义的模型权重作评估
 
   	# 评估
    metrics = model.val()  # 不需要传参，这里定义的模型会自动在训练的数据集上作评估
    print(metrics.box.map)  # map50-95
    print(metrics.box.map50)  # map50
    print(metrics.box.map75)  # map75
    print(metrics.box.maps)  # 包含每个类别的map50-95列表
    # print(metrics.box.all_ap) #F1值？
    # print(metrics.box.f1) #F1值？