import src

# 2. 从 src 导入 YOLO
from src import YOLO

if __name__ == '__main__':

    # =========================================================
    # 核心：使用 YAML 初始化模型
    # =========================================================
    # 这会根据 yaml 里的结构搭建网络，并使用【随机权重】初始化
    # 此时模型是一张白纸，没有之前的训练记忆
    model = YOLO('yolov8-rtdetr.yaml', task='detect') 

    # 【可选技巧】如果你没改网络结构，只是想用 yaml 的配置，但又想用预训练权重加速：
    # model = YOLO('my_yolov8.yaml').load('yolov8n.pt') 
    # 注意：这需要下载 yolov8n.pt 放在根目录
    
    # 3. 开始训练
    model.train(
        data='VisDrone.yaml',  # 指向你之前配置好的数据集 yaml
        epochs=100,               # 轮数
        imgsz=640,                # 图像大小
        project='runs/train',     # 保存路径
        workers=0,                # 多线程数
    )