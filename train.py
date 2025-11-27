import src

# 2. 从 src 导入 YOLO
from src import YOLO

if __name__ == '__main__':

    # =========================================================
    # 核心：使用 YAML 初始化模型
    # =========================================================
    # 这会根据 yaml 里的结构搭建网络，并使用【随机权重】初始化
    # 此时模型是一张白纸，没有之前的训练记忆
    model = YOLO('yolov8.yaml') 

    # 【可选技巧】如果你没改网络结构，只是想用 yaml 的配置，但又想用预训练权重加速：
    # model = YOLO('my_yolov8.yaml').load('yolov8n.pt') 
    # 注意：这需要下载 yolov8n.pt 放在根目录
    
    # 3. 开始训练
    model.train(
        data='VisDrone.yaml',  # 指向你之前配置好的数据集 yaml
        epochs=100,               # 轮数
        imgsz=1280,                # 图像大小
        batch=8,                  # 批次
        workers=0,                # Windows 必须为 0
        device='0',               # 显卡
        project='runs/train',     # 保存路径
        amp=False                 # 如果遇到显存溢出或报错，可以尝试设为 False
    )