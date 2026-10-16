import torch.onnx
import onnxruntime as ort
from model.yolo import YoloBody

# 创建.pth模型

model = YoloBody
# 加载权重
model_path = 'E:/Project/yolov8_yuanma/ultralytics-main/weights/0710_best_epoch_weights.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_statedict = torch.load(model_path, map_location=device)
model.load_state_dict(model_statedict)

model.to(device)
model.eval()

input_data = torch.randn(1, 3, 640, 640, device=device)

# 转化为onnx模型
input_names = ['input']
output_names = ['output']

torch.onnx.export(model, input_data, './model_data/rdd_yolo.onnx', opset_version=9, verbose=True, input_names=input_names, output_names = output_names)