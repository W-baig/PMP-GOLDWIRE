import torch
from models.model import PMPNetPlus
from config_c3d import cfg

from collections import OrderedDict
import os

import torchvision.models as models

# # 加载带有预训练权重的ResNet-50模型
# resnext50_32x4d = models.resnext50_32x4d(pretrained=True).cuda()
# # 将pytorch模型保存为onnx模型时需要输入一个虚拟的Batch，这里随机生成一个
# BATCH_SIZE = 32
# dummy_input=torch.randn(BATCH_SIZE, 3, 224, 224).cuda()
# # 导出为onnx模型
# torch.onnx.export(resnext50_32x4d,
#                   dummy_input, 
#                   '/home/wanghao/Projects/PMP-Net-main-WIRE/exp/output/checkpoints/WEIGHT.onnx',    # 导出的 ONNX 文件名
#                   verbose=False,
#                   input_names=["input"],
#                   output_names=["output"],
#                   keep_initializers_as_inputs=True,
#                   opset_version=14)

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

b = 32
x = torch.randn(b, 2048, 3)
prev_s = (
    torch.normal(mean=0, std=torch.ones((b, 128, 2048), dtype=torch.float, device=device) * 0.01),
    torch.normal(mean=0, std=torch.ones((b, 128, 512), dtype=torch.float, device=device) * 0.01),
    torch.normal(mean=0, std=torch.ones((b, 256, 128), dtype=torch.float, device=device) * 0.01)
)

model = PMPNetPlus(dataset=cfg.DATASET.TRAIN_DATASET)
if torch.cuda.is_available():
    model = model.cuda()
    x = x.cuda()


assert 'WEIGHTS' in cfg.CONST and cfg.CONST.WEIGHTS
# logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
checkpoint = torch.load(cfg.CONST.WEIGHTS)
state_dict = checkpoint['model']

new_state_dict = OrderedDict()
for k, v in state_dict.items():

    name = k[7:]  # remove "module."
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
# # model.to(device)
model.eval()

# # output = model(x)[0][-1]
# # print("pytorch result:", output.shape)

with torch.no_grad():
# #     torch.onnx.export(
# #         model,                       # 要转换的模型
# #         x,                           # 模型的任意一组输入
# #         '/home/wanghao/Projects/PMP-Net-main-WIRE/exp/output/checkpoints/WEIGHT.onnx',    # 导出的 ONNX 文件名
# #         opset_version=12,            # ONNX 算子集版本
# #         input_names=['input'],       # 输入 Tensor 的名称（自己起名字）
# #         output_names=['output']      # 输出 Tensor 的名称（自己起名字）
# #     )

    torch.onnx.export(
        model,
        (x, prev_s),                           # 模型的任意一组输入
        '/home/wanghao/TensorRT-8.6.1.6/bin/WEIGHT.onnx',    # 导出的 ONNX 文件名
        export_params=True,
        input_names=["input1", "input2"], 
        output_names=["output"], 
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        opset_version=14,
        # dynamic_axes={"input": {0: "nBatchSize"}, "output": {0: "nBatchSize"}}
        )



# 验证onnx模型导出成功
import onnx
import onnxruntime as ort
# 读取 ONNX 模型
onnx_model = onnx.load('/home/wanghao/TensorRT-8.6.1.6/bin/WEIGHT.onnx')

for input_tensor in onnx_model.graph.input:
    print(f"Input name: {input_tensor.name}, Type: {input_tensor.type}")
# 检查模型格式是否正确
onnx.checker.check_model(onnx_model)
print('无报错，onnx模型载入成功')

try:
    ort_session = ort.InferenceSession('/home/wanghao/TensorRT-8.6.1.6/bin/WEIGHT.onnx',
                                        providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
    print("\nONNX Runtime 模型的输入定义:")
    for input in ort_session.get_inputs():
        print(f"Name: {input.name}")
        print(f"Type: {input.type}")
        print(f"Shape: {input.shape}")
except Exception as e:
    print(f"Error loading model with onnxruntime: {e}")
