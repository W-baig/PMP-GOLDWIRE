import torch
from config_c3d import cfg
from models.model import PMPNetPlus
from collections import OrderedDict
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
x = torch.randn(4, 2048, 3).to(device)
model = PMPNetPlus(dataset=cfg.DATASET.TRAIN_DATASET).to(device)
# if torch.cuda.is_available():
#     model = model.cuda()

assert 'WEIGHTS' in cfg.CONST and cfg.CONST.WEIGHTS
# logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
checkpoint = torch.load(cfg.CONST.WEIGHTS)
state_dict = checkpoint['model']

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]  # remove "module."
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
# model.to(device)
model.eval()

# output = model(x)[0][-1]
# print("pytorch result:", output.shape)

with torch.no_grad():
#     torch.onnx.export(
#         model,                       # 要转换的模型
#         x,                           # 模型的任意一组输入
#         '/home/wanghao/Projects/PMP-Net-main-WIRE/exp/output/checkpoints/WEIGHT.onnx',    # 导出的 ONNX 文件名
#         opset_version=12,            # ONNX 算子集版本
#         input_names=['input'],       # 输入 Tensor 的名称（自己起名字）
#         output_names=['output']      # 输出 Tensor 的名称（自己起名字）
#     )

    torch.onnx.export(
        model,
        x,                           # 模型的任意一组输入
        '/home/wanghao/Projects/PMP-Net-main-WIRE/exp/output/checkpoints/WEIGHT.onnx',    # 导出的 ONNX 文件名
        export_params=True,
        input_names=["input"], 
        output_names=["output"], 
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        opset_version=12,
        dynamic_axes={"input": {0: "nBatchSize"}, "output": {0: "nBatchSize"}})



# 验证onnx模型导出成功
import onnx
# 读取 ONNX 模型
onnx_model = onnx.load('/home/wanghao/Projects/PMP-Net-main-WIRE/exp/output/checkpoints/WEIGHT.onnx')

for input_tensor in onnx_model.graph.input:
    print(f"Input name: {input_tensor.name}, Type: {input_tensor.type}")
# 检查模型格式是否正确
onnx.checker.check_model(onnx_model)
print('无报错，onnx模型载入成功')
