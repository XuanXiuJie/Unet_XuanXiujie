import torch

# 检查CUDA是否可用
if torch.cuda.is_available():
    # 获取可用GPU数量
    num_gpus = torch.cuda.device_count()
    print("可用GPU数量：", num_gpus)

    # 获取每个GPU的详细信息
    for i in range(num_gpus):
        device = torch.device(f"cuda:{i}")
        props = torch.cuda.get_device_properties(device)
        print(f"GPU {i}: {props.name}")
else:
    print("没有可用的GPU设备。")
