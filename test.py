import torch


xpu_available = torch.xpu.is_available()

if xpu_available:
    print("Intel XPU is available!")
else:
    print("Intel XPU is not available.")
