"""Verify a torch build can actually run on this GPU: arch list AND a real fp16 kernel.

torch.cuda.is_available() returning True proved nothing on the V100s: the cu128 wheel has
no sm_70 kernels and failed on the first matmul.
"""
import sys

import torch

arch = torch.cuda.get_arch_list()
cap = torch.cuda.get_device_capability(0)
need = f"sm_{cap[0]}{cap[1]}"
print(f"torch {torch.__version__} cuda {torch.version.cuda} arch {arch} device {need}")
if need not in arch:
    sys.exit(f"FAIL: {need} not in arch list")
x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
y = (x @ x).float()
torch.cuda.synchronize()
ref = (x.float() @ x.float())
err = (y - ref).abs().max().item() / ref.abs().max().item()
conv = torch.nn.functional.conv2d(torch.randn(1, 3, 64, 64, device="cuda", dtype=torch.float16),
                                  torch.randn(8, 3, 3, 3, device="cuda", dtype=torch.float16))
torch.cuda.synchronize()
print(f"fp16 matmul ok, rel err {err:.2e}; fp16 conv2d ok {tuple(conv.shape)}")
if not err < 1e-2:
    sys.exit("FAIL: fp16 matmul result wrong")
print("PASS")
