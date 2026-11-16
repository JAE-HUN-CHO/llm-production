import copy
import torch
import torch.ao.quantization as q

# --- demo setup (the book assumes `model_fp32` and `dataset` already exist) ---
if "model_fp32" not in globals():
    model_fp32 = torch.nn.Sequential(
        torch.ao.quantization.QuantStub(),
        torch.nn.Linear(16, 8),
        torch.nn.ReLU(),
        torch.ao.quantization.DeQuantStub(),
    )
if "dataset" not in globals():
    dataset = [torch.randn(4, 16) for _ in range(8)]
torch.backends.quantized.engine = "qnnpack"
# -------------------------------------------------------------------------------

# deep copy the original model as quantization is done in place
model_to_quantize = copy.deepcopy(model_fp32)
model_to_quantize.eval()

# get mappings - note use “qnnpack” for ARM and “fbgemm” for x86 CPU
model_to_quantize.qconfig = q.get_default_qconfig("qnnpack")

# prepare
prepared_model = q.prepare(model_to_quantize)

# calibrate - you’ll want to use representative (validation) data.
with torch.inference_mode():
    for x in dataset:
        prepared_model(x)

# quantize
model_quantized = q.convert(prepared_model)

print(model_quantized)
