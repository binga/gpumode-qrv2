import torch
from task import input_t, output_t

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def custom_kernel(data: input_t) -> output_t:
    return torch.geqrf(data)
