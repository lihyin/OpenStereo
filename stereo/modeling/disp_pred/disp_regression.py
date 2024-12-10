# @Time    : 2023/11/9 17:41
# @Author  : zhangchenming
import numpy as np
import torch
import torch.nn as nn


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    
    # Modify for SiMa: replace sum by mean() / size()
    # return torch.sum(x * disp_values, 1, keepdim=True)
    return torch.mean(x * disp_values, 1, keepdim=True) / disp_values.size(1)
