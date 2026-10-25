import cv2
import numpy as np
import torch

n = torch.arange(12).reshape(2, 2, 3)
n = n.copy().unsqueeze(0)
print(n)

