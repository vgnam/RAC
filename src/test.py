import torch as th
import numpy as np


a = th.randn((2, 3, 4, 5))
b = th.ones((2, 3, 1), dtype=th.long)
print(th.gather(a, dim=2, index=b.unsqueeze(dim=-1).expand(-1, -1, -1, 5)).shape)