# %%
import torch
import torch.nn as nn

# %%
tensor = torch.rand(10,10)
print(tensor)

# %%
relu = nn.ReLU()

input_layer = torch.rand(1, 4)
weight_1 = torch.rand(4, 6)
weight_2 = torch.rand(6, 2)

hidden_1 = torch.matmul(input_layer, weight_1)
hidden_1_activated = relu(hidden_1)
print(torch.matmul(hidden_1_activated, weight_2))

# %%
import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms



# %%
