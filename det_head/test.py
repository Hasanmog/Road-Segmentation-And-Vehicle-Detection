import torch
from detection import Det_Head

model = Det_Head()

model.eval()

x = torch.randn((1 , 256 , 64 , 64))

output = model(x)
print("output shape" , output.shape)