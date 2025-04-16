from pred_head import Seg_Head 
import torch


x= torch.randn((1 , 256 , 64 , 64))

model = Seg_Head()

model.eval()

# Forward pass
with torch.no_grad():
    out = model(x)

print("Output shape:", out.shape)  # Expected: [1, 1, 512, 512]