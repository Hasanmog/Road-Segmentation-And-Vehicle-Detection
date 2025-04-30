import torch.nn as nn
from model.encoder.cross_attention.crossvit import MultiScaleBlock


class CrossAttnBlock(nn.Module): 
    def __init__(self ,
                 dim, # list
                 patches, 
                 depth, 
                 num_heads,
                 mlp_ratio
                 ):
        super().__init__()
        
        self.multi_scale_block = MultiScaleBlock(
            dim , 
            patches , 
            depth , 
            num_heads , 
            mlp_ratio
        )   
    def forward(self , x):
            
        cross_feat = self.multi_scale_block(x)
        
        return cross_feat

