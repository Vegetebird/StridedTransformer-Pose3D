import torch
import torch.nn as nn
from einops import rearrange
from model.block.vanilla_transformer_encoder import Transformer
from model.block.strided_transformer_encoder import Transformer as Transformer_reduce

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(2*args.n_joints, args.channel, kernel_size=1),
            nn.BatchNorm1d(args.channel, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25)
        )

        self.Transformer = Transformer(args.layers, args.channel, args.d_hid, length=args.frames)
        self.Transformer_reduce = Transformer_reduce(len(args.stride_num), args.channel, args.d_hid, \
            length=args.frames, stride_num=args.stride_num)
        
        self.fcn = nn.Sequential(
            nn.BatchNorm1d(args.channel, momentum=0.1),
            nn.Conv1d(args.channel, 3*args.out_joints, kernel_size=1)
        )

        self.fcn_1 = nn.Sequential(
            nn.BatchNorm1d(args.channel, momentum=0.1),
            nn.Conv1d(args.channel, 3*args.out_joints, kernel_size=1)
        )

    def forward(self, x):
        B, F, J, C = x.shape
        x = rearrange(x, 'b f j c -> b (j c) f').contiguous()

        x = self.encoder(x) 
        x = x.permute(0, 2, 1).contiguous()

        x = self.Transformer(x) 

        x_VTE = x
        x_VTE = x_VTE.permute(0, 2, 1).contiguous()
        x_VTE = self.fcn_1(x_VTE) 
        x_VTE = rearrange(x_VTE, 'b (j c) f -> b f j c', j=J).contiguous()

        x = self.Transformer_reduce(x) 
        
        x = x.permute(0, 2, 1).contiguous() 
        x = self.fcn(x) 
        x = rearrange(x, 'b (j c) f -> b f j c', j=J).contiguous() 
        
        return x, x_VTE




