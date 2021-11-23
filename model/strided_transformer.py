import torch
import torch.nn as nn
from model.block.vanilla_transformer_encoder import Transformer
from model.block.strided_transformer_encoder import Transformer as Transformer_reduce

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        layers, channle, d_hid, length  = args.layers, args.channle, args.d_hid, args.frames
        stride_num = args.stride_num
        self.num_joints_in, self.num_joints_out = args.n_joints, args.out_joints

        self.encoder = nn.Sequential(
            nn.Conv1d(2*self.num_joints_in, channle, kernel_size=1),
            nn.BatchNorm1d(channle, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25)
        )

        self.Transformer = Transformer(layers, channle, d_hid, length=length)
        self.Transformer_reduce = Transformer_reduce(len(stride_num), channle, d_hid, \
            length=length, stride_num=stride_num)
        
        self.fcn = nn.Sequential(
            nn.BatchNorm1d(channle, momentum=0.1),
            nn.Conv1d(channle, 3*self.num_joints_out, kernel_size=1)
        )

        self.fcn_1 = nn.Sequential(
            nn.BatchNorm1d(channle, momentum=0.1),
            nn.Conv1d(channle, 3*self.num_joints_out, kernel_size=1)
        )

    def forward(self, x):
        x = x[:, :, :, :, 0].permute(0, 2, 3, 1).contiguous() 
        x_shape = x.shape

        x = x.view(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1).contiguous() 

        x = self.encoder(x) 

        x = x.permute(0, 2, 1).contiguous()
        x = self.Transformer(x) 

        x_VTE = x
        x_VTE = x_VTE.permute(0, 2, 1).contiguous()
        x_VTE = self.fcn_1(x_VTE) 

        x_VTE = x_VTE.view(x_shape[0], self.num_joints_out, -1, x_VTE.shape[2])
        x_VTE = x_VTE.permute(0, 2, 3, 1).contiguous().unsqueeze(dim=-1)

        x = self.Transformer_reduce(x) 
        x = x.permute(0, 2, 1).contiguous() 
        x = self.fcn(x) 

        x = x.view(x_shape[0], self.num_joints_out, -1, x.shape[2])
        x = x.permute(0, 2, 3, 1).contiguous().unsqueeze(dim=-1)
        
        return x, x_VTE




