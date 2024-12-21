import torch
import torch.nn as nn
from einops import rearrange
from model.module_cfp.trans import Transformer as Transformer_encoder
from model.module_cfp.trans_hypothesis import Transformer as Transformer_hypothesis
from model.module_cfp.xboost import xboost
class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        ## MHG      层标准化
        self.norm_1 = nn.LayerNorm(args.frames)
        self.norm_2 = nn.LayerNorm(args.frames)
        self.norm_3 = nn.LayerNorm(args.frames)

        self.Transformer_encoder_1 = Transformer_encoder(4, args.frames, args.frames*2, length=4*args.n_joints, h=9)
        self.Transformer_encoder_2 = Transformer_encoder(4, args.frames, args.frames*2, length=4*args.n_joints, h=9)
        self.Transformer_encoder_3 = Transformer_encoder(4, args.frames, args.frames*2, length=4*args.n_joints, h=9)

        ## Embedding
        if args.frames > 27:
            self.embedding_1 = nn.Conv1d(4*args.n_joints, args.channel, kernel_size=1)  # args.channel默认是512
            self.embedding_2 = nn.Conv1d(4*args.n_joints, args.channel, kernel_size=1)
            self.embedding_3 = nn.Conv1d(4*args.n_joints, args.channel, kernel_size=1)
        else:
            self.embedding_1 = nn.Sequential(
                nn.Conv1d(4*args.n_joints, args.channel, kernel_size=1),
                nn.BatchNorm1d(args.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

            self.embedding_2 = nn.Sequential(
                nn.Conv1d(4*args.n_joints, args.channel, kernel_size=1),
                nn.BatchNorm1d(args.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

            self.embedding_3 = nn.Sequential(
                nn.Conv1d(4*args.out_joints, args.channel, kernel_size=1),
                nn.BatchNorm1d(args.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

        ## SHR & CHI
        self.Transformer_hypothesis = Transformer_hypothesis(args.layers, args.channel, args.d_hid, length=args.frames,
                                                             dct_ratio=args.dct_ratio, freq_prune_layer=args.freq_prune_layer)
        
        ## Regression
        self.regression = nn.Sequential(
            nn.BatchNorm1d(args.channel*3, momentum=0.1),
            nn.Conv1d(args.channel*3, 3*args.out_joints, kernel_size=1)
        )

    def forward(self, x):
        B, F, J, C = x.shape
        x_boost = xboost(x)
        x = torch.cat((x, x_boost), dim=-1)
        x = rearrange(x, 'b f j c -> b (j c) f').contiguous()   # rearrange函数用于对x的维度进行变化

        ## MHG
        x_1 = x   + self.Transformer_encoder_1(self.norm_1(x))
        x_2 = x_1 + self.Transformer_encoder_2(self.norm_2(x_1)) 
        x_3 = x_2 + self.Transformer_encoder_3(self.norm_3(x_2))
        
        ## Embedding        contiguous，深度拷贝一份数据给对应的变量
        x_1 = self.embedding_1(x_1).permute(0, 2, 1).contiguous() 
        x_2 = self.embedding_2(x_2).permute(0, 2, 1).contiguous()
        x_3 = self.embedding_3(x_3).permute(0, 2, 1).contiguous()

        ## SHR & CHI
        x = self.Transformer_hypothesis(x_1, x_2, x_3) 

        ## Regression
        x = x.permute(0, 2, 1).contiguous() 
        x = self.regression(x) 
        x = rearrange(x, 'b (j c) f -> b f j c', j=J).contiguous()

        return x


if __name__ == '__main__':
    import argparse

    args = argparse.ArgumentParser().parse_args()
    args.layers, args.channel, args.d_hid, args.frames = 3, 512, 1024, 9
    args.n_joints, args.out_joints = 17, 17
    args.dct_ratio = 0.6
    args.freq_prune_layer = [0, 1]
    input_2d = torch.rand(1, args.frames, 17, 2)

    with torch.no_grad():
        model = Model(args)
        model.eval()

        model_params = 0
        for parameter in model.parameters():
            model_params += parameter.numel()
        print('INFO: Trainable parameter count:', model_params/ 1000000)

        print(input_2d.shape, 1)
        output = model(input_2d)
        print(output.shape, 2)

    from thop import profile
    from thop import clever_format
    macs, params = profile(model, inputs=(input_2d, ))
    print('macs: ', macs/1000000, 'params: ', params/1000000)
    macs, params = clever_format([macs*2, params], "%.3f")
    print('flops: ', macs, 'params: ', params)



