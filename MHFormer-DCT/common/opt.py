import argparse
import os
import math
import time
import torch

class opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        self.parser.add_argument('--layers', default=3, type=int)
        self.parser.add_argument('--channel', default=512, type=int)
        self.parser.add_argument('--d_hid', default=1024, type=int)
        self.parser.add_argument('--dataset', type=str, default='h36m')
        self.parser.add_argument('-k', '--keypoints', default='cpn_ft_h36m_dbb', type=str)
        self.parser.add_argument('--data_augmentation', type=bool, default=True)
        self.parser.add_argument('--reverse_augmentation', type=bool, default=False)
        self.parser.add_argument('--test_augmentation', type=bool, default=True)
        self.parser.add_argument('--crop_uv', type=int, default=0)
        self.parser.add_argument('--root_path', type=str, default='dataset/')
        self.parser.add_argument('-a', '--actions', default='*', type=str)
        self.parser.add_argument('--downsample', default=1, type=int)
        self.parser.add_argument('--subset', default=1, type=float)
        self.parser.add_argument('-s', '--stride', default=1, type=int)
        self.parser.add_argument('--gpu', default='1', type=str, help='')
        self.parser.add_argument('--train', default=1)
        self.parser.add_argument('--test', action='store_true')
        self.parser.add_argument('--nepoch', type=int, default=20)
        self.parser.add_argument('--batch_size', type=int, default=256, help='can be changed depending on your machine')
        self.parser.add_argument('--lr', type=float, default=1e-3)
        self.parser.add_argument('--lr_decay_large', type=float, default=0.5)
        self.parser.add_argument('--large_decay_epoch', type=int, default=5)
        self.parser.add_argument('--workers', type=int, default=8)
        self.parser.add_argument('-lrd', '--lr_decay', default=0.95, type=float)
        self.parser.add_argument('--frames', type=int, default=351)
        self.parser.add_argument('--pad', type=int, default=175) 
        self.parser.add_argument('--checkpoint', type=str, default='')
        self.parser.add_argument('--previous_dir', type=str, default='')
        self.parser.add_argument('--n_joints', type=int, default=17)
        self.parser.add_argument('--out_joints', type=int, default=17)
        self.parser.add_argument('--out_all', type=int, default=1)
        self.parser.add_argument('--in_channels', type=int, default=2)
        self.parser.add_argument('--out_channels', type=int, default=3)
        self.parser.add_argument('-previous_best_threshold', type=float, default= math.inf)
        self.parser.add_argument('-previous_name', type=str, default='')
        self.parser.add_argument('--dct_ratio', type=float, default=0.6)
        self.parser.add_argument('--freq_prune_layer', default=[0,1])
    def parse(self):
        self.init()         # 初始化参数
        
        self.opt = self.parser.parse_args()         # 将所有的参数赋给opt

        if self.opt.test:          # 若test参数调用时，将train设置为0
            self.opt.train = 0
            
        self.opt.pad = (self.opt.frames-1) // 2

        self.opt.subjects_train = 'S1,S5,S6,S7,S8'      # 添加新的参数subjects_train和subjects_test
        self.opt.subjects_test = 'S9,S11'

        if self.opt.train:
            logtime = time.strftime('%m%d_%H%M_%S_')    # 初始化时间为：年月_时分_秒_
            self.opt.checkpoint = 'checkpoint/' + logtime + '%d'%(self.opt.frames)  # 设置checkpoint的路径和名称
            if not os.path.exists(self.opt.checkpoint):
                os.makedirs(self.opt.checkpoint)

            args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                    if not name.startswith('_'))    # 将opt中所有的name和其对应的属性值组成字典
            file_name = os.path.join(self.opt.checkpoint, 'opt.txt')    # checkpoint对应的txt文件
            with open(file_name, 'wt') as opt_file:
                opt_file.write('==> Args:\n')       # 在txt文件中写入==> Args:
                for k, v in sorted(args.items()):   # 将各个参数及其对应的值写入txt文件
                    opt_file.write('  %s: %s\n' % (str(k), str(v)))
                opt_file.write('==> Args:\n')
       
        return self.opt     # 返回参数opt





        
