import glob
import random
import logging
from tqdm import tqdm
import torch.utils.data
import torch.optim as optim
from common.opt import opts
from common.utils import *
from common.load_data_hm36 import Fusion
from common.h36m_dataset import Human36mDataset
from model.mhformer_cfp import Model
from torch import nn

opt = opts().parse()    # 返回从opt文件中opts类的函数parse返回的opt参数对象
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu    # 将opt.gpu（字符串形式）写入环境变量CUDA_VISIBLE_DEVICES
torch.backends.cudnn.enabled = False

def train(opt, actions, train_loader, model, optimizer, epoch):
    return step('train', opt, actions, train_loader, model, optimizer, epoch)

def val(opt, actions, val_loader, model):
    with torch.no_grad():
        return step('test', opt, actions, val_loader, model)

def step(split, opt, actions, dataLoader, model, optimizer=None, epoch=None):
    loss_all = {'loss': AccumLoss()}
    action_error_sum = define_error_list(actions)

    if split == 'train':
        model.train()
    else:
        model.eval()

    for i, data in enumerate(tqdm(dataLoader, 0)):
        batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, cam_ind = data
        input_2D, gt_3D, batch_cam = input_2D.cuda(), gt_3D.cuda(), batch_cam.cuda()
        if split =='train':
            output_3D = model(input_2D) 
        else:
            input_2D, output_3D = input_augmentation(input_2D, model)

        out_target = gt_3D.clone()
        out_target[:, :, 0] = 0

        if split == 'train':
            #b, f, j, c = output_3D.shape
            #middle_index = out_target.shape[1]//2
            #left_count = (f - 1) // 2
            #right_count = f - 1 - left_count
            #out_target = out_target[:, middle_index - left_count : middle_index + right_count + 1, ...]
            #out_target_freq = dct(out_target.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
            #output_3D_freq = dct(output_3D.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

            loss = mpjpe_cal(output_3D, out_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            N = input_2D.size(0)
            loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)

        elif split == 'test':
            #opt.pad = output_3D.shape[1]//2
            output_3D = output_3D[:, opt.pad].unsqueeze(1) 
            output_3D[:, :, 0, :] = 0
            action_error_sum = test_calculation(output_3D, out_target, action, action_error_sum, opt.dataset, subject)

    if split == 'train':
        return loss_all['loss'].avg
    elif split == 'test':
        p1, p2 = print_error(opt.dataset, action_error_sum, opt.train)

        return p1, p2

def input_augmentation(input_2D, model):
    joints_left = [4, 5, 6, 11, 12, 13] 
    joints_right = [1, 2, 3, 14, 15, 16]

    input_2D_non_flip = input_2D[:, 0]
    input_2D_flip = input_2D[:, 1]

    output_3D_non_flip = model(input_2D_non_flip)
    output_3D_flip     = model(input_2D_flip)

    output_3D_flip[:, :, :, 0] *= -1
    output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :] 

    output_3D = (output_3D_non_flip + output_3D_flip) / 2

    input_2D = input_2D_non_flip

    return input_2D, output_3D

if __name__ == '__main__':
    opt.manualSeed = 1          # 在opt中添加随机种子
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.train:
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
            filename=os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO)
    
    root_path = opt.root_path   # rootpath默认为dataset/
    dataset_path = root_path + 'data_3d_' + opt.dataset + '.npz'    # dataset为'h36m'

    dataset = Human36mDataset(dataset_path, opt)    # 使用Human36mDataset类处理数据h36m数据集，返回dataset对象
    actions = define_actions(opt.actions)           # opt.actions是字符串'*'，返回的是动作类型名称构成的字符串列表

    if opt.train:
        train_data = Fusion(opt=opt, train=True, dataset=dataset, root_path=root_path)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size,
                                                       shuffle=True, num_workers=int(opt.workers), pin_memory=True)

    test_data = Fusion(opt=opt, train=False, dataset=dataset, root_path =root_path)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size,
                                                  shuffle=False, num_workers=int(opt.workers), pin_memory=True)
# -------------------------------------------------------------------------------------------------
    model = Model(opt)
    model = nn.DataParallel(model)
    model = model.cuda()

    model_dict = model.state_dict()
    if opt.previous_dir != '':
        model_paths = sorted(glob.glob(os.path.join(opt.previous_dir, '*.pth')))

        for path in model_paths:
            if path.split('/')[-1].startswith('model'):
                model_path = path
                print(model_path)
        pre_dict = torch.load(model_path)

        model_dict = model.state_dict()
        state_dict = {k: v for k, v in pre_dict.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

    lr = opt.lr
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, amsgrad=True)

    for epoch in range(1, opt.nepoch):
        if opt.train: 
            loss = train(opt, actions, train_dataloader, model, optimizer, epoch)
        
        p1, p2 = val(opt, actions, test_dataloader, model)

        if opt.train:
            save_model_epoch(opt.checkpoint, epoch, model)

            if p1 < opt.previous_best_threshold:
                opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, p1, model)
                opt.previous_best_threshold = p1

        if opt.train == 0:
            print('p1: %.2f, p2: %.2f' % (p1, p2))
            break
        else:
            logging.info('epoch: %d, lr: %.7f, loss: %.4f, p1: %.2f, p2: %.2f' % (epoch, lr, loss, p1, p2))
            print('e: %d, lr: %.7f, loss: %.4f, p1: %.2f, p2: %.2f' % (epoch, lr, loss, p1, p2))

        if epoch % opt.large_decay_epoch == 0: 
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opt.lr_decay_large
                lr *= opt.lr_decay_large
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opt.lr_decay
                lr *= opt.lr_decay

    print(opt.checkpoint)








