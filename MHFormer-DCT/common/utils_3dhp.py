from torch.autograd import Variable
import torch
import numpy as np
import glob
import os

def get_variable(split, target):
    num = len(target)
    var = []
    if split == 'train':
        for i in range(num):
            temp = Variable(target[i], requires_grad=False).contiguous().type(torch.cuda.FloatTensor)
            var.append(temp)
    else:
        for i in range(num):
            temp = Variable(target[i]).contiguous().cuda().type(torch.cuda.FloatTensor)
            var.append(temp)

    return var


def mpjpe_cal(predicted, target):
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))


def denormalize(pred, seq):
    out = pred.cpu().numpy()
    for idx in range(out.shape[0]):
        if seq[idx] in ['TS5', 'TS6']:
            res_w, res_h = 1920, 1080
        else:
            res_w, res_h = 2048, 2048
        out[idx, :, :, :2] = (out[idx, :, :, :2] + np.array([1, res_h / res_w])) * res_w / 2
        out[idx, :, :, 2:] = out[idx, :, :, 2:] * res_w / 2
    out = out - out[..., 0:1, :]
    return torch.tensor(out).cuda()


def Load_model(args, model):
    model_paths = sorted(glob.glob(os.path.join(args.previous_dir, '*.pth')))
    model_path = model_paths[0]
    print(model_path)

    pre_dict = torch.load(model_path)
    model_dict = model.state_dict()

    state_dict = {k: v for k, v in pre_dict.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

