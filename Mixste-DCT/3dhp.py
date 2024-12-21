import random
import logging
from tqdm import tqdm
import torch.utils.data
from common.utils import *
import torch.optim as optim
from common.camera import *
import common.loss as eval_loss
from common.arguments import parse_args
from dataset.reader.motion_dataset import MPI3DHP,Fusion
from common.h36m_dataset import Human36mDataset
from model.mixste import Model
from model.dct import dct, idct
from common.utils_3dhp import get_variable,mpjpe_cal,denormalize
import scipy.io as scio
args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def save_data_inference(path, data_inference, latest):
    if latest:
        mat_path = os.path.join(path, 'inference_data.mat')
    else:
        mat_path = os.path.join(path, 'inference_data_best.mat')
    scio.savemat(mat_path, data_inference)

def input_augmentation(input_2D, model, joints_left, joints_right):
    N, _, T, J, C = input_2D.shape

    input_2D_flip = input_2D[:, 1]
    input_2D_non_flip = input_2D[:, 0]

    output_3D_flip = model(input_2D_flip)

    output_3D_flip[..., 0] *= -1

    output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]

    output_3D_non_flip = model(input_2D_non_flip)

    output_3D = (output_3D_non_flip + output_3D_flip) / 2

    input_2D = input_2D_non_flip

    return input_2D, output_3D


def train(dataloader, model, optimizer, epoch, opt):
    model.train()
    loss_all = {'loss': AccumLoss()}

    for input_2D, gt_3D in tqdm(dataloader):
        input_2D, gt_3D = input_2D.cuda(),  gt_3D.cuda()

        output_3D = model(input_2D)

        out_target = gt_3D.clone()
        #out_target_freq = dct(out_target.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        #output_3D_freq = dct(output_3D.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        w_mpjpe = torch.tensor([1, 1, 2.5, 2.5, 1, 2.5, 2.5, 1, 1, 1, 1.5, 1.5, 4, 4, 1.5, 4, 4]).cuda()

        loss_w_mpjpe = eval_loss.weighted_mpjpe(output_3D, out_target, w_mpjpe)
        loss_temporal = eval_loss.temporal_consistency(output_3D, out_target, w_mpjpe)
        loss_mean_velocity = eval_loss.mean_velocity(output_3D, out_target, axis=1)
        #loss_freq = eval_loss.mpjpe(output_3D_freq, out_target_freq)
        loss = loss_w_mpjpe + 0.5 * loss_temporal + 2.0 * loss_mean_velocity #+ 0.5 * loss_freq

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        N = input_2D.shape[0]
        loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)

    return loss_all['loss'].avg


def test(actions, test_loader, model, opt):
    model.eval()
    joints_left = [5, 6, 7, 11, 12, 13]
    joints_right = [2, 3, 4, 8, 9, 10]

    data_inference = {}
    error_sum_test = AccumLoss()

    for data in tqdm(test_loader, 0):
        batch_cam, gt_3D, input_2D, seq, scale, bb_box = data

        [input_2D, gt_3D, batch_cam, scale, bb_box] = get_variable('test', [input_2D, gt_3D, batch_cam, scale, bb_box])
        N = input_2D.size(0)

        out_target = gt_3D.clone().view(N, -1, 17, 3)
        out_target[:, :, 14] = 0
        gt_3D = gt_3D.view(N, -1, 17, 3).type(torch.cuda.FloatTensor)

        input_2D, output_3D = input_augmentation(input_2D, model, joints_left, joints_right)

        output_3D = output_3D * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, output_3D.size(1), 17, 3)
        pad = (opt.frames - 1) // 2
        pred_out = output_3D[:, pad].unsqueeze(1)

        pred_out[..., 14, :] = 0
        pred_out = denormalize(pred_out, seq)

        pred_out = pred_out - pred_out[..., 14:15, :]  # Root-relative prediction

        inference_out = pred_out + out_target[..., 14:15, :]  # final inference (for PCK and AUC) is not root relative

        out_target = out_target - out_target[..., 14:15, :]  # Root-relative prediction

        joint_error_test = mpjpe_cal(pred_out, out_target).item()

        for seq_cnt in range(len(seq)):
            seq_name = seq[seq_cnt]
            if seq_name in data_inference:
                data_inference[seq_name] = np.concatenate(
                    (data_inference[seq_name], inference_out[seq_cnt].permute(2, 1, 0).cpu().numpy()), axis=2)
            else:
                data_inference[seq_name] = inference_out[seq_cnt].permute(2, 1, 0).cpu().numpy()

        error_sum_test.update(joint_error_test * N, N)

    for seq_name in data_inference.keys():
        data_inference[seq_name] = data_inference[seq_name][:, :, None, :]

    print(f'Protocol #1 Error (MPJPE): {error_sum_test.avg:.2f} mm')

    return error_sum_test.avg, data_inference


def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    args.new_checkpoint = 'mpi-checkpoint'
    seed = 1126
    create_directory_if_not_exists(args.new_checkpoint)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    dataset_path = args.root_path + 'data_3d_' + args.dataset + '.npz'
    #dataset = Human36mDataset(dataset_path, args)
    actions = define_actions(args.actions)
    common_loader_params = {
        'num_workers': args.workers - 1,
        'pin_memory': True,
        'prefetch_factor': (args.workers - 1) // 3,
        'persistent_workers': True
    }
    if args.train:
        train_dataset = MPI3DHP(args, train=True)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, **common_loader_params)
    test_dataset = Fusion(args, train=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=args.test_batch_size, **common_loader_params)

    model = Model(args)
    model = nn.DataParallel(model)
    model = model.cuda()
    if args.previous_dir != '':
        Load_model(args, model)

    lr = args.lr
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    best_epoch = 0
    loss_epochs = []
    mpjpes = []

    for epoch in range(1, args.nepoch + 1):
        if args.train:
            loss = train(train_dataloader, model, optimizer, epoch, args)
            loss_epochs.append(loss * 1000)

        with torch.no_grad():
            p1, data_inference = test(actions, test_dataloader, model, args)
            mpjpes.append(p1)

        if args.train and p1 < args.previous_best:
            best_epoch = epoch
            args.previous_name = save_model(args, epoch, p1, model, 'model')
            save_data_inference(args.new_checkpoint, data_inference, latest=False)
            args.previous_best = p1

        if args.train:
            logging.info('epoch: %d, lr: %.6f, l: %.4f, p1: %.2f, %d: %.2f' % (
            epoch, lr, loss, p1, best_epoch, args.previous_best))
            print('%d, lr: %.6f, l: %.4f, p1: %.2f, %d: %.2f' % (
            epoch, lr, loss, p1, best_epoch, args.previous_best))

            if epoch % args.lr_decay_epoch == 0:
                lr *= args.lr_decay_large
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.lr_decay_large
            else:
                lr *= args.lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.lr_decay
        else:
            print('p1: %.2f, p2: %.2f' % (p1))
            break

