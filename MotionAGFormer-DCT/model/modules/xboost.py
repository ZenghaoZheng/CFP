import torch

def xboost(x):
    # x.shape: b f j c
    b, f, j, c = x.shape
    y = torch.zeros_like(x, device='cuda:1' if torch.cuda.is_available() else 'cpu')
    y[:, :, 0] = x[:, :, 0]
    y[:, :, 1:4] = (x[:, :, 1:4]+x[:, :, 0:3])/2
    y[:, :, 4:7] = (x[:, :, 4:7]+x[:, :, (0, 4, 5)])/2
    y[:, :, 7:11] = (x[:, :, 7:11]+x[:, :, (0, 7, 8, 9)])/2
    y[:, :, 11:14] = (x[:, :, 11:14]+x[:, :, (8, 11, 12)])/2
    y[:, :, 14:17] = (x[:, :, 14:17]+x[:, :, (8, 14, 15)])/2
    return y


if __name__ == '__main__':
    x = torch.rand((1,243,17,3))
    print(x)
    y = xboost(x)
    print(y)