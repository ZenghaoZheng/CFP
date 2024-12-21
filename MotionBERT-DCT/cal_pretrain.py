import torch
from lib.model.model_action import *
from lib.model.DSTformer import DSTformer
from lib.model.DSTformer_main import DSTformer_main

def _test():
    import warnings
    warnings.filterwarnings('ignore')
    b, h, t, j, c = 1, 2, 243, 17, 3
    random_x = torch.ones((b, t, j, c))
    model = DSTformer_main(dim_in=3, dim_out=3, dim_feat=512, dim_rep=512,
                      depth=5, num_heads=8, mlp_ratio=2,
                        num_joints=17, maxlen=243,
                        qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                        att_fuse=True)
    # model = ActionNet(backbone, dim_rep=512, num_classes=60, dropout_ratio=0., version='class', hidden_dim=2048, num_joints=17)

    # chk_filename = 'checkpoint/pretrain/MB_release/latest_epoch.bin'
    # checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    # pretrained_dict = {}
    # for k, v in checkpoint['model_pos'].items():
    #     pretrained_dict[k[7:]] = v
    # model.load_state_dict(pretrained_dict)
    model = model.cuda()
    random_x = random_x.cuda()
    model.eval()

    # model_params = 0
    # for parameter in model.parameters():
    #     model_params = model_params + parameter.numel()
    # print(f"Model parameter #: {model_params:,}")
    # print(f"Model FLOPS #: {2 * profile_macs(model, random_x):,}")

    import time
    num_iterations = 1000

    # Measure the inference time for 'num_iterations' iterations
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            out = model(random_x)
    end_time = time.time()
    # Calculate the average inference time per iteration
    average_inference_time = (end_time - start_time) / num_iterations
    # Calculate FPS
    fps = 1.0 / average_inference_time
    print(f"FPS: {fps}")
    assert out.shape == (b, t, j, 3), f"Output shape should be {b}x{t}x{j}x3 but it is {out.shape}"


if __name__ == '__main__':
    _test()