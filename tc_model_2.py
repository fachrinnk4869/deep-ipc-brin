import torch
from xr14.model import xr14

import torchvision.models as models


class Config:
    def __init__(self):
        self.n_fmap_b3 = [[32, 16], [16, 24], [24, 40], [40, 80], [80, 112]]
        self.n_fmap_b1 = [[32, 16], [16, 24], [24, 40], [40, 80], [80, 112]]
        self.n_class = 10
        self.coverage_area = 50
        self.res_resize = [240, 320]
        self.seq_len = 5
        self.pred_len = 5
        self.turn_KP = 1.0
        self.turn_KI = 0.0
        self.turn_KD = 0.0
        self.turn_n = 20
        self.speed_KP = 1.0
        self.speed_KI = 0.0
        self.speed_KD = 0.0
        self.speed_n = 20
        self.max_throttle = 1.0
        self.err_angle_mul = 1.0
        self.des_speed_mul = 1.0
        self.wheel_radius = 1.0
        self.clip_delta = 1.0
        self.min_act_thrt = 0.1
        self.cw_pid = [0.5, 0.5]
        self.cw_mlp = [0.5, 0.5]
        self.ctrl_opt = "one_of"
        self.SEG_CLASSES = {'colors': [(255, 0, 0), (0, 255, 0), (0, 0, 255)]}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = Config()


def test_forward():
    model = xr14(config, device).to(device)
    rgbs = [torch.randn(1, 3, 240, 320).to(device)
            for _ in range(config.seq_len)]
    pt_cloud_xs = [torch.randn(1, 240, 320).to(device)
                   for _ in range(config.seq_len)]
    pt_cloud_zs = [torch.randn(1, 240, 320).to(device)
                   for _ in range(config.seq_len)]
    rp1 = torch.randn(1, 2).to(device)
    rp2 = torch.randn(1, 2).to(device)
    velo_in = torch.randn(1, 2).to(device)

    segs_f, pred_wp, steering, throttle, sdcs = model(
        rgbs, pt_cloud_xs, pt_cloud_zs, rp1, rp2, velo_in)

    assert len(segs_f) == config.seq_len
    assert pred_wp.shape == (1, config.pred_len, 2)
    assert isinstance(steering, torch.Tensor)
    assert isinstance(throttle, torch.Tensor)
    assert len(sdcs) == config.seq_len


if __name__ == "__main__":
    test_forward()
    print("All tests passed!")
