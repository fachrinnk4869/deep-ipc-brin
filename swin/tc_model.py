import unittest
import torch
from swin import swin
from config import GlobalConfig as Config


class TestSKGESWIN(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.device = torch.device('cpu')
        self.model = swin(self.config, self.device)
        self.h, self.w = self.config.res_resize

    def test_init(self):
        self.assertEqual(self.model.config, self.config)
        self.assertEqual(self.model.gpu_device, self.device)

    def test_rgb_encoder(self):
        # Assuming RGB_Encoder is a method of skgeswin
        input_tensor = torch.randn(
            self.config.batch_size, 3, self.h, self.w)  # Example input
        output = self.model.RGB_encoder(input_tensor)
        self.assertIsInstance(output, torch.Tensor)
        # Add more assertions based on expected output shape and values

    def test_forward(self):
        batch_size = 1
        rgbs = [torch.randn(batch_size, 3, self.h, self.w).to(self.device)
                for _ in range(self.config.seq_len)]
        pt_cloud_xs = [torch.randn(batch_size, self.h, self.w).to(self.device)
                       for _ in range(self.config.seq_len)]
        pt_cloud_zs = [torch.randn(batch_size, self.h, self.w).to(self.device)
                       for _ in range(self.config.seq_len)]
        rp1 = torch.randn(batch_size, 2).to(self.device)
        rp2 = torch.randn(batch_size, 2).to(self.device)
        velo_in = torch.randn(batch_size, 2).to(self.device)

        segs_f, pred_wp, steering, throttle, sdcs = self.model(
            rgbs, pt_cloud_xs, pt_cloud_zs, rp1, rp2, velo_in)

        assert len(segs_f) == self.config.seq_len
        # is contigous
        for seg in segs_f:
            assert seg.is_contiguous()
        assert pred_wp.shape == (
            batch_size, self.config.pred_len, 2)
        assert isinstance(steering, torch.Tensor)
        assert isinstance(throttle, torch.Tensor)
        # print(sdcs)
        assert len(sdcs) == self.config.seq_len

    def test_num_params(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        self.assertGreater(total_params, 0)
    # def test_sc_encoder(self):
    #     # Assuming SC_encoder is a method of skgeswin
    #     input_tensor = torch.randn(
    #         4, 20, self.h, self.w)  # Example input
    #     output = self.model.SC_encoder(input_tensor)
    #     self.assertIsInstance(output, torch.Tensor)
        # Add more assertions based on expected output shape and values


if __name__ == '__main__':
    unittest.main()
