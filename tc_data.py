import unittest
import os
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
# Replace with your dataset file import
from config import GlobalConfig
from data import WHILL_Data


class TestCustomDataset(unittest.TestCase):
    def setUp(self):
        # Set up test directory and dummy images
        self.test_dir = "test_images/"
        os.makedirs(self.test_dir, exist_ok=True)
        for i in range(5):  # Create 5 dummy images
            dummy_image = np.random.randint(
                0, 256, (360, 640, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(self.test_dir,
                        f"image_{i}.jpg"), dummy_image)
        self.config = GlobalConfig()
        # Initialize dataset
        self.dataset = WHILL_Data(data_root=self.config.train_dir,
                                  conditions=self.config.train_conditions, config=self.config)

    def tearDown(self):
        # Clean up test directory
        for f in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, f))
        os.rmdir(self.test_dir)

    def test_length(self):
        # Test the length of the dataset
        self.assertEqual(len(self.dataset), 654)

    def test_getitem(self):
        # Test __getitem__ functionality
        for i in range(len(self.dataset)):
            data = self.dataset[i]
            seg = data['segs'][0][0]
            print("halo", seg.shape)
            # Check that seg is a torch tensor
            self.assertIsInstance(seg, torch.Tensor)

            # Check that there are no zero elements in seg
            self.assertTrue(torch.all(seg > 0),
                            f"Segmentation map contains 0 at index {i}")

            print(f"Tested sample {i}: Segmentation map has no zeros.")

        print("All __getitem__ tests passed.")


# Run the tests
if __name__ == "__main__":
    unittest.main()
