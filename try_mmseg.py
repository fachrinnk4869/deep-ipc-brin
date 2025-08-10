import os
from mmseg.apis import init_segmentor
import torch
import torchvision.transforms as T
import requests

# Helper function to download files


def download_file(url, dest_path):
    if not os.path.exists(dest_path):
        print(f"Downloading {url} to {dest_path}...")
        response = requests.get(url, stream=True)
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded: {dest_path}")
    else:
        print(f"File already exists: {dest_path}")


# Paths to download from mmsegmentation GitHub
CONFIG_URL = 'https://raw.githubusercontent.com/open-mmlab/mmsegmentation/refs/heads/main/configs/setr/setr_vit-l_naive_8xb1-80k_cityscapes-768x768.py'
CHECKPOINT_URL = 'https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_pup_512x512_160k_ade20k/setr_pup_512x512_160k_ade20k_20220125_214425-55b6af6e.pth'

# Local file paths
CONFIG_PATH = 'setr_pup_512x512_160k_ade20k.py'
CHECKPOINT_PATH = 'setr_pup_512x512_160k_ade20k.pth'

# Download files if necessary
download_file(CONFIG_URL, CONFIG_PATH)
download_file(CHECKPOINT_URL, CHECKPOINT_PATH)

# Load mmsegmentation model


def load_mmseg_model(config_path, checkpoint_path, device='cuda:0'):
    """Load mmsegmentation model with config and checkpoint."""
    model = init_segmentor(config_path, checkpoint_path, device=device)
    model.eval()  # Set to evaluation mode
    return model

# Step 2: Wrap mmsegmentation model for PyTorch compatibility


class MMSEGPyTorchWrapper(torch.nn.Module):
    def __init__(self, mmseg_model):
        super(MMSEGPyTorchWrapper, self).__init__()
        self.mmseg_model = mmseg_model

    def forward(self, x):
        # mmsegmentation expects a dictionary input
        result = self.mmseg_model.forward_dummy(x)
        return result

# Step 3: Preprocessing function


def preprocess_image(image, target_size=(512, 512)):
    """Preprocess image to match mmsegmentation input pipeline."""
    transform = T.Compose([
        T.Resize(target_size),
        T.ToTensor(),
        T.Normalize(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375]),
    ])
    return transform(image).unsqueeze(0)


# Step 4: Main function
if __name__ == "__main__":
    # Load mmsegmentation model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mmseg_model = load_mmseg_model(CONFIG_PATH, CHECKPOINT_PATH, device=device)

    # Wrap the model in PyTorch-compatible class
    pytorch_model = MMSEGPyTorchWrapper(mmseg_model).to(device)

    # Example: Load an image
    from PIL import Image
    # Replace with your image path
    image_path = '/home/fachrikid/[sample_driving_data]/code/code_gather_data/dataset/datasetx/dataset_0/test_routes/sunny/2024-11-13_route01/camera/front/rgb/1731485021_0000000117.png'
    input_image = Image.open(image_path).convert("RGB")

    # Preprocess image
    input_tensor = preprocess_image(input_image).to(device)

    # Forward pass
    with torch.no_grad():
        output = pytorch_model(input_tensor)

    # Output shape
    print("Output shape:", output.shape)

    # Postprocess (e.g., argmax for segmentation map)
    seg_map = output.argmax(dim=1).squeeze(0).cpu().numpy()

    # Visualize segmentation map
    import matplotlib.pyplot as plt
    plt.imshow(seg_map, cmap='jet')
    plt.title("Segmentation Map")
    plt.axis('off')
    plt.show()
