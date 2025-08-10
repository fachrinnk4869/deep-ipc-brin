from mmseg.models import build_segmentor
from mmseg.apis import inference_model, init_model, show_result_pyplot
import torch

# Load the config for SETR
# Adjust path if needed
config_file = './mmsegmentation/configs/setr/setr_vit-l_naive_8xb1-80k_cityscapes-768x768.py'
checkpoint_file = './setr_naive_vit-large_8x1_768x768_80k_cityscapes.pth'

# Load model config
model = init_model(config_file, checkpoint_file)
encoder = model.backbone

# Test the encoder with a random input tensor

def to_device(data, device):
    """Move tensor(s) to the specified device."""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the device
model = encoder.to(device)

input_tensor = torch.randn(1, 3, 768, 768)
# Move the input tensor to the device
input_tensor = to_device(input_tensor, device)

# Example input (batch_size, channels, height, width)

features = encoder(input_tensor)
print([f.shape for f in features])  # Print shapes of feature maps
