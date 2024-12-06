from torchinfo import summary
from torchvision.models import vit_b_16

model = vit_b_16(pretrained=True).to("cpu")
summary(model, input_size=(1, 3, 224, 224), depth=2)
