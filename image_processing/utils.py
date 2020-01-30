import torch
from torch.autograd import Variable
from torchvision import transforms, models
from torch import nn

from PIL import Image
model = models.resnet18(pretrained=True)
layer = model._modules.get('avgpool')
model.eval()

scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

modules = list(model.children())[:-1]
model = nn.Sequential(*modules)

def get_vector(image_name):
    img = Image.open(image_name)
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    feature = model(t_img)
    feature = feature.squeeze(0).squeeze(1).squeeze(1)

    return feature.detach().numpy()
