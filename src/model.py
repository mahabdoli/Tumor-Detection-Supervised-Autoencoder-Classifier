import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F


def build_model(pretrained=True, fine_tune=True, num_classes=2):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    else:
        print('[INFO]: Not loading pre-trained weights')

    model_ft = models.squeezenet1_0(pretrained=pretrained)
    for param in model_ft.parameters():
      param.requires_grad = False
    model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model_ft.num_classes = num_classes

    return model_ft
    

class SupervisedAE(nn.Module):
  def __init__(self):
    super(SupervisedAE, self).__init__()

    self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, stride=(1,1), kernel_size=(3,3))
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1))
    self.fc_1 = nn.Linear(in_features=64*28*28, out_features = 256)
    self.fc_2 = nn.Linear(in_features=256, out_features=32)

    self.fc_3 = nn.Linear(in_features=32,out_features=256)
    self.fc_4 = nn.Linear(in_features=256,out_features=50176)
    self.conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1))
    self.conv4 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=(3,3), stride=(1,1))

    self.softmax = nn.Linear(in_features=32, out_features=2)

  def forward(self, x):
    xe = F.relu(self.conv1(x))
    xe = F.relu(self.conv2(xe))        
    shp = [xe.shape[0],xe.shape[1],xe.shape[2],xe.shape[3]]
    xe = xe.view(-1,shp[1]*shp[2]*shp[3])
    xe = F.relu(self.fc_1(xe))
    xe = F.relu(self.fc_2(xe))
    
    xd = F.relu(self.fc_3(xe))
    xd = F.relu(self.fc_4(xd))
    xd = torch.reshape(xd,(shp[0],shp[1],shp[2],shp[3]))
    xd = F.relu(self.conv3(xd))
    # xd = F.upsample(xd,30)
    x_hat = F.relu(self.conv4(xd))
    
    y_hat = self.softmax(xe)
    
    return y_hat,x_hat











