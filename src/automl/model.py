import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# class CNNModel(nn.Module):
#     def __init__(self, img_height, img_width, channels, num_classes):
#         super(CNNModel, self).__init__()
        
#         self.batch_norm1 = nn.BatchNorm2d(channels)
#         self.conv1 = nn.Conv2d(channels, 64, kernel_size=4, padding=1)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.dropout1 = nn.Dropout(0.1)
        
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=4)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.dropout2 = nn.Dropout(0.3)
        
#         self.flatten = nn.Flatten()
        
#         # Calculate the input size for the fully connected layer
#         self.fc_input_size = self._get_conv_output(img_height, img_width, channels)
        
#         self.fc1 = nn.Linear(self.fc_input_size, 256)
#         self.dropout3 = nn.Dropout(0.5)
        
#         self.fc2 = nn.Linear(256, 64)
#         self.batch_norm2 = nn.BatchNorm1d(64)
        
#         self.fc3 = nn.Linear(64, num_classes)
        
#     def _get_conv_output(self, img_height, img_width, channels):
#         # Create a dummy input to pass through the conv layers to get the output size
#         dummy_input = torch.zeros(1, channels, img_height, img_width)
#         output_feat = self._forward_conv(dummy_input)
#         return int(torch.prod(torch.tensor(output_feat.size())))
    
#     def _forward_conv(self, x):
#         x = self.batch_norm1(x)
#         x = F.relu(self.conv1(x))
#         x = self.pool1(x)
#         x = self.dropout1(x)
        
#         x = F.relu(self.conv2(x))
#         x = self.pool2(x)
#         x = self.dropout2(x)
        
#         return x
    
#     def forward(self, x):
#         x = self._forward_conv(x)
#         x = self.flatten(x)
        
#         x = F.relu(self.fc1(x))
#         x = self.dropout3(x)
        
#         x = F.relu(self.fc2(x))
#         x = self.batch_norm2(x)
        
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1)
    
# class CNNModel(nn.Module):
#     def __init__(self, img_height, img_width, channels, num_classes):
#         super().__init__()

#         # START TODO #############
#         self.conv1 = nn.Conv2d(channels, out_channels=16, kernel_size=3, padding=1)
#         self.conv1_bn = nn.BatchNorm2d(16)
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
#         self.conv2_bn = nn.BatchNorm2d(32, eps=2e-05, momentum=0.05)
#         self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
#         self.conv3_bn = nn.BatchNorm2d(64, eps=2e-05, momentum=0.2)
#         self.pool = nn.MaxPool2d(kernel_size=2)
#         self.fc_input_size = self._get_conv_output(img_height, img_width, channels)
#         self.fc1 = nn.Linear(self.fc_input_size, num_classes)
#         # END TODO #############

#     def forward(self, x) -> torch.Tensor:
#         """
#         Args:
#             x: The input tensor with shape (batch_size, *feature_dim)
#             The input to the network will be a minibatch of data

#         Returns:
#             scores: PyTorch Tensor of shape (N, C) giving classification scores for x
#         """
#         # START TODO #############
#         x = self._forward_conv(x)
#         x = x.view(-1, self.fc_input_size)
#         x = self.fc1(x)
#         return x
#         # END TODO #############
#     # START TODO #############

#     def _get_conv_output(self, img_height, img_width, channels):
#         # Create a dummy input to pass through the conv layers to get the output size
#         dummy_input = torch.zeros(1, channels, img_height, img_width)
#         output_feat = self._forward_conv(dummy_input)
#         return int(torch.prod(torch.tensor(output_feat.size())))
    
#     def _forward_conv(self, x):
#         x = self.conv1(x)
#         x = self.conv1_bn(x)
#         x = F.relu(x)
#         x = self.pool(x)

#         x = self.conv2(x)
#         x = self.conv2_bn(x)
#         x = F.relu(x)
#         x = self.pool(x)

#         x = self.conv3(x)
#         x = self.conv3_bn(x)
#         x = F.relu(x)
#         x = self.pool(x)

#         return x
    
class ModifiedNetBase(nn.Module):

    def __init__(self):
        super().__init__()
        # Load ResNet model pretrained on ImageNet data
        self.net = models.resnet50(pretrained=True)
        # self.net = models.efficientnet_b1(pretrained=True)

    def disable_gradients(self, model) -> None:
        for param in model.parameters():
            param.requires_grad = False
        # END TODO #############

    def forward(self, x) -> torch.Tensor:
        # START TODO #############
        return self.net(x)
        # END TODO #############

class ModifiedNet(ModifiedNetBase):
    def __init__(self, num_classes):
        super().__init__()
        self.disable_gradients(self.net)
        self.net.fc = nn.Linear(512, num_classes)
