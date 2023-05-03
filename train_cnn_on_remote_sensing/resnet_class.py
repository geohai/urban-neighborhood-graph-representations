import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# load the pre-trained ResNet-18 model
class RSResNet18(torch.nn.Module):
    def __init__(self, num_classes=39, train=True):
        super().__init__()
        model = torchvision.models.resnet18(pretrained=True)
        num_features = model.fc.in_features # set the number of classes to 8 (i.e. MillionAID has 8 classes)
        model.fc = nn.Linear(num_features, num_classes)
        self.resnet18 = model
        self.transforms =  torchvision.transforms.Compose([
#             torchvision.transforms.RandomHorizontalFlip(p=0.5),
#             torchvision.transforms.Resize((224, 224)),
#             torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        if train == True:
            self.resnet18.train()
        else:
            self.resnet18.eval()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_pred = self.resnet18(x)
        return y_pred.argmax(dim=1)
    
    def __call__(self, x):
        x = self.transforms(x)
        x = x.unsqueeze(0)
        y_pred = self.resnet18(x)
        y_pred = torch.softmax(y_pred, dim=1)
        
#         print(y_pred.shape)
        return y_pred
        
    def modify_output_layer(self, out_features):
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_features, out_features)
        print('Layer is modified')
        
        
    