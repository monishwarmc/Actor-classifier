from torch import nn
from torchvision.models import resnet18, ResNet18_Weights, convnext_tiny, ConvNeXt_Tiny_Weights, convnext_base, ConvNeXt_Base_Weights


def resnet(suffix:int, num_classes : int)->nn.Module:
    if suffix == 18:
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)

        for param in model.parameters():
            param.requires_grad = True
            
        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes
        )

        for param in model.fc.parameters():
            param.requires_grad = True
            
        return model
    raise ValueError(
    f"No ResNet model found with suffix {suffix}"
    )
    
def convnext(suffix:str, num_classes : int) -> nn.Module:
    if suffix == "tiny":
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        model = convnext_tiny(weights=weights)
        
        model.classifier[2] = nn.Linear(
        model.classifier[2].in_features,
        num_classes
        )
        return model
    if suffix == "base":
        weights = ConvNeXt_Base_Weights.DEFAULT
        model = convnext_base(weights=weights)
        
        model.classifier[2] = nn.Linear(
        model.classifier[2].in_features,
        num_classes
        )
        return model
        
    raise ValueError(
    f"No ResNet model found with suffix {suffix}"
    )



class TinyVGG(nn.Module):
    """
    Creating TinyVGG model with CNN explainer architechture (https://poloclub.github.io/cnn-explainer/) 
    but image size will be 224 * 224
    """
    
    def __init__(
        self,
        in_ : int,
        out : int,
        hidden : int
    ):
        super().__init__()
        self.block1 = nn.Sequential(
            # input image (224, 224, 3)
            nn.Conv2d(in_, hidden, 3), # (222, 222, hidden)
            nn.ReLU(),# (222, 222, hidden)
            nn.Conv2d(hidden, hidden, 3),# (220, 220, hidden)
            nn.ReLU(),# (222, 222, hidden)
            nn.MaxPool2d(2, 2)# (110, 110, hidden)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3),# (108, 108, hidden)
            nn.ReLU(),# (108, 108, hidden)
            nn.Conv2d(hidden, hidden, 3),# (106, 106, hidden)
            nn.ReLU(),# (106, 106, hidden)
            nn.MaxPool2d(2,2)# (53, 53, hidden)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), # (hidden*53*53)
            nn.Linear(hidden*53*53, out) # (10)
        )
    
    def forward(self, x):
        return self.classifier(self.block2(self.block1(x)))
    
    