from torchvision import transforms
from torchvision.models import ResNet18_Weights, ConvNeXt_Tiny_Weights, ConvNeXt_Base_Weights
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

class Config:
    def __init__(self):
        self.LR = 1e-4
        self.TRAIN_SIZE = 0.95
        self.SEED = 42
        self.BATCH_SIZE = 8
        self.HIDDEN = 64
        self.EPOCHS = 30
        self.NUM_WORKERS = 6
        self.NUM_CLASSES = 13
        self.CLASSES = ['Ajith_Kumar', 'Anushka_Shetty', 'Dhanush', 'Nayanthara', 'Rajinikanth', 'Samantha', 'Simbu', 'Sivakarthikeyan', 'Suriya', 'Tamannah', 'Trisha', 'Vijay', 'Vijay_Sethupathi']
        self.DEVICE = device
        self.TRAIN_TRANSFORM = ConvNeXt_Base_Weights.DEFAULT.transforms()

        self.TEST_TRANSFORM = ConvNeXt_Base_Weights.DEFAULT.transforms()
        self.DATA_DIR = "cropped_faces"

            
config = Config()