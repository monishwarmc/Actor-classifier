import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

def create_writer(exp_name : str,
                  model_name: str,
                  extra: str = None) -> torch.utils.tensorboard.writer.SummaryWriter :
    """
    creating a function to initialize the summary writer with names and time stamps
    """
    
    timestamp = datetime.now().strftime("%d-%m-%Y__%H-%M-%S")
    
    if extra :
        log_dir = os.path.join("runs", timestamp, exp_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, exp_name, model_name)
        
    
    return SummaryWriter(log_dir=log_dir)