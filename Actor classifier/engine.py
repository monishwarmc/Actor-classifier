import torch, utils
from torch import nn
from tqdm.auto import tqdm
from typing import Tuple, Dict, List
from torchmetrics import Accuracy
import time

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_step(
    dataloader : torch.utils.data.DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim,
    num_classes : int,
    device: str = device,
    ) -> Tuple[float, float]:
    """
    trains the given model and returns train loss and train accuracy
    """
    
    model.train()
    
    train_loss, train_acc =  0.0, 0.0
    
    if num_classes <= 2:
        task = "binary"
        acc = Accuracy(task="binary").to(device)
    else :
        task = "multiclass"
        acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    
    for X, y in tqdm(dataloader, desc="Training", total=len(dataloader), leave=False):
        X = X.to(device).to(torch.torch.float)
        y = y.to(device).to(torch.torch.float) if task == "binary" else y.to(torch.long).to(device)
        y_logits = model(X)
        loss = loss_fn(y_logits, y)
        train_loss += loss.item()
        train_acc += acc(y_logits, y).item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    
    return train_loss, train_acc

def test_step(
    dataloader : torch.utils.data.DataLoader,
    model : nn.Module,
    loss_fn : nn.Module,
    num_classes : int,
    device : str = device,
) -> Tuple[float, float]:
    """
    tests the given model and returns test loss and test acuuracy
    """
    
    model.eval()
    
    test_loss, test_acc = 0.0, 0.0
    
    if num_classes <= 2:
        task = "binary"
        acc = Accuracy(task="binary").to(device)
    else :
        task = "multiclass"
        acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    
    
    for X, y in tqdm(dataloader, desc="Testing", total=len(dataloader), leave=False):
        X = X.to(device).to(torch.float)
        y = y.to(device).to(torch.float) if task == "binary" else y.to(torch.long).to(device)
        with torch.inference_mode():
            y_logits = model(X)
            test_loss += loss_fn(y_logits, y).item()
            test_acc += acc(y_logits, y).item()
    
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    
    return test_loss, test_acc
        
        
def train_model(
    train_dataloader : torch.utils.data.DataLoader,
    test_dataloader : torch.utils.data.DataLoader,
    model : nn.Module,
    loss_fn : nn.Module,
    optimizer : torch.optim,
    epochs: int,
    num_classes : int,
    model_name : str = None,
    device: str = device,
    writer : torch.utils.tensorboard.writer.SummaryWriter = None
) -> Dict[str, List]:
    """
    trains and tests the given model then returns a dict of results as
    {
        "train_loss" : [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    """
    
    results = {
        "train_loss" : [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    model.to(device)
    
    st = time.time()
    
    best_acc = 0
    patience = 3
    counter = 0
    
    for epoch in tqdm(range(epochs), total=epochs, desc="Epochs", leave=False):
        
        train_loss, train_acc = train_step(
            dataloader=train_dataloader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            num_classes=num_classes
        )
        
        test_loss, test_acc = test_step(
            dataloader=test_dataloader,
            model=model,
            loss_fn=loss_fn,
            device=device,
            num_classes=num_classes
        )
        
        if test_acc > best_acc:
            best_acc = test_acc
            counter = 0
            if model_name:
                utils.save_model(
                    model=model,
                    model_name=model_name,
                    dir="models"
                )

        else:
            counter += 1

        if counter >= patience:
            print(
                f"Early stopping at epoch {epoch+1}"
            )
            break
        
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        
        print(
            f"Epoch: {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.4f}"
        )
        
        if writer:
            writer.add_scalars(main_tag='loss', 
                            tag_scalar_dict={"train loss" : train_loss,"test loss" : test_loss},
                            global_step=epoch) 
            writer.add_scalars(main_tag='accuracy',
                            tag_scalar_dict={"train accuracy" : train_acc, "test accuracy" : test_acc},
                            global_step=epoch)
    if writer:        
        writer.close()
    
    end = time.time()
    
    print(f"Exec. time = {(end-st):.3f} sec")
        
    return results
        
        