import torch
from config import config
import torchvision
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import List, Dict
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_preds(
    model : torch.nn.Module,
    transform: torchvision.transforms.Compose,
    classes: List[str],
    data: str | Path,
    device: str = device,
    use_facecrop : bool = False
) -> Dict[str, any]:
    """
    returns the predictions of the model
    """
    DATADIR = Path(data)
    if not DATADIR.exists():
        raise(f"{data} directory not found")
    
    valid_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = [i for i in DATADIR.iterdir() if i.suffix.lower() in valid_ext]
    
    task = "binary" if len(classes) <= 2 else "multiclass"
    
    model.eval()
    preds = {}
    model.to(device)
    with torch.inference_mode():
        for img in images:
            name = img.stem.replace(".*", "")
            img_pil = Image.open(img).convert("RGB")
            if use_facecrop:

                img_cv = cv2.cvtColor(
                    np.array(img_pil),
                    cv2.COLOR_RGB2BGR
                )

                gray = cv2.cvtColor(
                    img_cv,
                    cv2.COLOR_BGR2GRAY
                )

                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5
                )

                if len(faces) > 0:

                    x, y, w, h = max(
                        faces,
                        key=lambda f: f[2] * f[3]
                    )

                    margin = int(0.3 * max(w, h))

                    x1 = max(0, x - margin)
                    y1 = max(0, y - margin)
                    x2 = min(img_cv.shape[1], x + w + margin)
                    y2 = min(img_cv.shape[0], y + h + margin)

                    img_cv = img_cv[y1:y2, x1:x2]

                    img_pil = Image.fromarray(
                        cv2.cvtColor(
                            img_cv,
                            cv2.COLOR_BGR2RGB
                        )
                    )

            img = transform(img_pil).unsqueeze(0).to(device)
            
            y_logits = model(img)
            y_preds = torch.sigmoid(y_logits) if task == "binary" else torch.softmax(y_logits,dim=1)
            preds[name] = {"prob" : round(y_preds.max(dim=1).values.item() * 100, 2),
                                                                       "class" : classes[y_preds.argmax(1).item()]}
            
        
    return preds


def save_model(model : torch.nn.Module, 
               dir : str,
               model_name : str) -> None:
    """
    saves the given model to the given path
    """
    
    path = Path(dir)
    path.mkdir(parents=True, exist_ok=True)
    
    if not model_name.endswith((".pt",".pth")):
        model_name += ".pth"
        
    save_path = path / model_name
    
    print(f"[INFO]saving model to {save_path}")
        
    torch.save(obj=model.state_dict(), f=save_path)
    
def plot_confusion_matrix(
    model : torch.nn.Module,
    dataloader : torch.utils.data.DataLoader,
    classes : List[str],
    name : str
):
    """
    plots confusion matrix
    """
    
    model.eval()
    model.to(device)
    y_preds = []
    y_true = []
    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device)
            y_logits = model(X)
            y_preds.extend(
                torch.softmax(y_logits, dim=1)
                .argmax(1)
                .cpu()
                .numpy()
            )

            y_true.extend(
                y.cpu().numpy()
            )
    cm = confusion_matrix(y_true=y_true, y_pred=y_preds)
    
    plt.figure(figsize=(100,100))
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classes)
    
    disp.plot(
    xticks_rotation=90,
    cmap="Blues",
    values_format="d"
    )
    
    for text in disp.text_.ravel():
        if text is not None:
            text.set_fontsize(6)

    # Axis label size
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    
    plt.tight_layout()
    
    Path("results").mkdir(exist_ok=True)

    plt.savefig(
        "results/"+name+"_confusion_matrix.png",
        dpi=600,
        bbox_inches="tight"
    )
    plt.close()