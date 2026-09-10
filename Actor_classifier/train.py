import torch
import dataSetup, engine, tensor_board, models
from config import config
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def train():

    train_dataloader, test_dataloader, classes = dataSetup.create_dataloader(
        data=config.DATA_DIR,
        train_transform=config.TRAIN_TRANSFORM,
        test_transform = config.TEST_TRANSFORM,
        train_size=config.TRAIN_SIZE,
        seed=config.SEED,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )

    model = models.convnext(
        "base",len(classes)
    )
    
    train_dataset = train_dataloader.dataset

    targets = [label for _, label in train_dataset]

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(targets),
        y=targets
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32
    ).to(config.DEVICE)
    
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LR)

    writer = tensor_board.create_writer(
        exp_name="last exp",
        model_name="Convnext base",
        extra="lr=0.0001, Adam"
    )

    engine.train_model(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=config.EPOCHS,
        num_classes=len(classes),
        writer=writer,
        model_name="ConvNextBase"
    )

    
    print(classes)

if __name__ == "__main__":
    train()