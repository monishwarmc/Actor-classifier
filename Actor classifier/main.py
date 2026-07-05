from config import config
import torch, models, utils, dataSetup
from pathlib import Path
import pandas as pd

def main():
    train_dataloader, test_dataloader, classes = dataSetup.create_dataloader(
            data=config.DATA_DIR,
            train_transform=config.TRAIN_TRANSFORM,
            test_transform = config.TEST_TRANSFORM,
            train_size=config.TRAIN_SIZE,
            seed=config.SEED,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS
        )

    model = models.convnext("base", len(classes))
    DATA_PATH = Path("testimg")

    model.load_state_dict(torch.load(f="models/ConvNextBase.pth"))

    def preds():
        preds = utils.get_preds(
            model=model,
            transform=config.TEST_TRANSFORM,
            classes=classes,
            data=DATA_PATH,
        )

        df = pd.DataFrame(preds)

        print(df)

    def num_images():
        path = Path("data")
        path = list(path.glob("*"))
        data = {}
        for p in path:
            l = len(list(p.glob("*")))
            data[p.stem] = l
        df = pd.DataFrame(
            data.items(),
            columns=["Actor", "num_images"]
        )
        print(df)
        
    def plot():
        utils.plot_confusion_matrix(
            model=model,
            dataloader=test_dataloader,
            classes=classes,
            name = "test_data"
        )
        utils.plot_confusion_matrix(
            model=model,
            dataloader=train_dataloader,
            classes=classes,
            name = "train_data"
        )
    preds()
        
if __name__ == "__main__":
    main()