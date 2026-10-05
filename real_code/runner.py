import time
import torch
from typing import Dict, List, Tuple

from monai.data import DataLoader

from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from models import build_model


def train_and_eval_model(
        model_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict
) -> Dict[str, float]:
    """Trenuje model i zwraca metryki końcowe."""
    print(f"\n==========================================")
    print(f" Start Treningu Modelu: {model_name}")
    print(f"==========================================")

    device = torch.device(config["device"])
    model = build_model(model_name, config).to(device)

    # DiceFocalLoss – idealne rozwiązanie przy małych ognisku choroby
    loss_fn = DiceFocalLoss(sigmoid=True, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-5)

    # Metryki badawcze
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    best_val_dice = -1.0
    start_time = time.time()

    for epoch in range(1, config["epochs"] + 1):
        # Phase 1: Trening
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Phase 2: Walidacja
        model.eval()
        dice_metric.reset()

        with torch.no_grad():
            for val_batch in val_loader:
                val_inputs = val_batch["image"].to(device)
                val_labels = val_batch["label"].to(device)

                # Prognowanie
                val_outputs = model(val_inputs)

                # Binaryzacja progiem 0.5
                val_preds = (torch.sigmoid(val_outputs) > 0.5).float()

                # Obliczanie Dice Score
                dice_metric(y_pred=val_preds, y=val_labels)

            val_dice = dice_metric.aggregate().item()
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                # Zapis najlepszego modelu na potrzeby pracy
                torch.save(model.state_dict(), f"best_{model_name}.pt")

        print(f"Epoch [{epoch:02d}/{config['epochs']:02d}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Dice: {val_dice:.4f}")

    total_time = time.time() - start_time
    print(f"Zakończono {model_name} w czasie: {total_time/60:.2f} min. Najlepszy Val Dice: {best_val_dice:.4f}")

    return {
        "Model": model_name,
        "Best_Val_Dice": best_val_dice,
        "Training_Time_Sec": round(total_time, 2)
    }