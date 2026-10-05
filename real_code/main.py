import pandas as pd
from monai.data import Dataset, DataLoader

import config
from runner import train_and_eval_model
from transformer import get_transforms


def main():
    # Zastąp te ścieżki swoimi plikami, gdy będziesz je posiadać
    # Skrypt akceptuje dowolną liczbę pacjentów w liście ze ścieżkami .nii.gz
    train_files = [
        {"image": "data/train/p1_flair.nii.gz", "label": "data/train/p1_mask.nii.gz"},
        {"image": "data/train/p2_flair.nii.gz", "label": "data/train/p2_mask.nii.gz"},
    ]
    val_files = [
        {"image": "data/val/p3_flair.nii.gz", "label": "data/val/p3_mask.nii.gz"},
    ]

    train_tf, val_tf = get_transforms(config.CONFIG["patch_size"])

    train_ds = Dataset(data=train_files, transform=train_tf)
    val_ds = Dataset(data=val_files, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=config.CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    models_to_evaluate = [
        "UNet3D",
        "AttentionUNet3D",
        "ResUNet3D",
        "SwinUNETR"
    ]

    results = []

    # Wykonanie eksperymentów porównawczych
    for model_name in models_to_evaluate:
        res = train_and_eval_model(model_name, train_loader, val_loader, config.CONFIG)
        results.append(res)

    # Zapis i wyświetlenie wyników porównania do pracy magisterskiej
    df_results = pd.DataFrame(results)
    df_results.to_csv("wyniki_magisterka_porownanie.csv", index=False)

    print("\n==========================================")
    print(" PODSUMOWANIE EKSPERYMENTÓW DO MAGISTERKI ")
    print("==========================================")
    print(df_results.to_string(index=False))


if __name__ == "__main__":
    # Uwaga: Uruchomienie powższej funkcji wymaga rzeczywistych plików .nii.gz w katalogach.
    print("Skrypt przygotowany pod strukturę magisterską.")