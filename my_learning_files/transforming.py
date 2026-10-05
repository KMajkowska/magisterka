import torch
import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandRotated,
    RandZoomd,
    RandGaussianNoised,
    RandAdjustContrastd,
    ToTensord,
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import AttentionUnet
from monai.losses import DiceFocalLoss


def get_ms_transforms(crop_shape=(96, 96, 96)):
    """
    Tworzy pipeline transformacji optymalizowany pod kątem małych ognisk w 3D MRI.
    """
    train_transforms = Compose(
        [
            # 1. Ładowanie obrazu i maski z plików NIfTI (.nii.gz)
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),

            # 2. Normalizacja intensywności sygnału MRI (zerowa średnia, jednostkowa wariancja)
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),

            # 3. KROK KLUCZOWY: Wycinanie wycinków 3D z preferencją dla zmian (pos/neg sampler)
            # pos=3, neg=1 oznacza, że 75% wyciętych płatków będzie wymuszenie zawierało ognisko MS
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=crop_shape,  # np. (96, 96, 96)
                pos=3,  # Waga dla obszarów z chorobą
                neg=1,  # Waga dla obszarów tła
                num_samples=4,  # Wygeneruj 4 takie wycinki z każdego pacjenta
                image_key="image",
            ),

            # 4. Augmentacja Przestrzenna 3D (odporność na różny układ głowy pacjenta)
            RandRotated(
                keys=["image", "label"],
                range_x=np.pi / 12,  # Obrót max o ~15 stopni
                range_y=np.pi / 12,
                range_z=np.pi / 12,
                prob=0.5,
                mode=["bilinear", "nearest"]  # 'nearest' dla maski, aby nie zaburzyć etykiet!
            ),
            RandZoomd(
                keys=["image", "label"],
                min_zoom=0.9,
                max_zoom=1.1,
                prob=0.3,
                mode=["bilinear", "nearest"]
            ),

            # 5. Augmentacja Intensywności (odporność na szumy i artefakty ze skanera)
            RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.1),
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),

            # 6. Konwersja do Tensora PyTorch
            ToTensord(keys=["image", "label"]),
        ]
    )
    return train_transforms


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Przykładowa lista pacjentów
    data_dicts = [
        {"image": "pacjent1_FLAIR.nii.gz", "label": "pacjent1_maska.nii.gz"},
        {"image": "pacjent2_FLAIR.nii.gz", "label": "pacjent2_maska.nii.gz"},
    ]

    # Inicjalizacja danych
    train_transforms = get_ms_transforms(crop_shape=(96, 96, 96))

    # Uwaga: Użycie Dataset z transformacją RandCropByPosNegLabeld sprawi,
    # że z 1 pacjenta dostaniemy num_samples=4 płatków 3D.
    train_ds = Dataset(data=data_dicts, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)

    # Inicjalizacja Modelu (Attention U-Net 3D)
    model = AttentionUnet(
        spatial_dims=3,
        in_channels=1,  # 1 dla FLAIR (zmień na 2 jeśli dołączysz T1)
        out_channels=1,  # 1 kanał binarnej maski wyjściowej
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
    ).to(device)

    # KROK KLUCZOWY: Funkcja straty optymalna dla bardzo małych obiektów
    # DiceFocalLoss łączy Dice Loss z Focal Loss (kary za trudne, małe przykłady)
    loss_function = DiceFocalLoss(
        sigmoid=True,
        gamma=2.0,  # Skupia uwagę sieci na trudnych do zaklasyfikowania pikselach
        focal_weight=0.5
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    print("Pipeline zoptymalizowany pod MS gotowy. Rozpoczynanie przykładowej pętli...")

    # Pętla treningowa
    model.train()
    for batch_data in train_loader:
        # Płatki 3D wycięte przez sampler
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        print(f"Batch shape: {inputs.shape} | Loss: {loss.item():.4f}")
        break  # Przerwane dla celów demonstracyjnych


if __name__ == "__main__":
    main()