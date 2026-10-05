import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ToTensord,
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import AttentionUnet
from monai.losses import DiceCELoss, DiceFocalLoss # -> DiceFocalLoss podobno super do MRI

# 1. Definicja ścieżek do gotowych plików 3D (MRI oraz masek z adnotacjami)
data_dicts = [
    {"image": "pacjent1_FLAIR.nii.gz", "label": "pacjent1_maska.nii.gz"},
    {"image": "pacjent2_FLAIR.nii.gz", "label": "pacjent2_maska.nii.gz"},
]

# 2. Pipeline transformacji wyciągający dane z plików NIfTI do PyTorch Tensor 3D
transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),  # Wymiar [C, H, W, D]
    ToTensord(keys=["image", "label"]),
])

# 3. Przygotowanie DataLoader
dataset = Dataset(data=data_dicts, transform=transforms)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# 4. Inicjalizacja modelu Attention U-Net 3D dla stwardnienia rozsianego
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AttentionUnet(
    spatial_dims=3,
    in_channels=1,  # 1 sekwencja MRI (np. FLAIR)
    out_channels=1,  # 1 kanał wyjściowy (skala prawdopodobieństwa: zmiana vs tło)
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
).to(device)

# 5. Funkcja straty optymalna dla małych zmian SM (Dice + Cross Entropy)
loss_function = DiceCELoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 6. Przykładowa pojedyncza pętla treningowa
model.train()
for batch_data in train_loader:
    inputs = batch_data["image"].to(device)
    labels = batch_data["label"].to(device)

    optimizer.zero_grad()
    outputs = model(inputs)

    # Obliczenie straty dla małych ognisk demielinizacyjnych
    loss = loss_function(outputs, labels)
    loss.backward()
    optimizer.step()

    print(f"Loss dla pacjenta: {loss.item():.4f}")
    break