import torch

# 1. Standardowy 3D U-Net
from monai.networks.nets import UNet

# 2. Attention U-Net 3D
from monai.networks.nets import AttentionUnet

# 3. ResUNet 3D (w MONAI implementowany jako VNet lub customowy UNet z blokami Residual)
from monai.networks.nets import VNet

# 4. Swin UNETR 3D (Vision Transformer dla danych 3D)
from monai.networks.nets import SwinUNETR


def run_unet_3d_examples():
    # Przykladowy tensor wejsciowy (Syntetyczny skan 3D, np. MRI/CT)
    # Format PyTorch 3D: [Batch_Size, Channels, Depth, Height, Width]
    # np. 1 pacjent, 1 kanał (monochromatyczny CT), rozmiar 96x96x96
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.randn(1, 1, 96, 96, 96).to(device)

    print(f"Kształt wejściowy: {input_tensor.shape}\n" + "-" * 40)

    # ----------------------------------------------------
    # 1. Standardowy 3D U-Net
    # ----------------------------------------------------
    model_unet3d = UNet(
        spatial_dims=3,  # 3D
        in_channels=1,  # 1 kanał wejściowy (np. CT)
        out_channels=2,  # 2 kanały wyjściowe (np. tło + segmentowany narząd)
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    out_unet = model_unet3d(input_tensor)
    print(f"1. UNet 3D output: {out_unet.shape}")

    # ----------------------------------------------------
    # 2. Attention U-Net 3D
    # ----------------------------------------------------
    # Wykorzystuje mechanizmy uwagi (Attention Gates) w połączeniach skróconych (skip-connections)
    model_attention = AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
    ).to(device)

    out_attention = model_attention(input_tensor)
    print(f"2. Attention U-Net 3D output: {out_attention.shape}")

    # ----------------------------------------------------
    # 3. ResUNet 3D (reprezentowany tu przez VNet - klasyk ResUNet w 3D)
    # ----------------------------------------------------
    # Łączy architekturę UNet z blokami rezydualnymi (Residual Blocks) z ResNeta
    model_resunet = VNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
    ).to(device)

    out_resunet = model_resunet(input_tensor)
    print(f"3. ResUNet 3D (VNet) output: {out_resunet.shape}")

    # ----------------------------------------------------
    # 4. Swin UNETR 3D (Vision Transformer)
    # ----------------------------------------------------
    # Łączy cechy lokalne (enkoder Swin Transformer 3D) z dekoderem typu UNet
    model_swin = SwinUNETR(
        img_size=(96, 96, 96),  # Wymagane podanie dokładnego rozmiaru wejścia dla Swin Transformer
        in_channels=1,
        out_channels=2,
        feature_size=48,  # Rozmiar cech początkowych
        use_checkpoint=True  # Oszczędność pamięci VRAM GPU
    ).to(device)

    out_swin = model_swin(input_tensor)
    print(f"4. Swin UNETR 3D output: {out_swin.shape}")


if __name__ == "__main__":
    run_unet_3d_examples()