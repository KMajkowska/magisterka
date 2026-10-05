from monai.networks.nets import UNet, AttentionUnet, VNet, SwinUNETR
import torch

#tutaj trzeba by jeszcze dodać aby się parametry zmieniały
def build_model(model_name: str, config: dict) -> torch.nn.Module:
    """Zwraca żądaną architekturę na podstawie nazwy."""
    in_c = config["in_channels"]
    out_c = config["out_channels"]

    if model_name == "UNet3D":
        return UNet(
            spatial_dims=3,
            in_channels=in_c,
            out_channels=out_c,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2
        )
    elif model_name == "AttentionUNet3D":
        return AttentionUnet(
            spatial_dims=3,
            in_channels=in_c,
            out_channels=out_c,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2)
        )
    elif model_name == "ResUNet3D":
        return VNet(
            spatial_dims=3,
            in_channels=in_c,
            out_channels=out_c
        )
    elif model_name == "SwinUNETR":
        return SwinUNETR(
            in_channels=in_c,
            out_channels=out_c,
            feature_size=48,
            use_checkpoint=True
        )
    else:
        raise ValueError(f"Nieznany model: {model_name}")