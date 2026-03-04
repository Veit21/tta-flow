##############################################################
#
#   This script defines a wrapper class for the torchcfm UNet
#
##############################################################

# Imports
import torch

from torchcfm.models.unet.unet import UNetModel


#----------------------------------------------------------------------------
# Global variables

NUM_CLASSES = 1000


#----------------------------------------------------------------------------
# Wrapper class for the UNet model defined in the torchcfm library
# See Tong et al.

class UNetModelWrapper(UNetModel):
    def __init__(
        self,
        dim,
        num_channels,
        num_res_blocks,
        num_out_channels=1,     # NOTE: Added this "out_channel" argument for more flexibility in I/O
        channel_mult=None,
        learn_sigma=False,
        class_cond=False,
        num_classes=NUM_CLASSES,
        use_checkpoint=False,
        attention_resolutions="16",
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        dropout=0,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        remove_noise_conditioning=False     # Effectively sets the time scalar/batch to zero before being passed to the embedding. See "Is Noise Conditioning Necessary for Denoising Generative Models?", He et al.
    ):
        """Dim (tuple): (C, H, W)"""
        self.remove_noise_conditioning = remove_noise_conditioning
        image_size = dim[-1]
        if channel_mult is None:
            if image_size == 512:
                channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
            elif image_size == 256:
                channel_mult = (1, 1, 2, 2, 4, 4)
            elif image_size == 128:
                channel_mult = (1, 1, 2, 3, 4)
            elif image_size == 64:
                channel_mult = (1, 2, 3, 4)
            elif image_size == 32:
                channel_mult = (1, 2, 2, 2)
            elif image_size == 28:
                channel_mult = (1, 2, 2)
            else:
                raise ValueError(f"unsupported image size: {image_size}")
        else:
            channel_mult = list(channel_mult)

        attention_ds = []
        for res in attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))

        return super().__init__(
            image_size=image_size,
            in_channels=dim[0],
            model_channels=num_channels,
            out_channels=(num_out_channels if not learn_sigma else dim[0] * 2),              # (dim[0] if not learn_sigma else dim[0] * 2),
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=(num_classes if class_cond else None),
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
        )

    def forward(self, t, x, y=None, *args, **kwargs):
        if self.remove_noise_conditioning:
            t = torch.zeros_like(t)
        return super().forward(t, x, y=y)