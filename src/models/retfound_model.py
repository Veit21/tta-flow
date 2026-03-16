###########################################################################
#
# Contains model definitions for embedding images to latent space
#
###########################################################################

# Imports
from functools import partial

import timm.models.vision_transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, resize_input=True, normalize_input=True, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        self.resize_input = resize_input
        self.normalize_input = normalize_input

        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    # NOTE: This forward function only returns feature embeddings, omitting the final layer!
    def forward(self, x):
        if self.resize_input:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        
        if self.normalize_input:
            x_mean = x.mean(dim=(0, 2, 3), keepdim=True)
            x_std = x.std(dim=(0, 2, 3), keepdim=True)
            x = (x - x_mean)/x_std                      # Scale from [0, 1] to mean=0, std=1 for each channel. Normalize over B,H,W
        # print(f"Shape: {x.shape}, Min: {x.min():.2f}, Max: {x.max():.2f}, Mean: {x.mean(dim=(0, 2, 3))}, Std: {x.std(dim=(0, 2, 3))}")

        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1,keepdim=True)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]           # NOTE: Take only the embedding of cls token!

        return outcome


def RETFound_mae(**kwargs) -> nn.Module:
    """Constructs RETFound_mae model and loads the pretrained weights from huggingface_hub.
    Specifically loads the model card "RETFound_mae_natureOCT".
    In the forward pass, the input images are rescaled to 224x224 and Z-Score normalized automatically.
    For more information, see the HF hub: "https://huggingface.co/YukunZhou/RETFound_mae_natureOCT",
    or the publication "https://www.nature.com/articles/s41586-023-06555-x"

    Returns:
        nn.Module: The RETFound model and 
    """

    # Construct model
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    # Load pretrained weights - NOTE: Might need login to HF!
    chkpt_dir   = hf_hub_download(repo_id="YukunZhou/RETFound_mae_natureOCT", filename="RETFound_mae_natureOCT.pth")
    checkpoint  = torch.load(chkpt_dir, weights_only=False)
    msg         = model.load_state_dict(checkpoint['model'], strict=False)

    return model