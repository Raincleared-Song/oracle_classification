import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict
from exceptions.exceptions import InvalidBackboneError
from models.vit import VisionTransformer, interpolate_pos_embed


class ViTSimCLR(nn.Module):

    model_path_map = {
        'first':  '../ALBEF/output/tra_image_reconstruct_vit_all_mk25_ranoi_cls_lr4_upd/checkpoint_65.pth',
        'second': '../ALBEF/output/tra_finetune_single_mlm_p0_load_image_mk25_unrec_new/checkpoint_45.pth',
    }

    def __init__(self, model_path, image_res):
        super(ViTSimCLR, self).__init__()
        self.model_path = model_path
        self.image_res = image_res
        self.backbone = self._get_basemodel()

    def _get_basemodel(self):
        model = VisionTransformer(
            img_size=self.image_res, patch_size=16, embed_dim=768, depth=12, num_heads=12,
            in_chans=3, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True)
        state_dict = checkpoint["model"]
        pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], model)
        state_dict['pos_embed'] = pos_embed_reshaped
        msg = model.load_state_dict(state_dict, strict=False)
        print('ViT Initialization:', msg)
        if self.model_path == 'none':
            return model
        if self.model_path not in ViTSimCLR.model_path_map:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: none, first or second")
        pth_path = ViTSimCLR.model_path_map[self.model_path]
        print('Loading checkpoint from:', pth_path)
        states = torch.load(pth_path)['model']
        prefix = 'visual_encoder.'
        vit_states = OrderedDict({key[len(prefix):]: val for key, val in states.items() if key.startswith(prefix)})
        model.load_state_dict(vit_states)
        return model

    def forward(self, x):
        return self.backbone(x)[:, 0, :]
