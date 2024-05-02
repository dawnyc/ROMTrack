import torch
from torch import nn
import copy

from .Transformer import build_transformer_tiny, build_transformer_small
from .Transformer import build_transformer_base, build_transformer_large, build_transformer_huge
from .head import build_box_head

from lib.utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy


class ROMTrack(nn.Module):
    """ This is the base class for ROMTrack """
    def __init__(self, encoder, head, head_type='CORNER', depth=12):
        super().__init__()
        self.encoder = encoder

        self.head = head
        self.head_type = head_type

        self.depth = depth

    # stage1
    def forward_stage1(self, template, inherent_template, search, run_score_head=False, gt_bboxes=None):
        if template.dim() == 5:
            template = template.squeeze(0)
        if inherent_template.dim() == 5:
            inherent_template = inherent_template.squeeze(0)
        if search.dim() == 5:
            search = search.squeeze(0)

        for i in range(self.depth):
            template, inherent_template, search = self.encoder(template, inherent_template, search, i)

        out, outputs_coord_new = self.forward_box_head(search)

        return out, outputs_coord_new

    # stage2
    def forward_stage2(self, template, inherent_template, search1, search2, run_score_head=False, gt_bboxes=None):
        if template.dim() == 5:
            template = template.squeeze(0)
        if inherent_template.dim() == 5:
            inherent_template = inherent_template.squeeze(0)
        if search1.dim() == 5:
            search1 = search1.squeeze(0)
        if search2.dim() == 5:
            search2 = search2.squeeze(0)

        token_list = []

        template1 = template
        encoder_genvt = copy.deepcopy(self.encoder)
        with torch.no_grad():
            for i in range(self.depth):
                template1, inherent_template, search1, vt_it = encoder_genvt.forward_train_generate_variation_token(template1, inherent_template, search1, i)
                token_list.append(vt_it)

        for i in range(self.depth):
            template, search2 = self.encoder.forward_train_fuse_vt(template, search2, i, token_list[i])

        out, outputs_coord_new = self.forward_box_head(search2)

        return out, outputs_coord_new

    def set_online(self, ini_it_vt):
        if ini_it_vt.dim() == 5:
            ini_it_vt = ini_it_vt.squeeze(0)

        for i in range(self.depth):
            ini_it_vt = self.encoder.set_online(ini_it_vt, i)

    def forward_test(self, template, search):
        if template.dim() == 5:
            template = template.squeeze(0)
        if search.dim() == 5:
            search = search.squeeze(0)

        for i in range(self.depth):
            template, search = self.encoder.forward_test(template, search, i)

        out, outputs_coord_new = self.forward_box_head(search)

        return out, outputs_coord_new

    def set_online_without_vt(self, ini_it):
        if ini_it.dim() == 5:
            ini_it = ini_it.squeeze(0)

        for i in range(self.depth):
            ini_it = self.encoder.set_online_without_vt(ini_it, i)

    def forward_test_without_vt(self, template, search):
        if template.dim() == 5:
            template = template.squeeze(0)
        if search.dim() == 5:
            search = search.squeeze(0)

        for i in range(self.depth):
            template, search = self.encoder.forward_test_without_vt(template, search, i)

        out, outputs_coord_new = self.forward_box_head(search)

        return out, outputs_coord_new

    def forward_box_head(self, search):
        """
        :param search: (b, c, h, w)
        :return:
        """
        if self.head_type == "CORNER":
            # run the corner head
            b = search.size(0)
            outputs_coord = box_xyxy_to_cxcywh(self.head(search))
            outputs_coord_new = outputs_coord.view(b, 1, 4)
            out = {'pred_boxes': outputs_coord_new}
            return out, outputs_coord_new
        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.head(search, None)
            b = search.size(0)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(b, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out, outputs_coord_new
        else:
            raise KeyError

    def forward(self, template, inherent_template, search):
        if template.dim() == 5:
            template = template.squeeze(0)
        if inherent_template.dim() == 5:
            inherent_template = inherent_template.squeeze(0)
        if search.dim() == 5:
            search = search.squeeze(0)

        for i in range(self.depth):
            template, inherent_template, search = self.encoder.forward_profile(template, inherent_template, search, i)

        self.forward_box_head(search)


def build_vit_tiny(cfg):
    encoder = build_transformer_tiny(cfg.DATA.TEMPLATE.SIZE, cfg.DATA.SEARCH.SIZE)
    head = build_box_head(cfg)
    model = ROMTrack(encoder, head, cfg.MODEL.HEAD_TYPE, 12)
    return model


def build_vit_small(cfg):
    encoder = build_transformer_small(cfg.DATA.TEMPLATE.SIZE, cfg.DATA.SEARCH.SIZE)
    head = build_box_head(cfg)
    model = ROMTrack(encoder, head, cfg.MODEL.HEAD_TYPE, 12)
    return model


def build_vit_base(cfg):
    encoder = build_transformer_base(cfg.DATA.TEMPLATE.SIZE, cfg.DATA.SEARCH.SIZE)
    head = build_box_head(cfg)
    model = ROMTrack(encoder, head, cfg.MODEL.HEAD_TYPE, 12)
    return model


def build_vit_large(cfg):
    encoder = build_transformer_large(cfg.DATA.TEMPLATE.SIZE, cfg.DATA.SEARCH.SIZE)
    head = build_box_head(cfg)
    model = ROMTrack(encoder, head, cfg.MODEL.HEAD_TYPE, 24)
    return model


def build_vit_huge(cfg):
    encoder = build_transformer_huge(cfg.DATA.TEMPLATE.SIZE, cfg.DATA.SEARCH.SIZE)
    head = build_box_head(cfg, stride=14)
    model = ROMTrack(encoder, head, cfg.MODEL.HEAD_TYPE, 32)
    return model