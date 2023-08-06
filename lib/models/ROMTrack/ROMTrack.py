import torch
from torch import nn

from .Transformer import build_transformer
from .head import build_box_head

from lib.utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy


class ROMTrack(nn.Module):
    """ This is the base class for ROMTrack """
    def __init__(self, encoder, head, head_type='CORNER'):
        super().__init__()
        self.encoder = encoder

        self.head = head
        self.head_type = head_type

    # stage1
    def forward_stage1(self, template, inherent_template, search, run_score_head=False, gt_bboxes=None):
        if template.dim() == 5:
            template = template.squeeze(0)
        if inherent_template.dim() == 5:
            inherent_template = inherent_template.squeeze(0)
        if search.dim() == 5:
            search = search.squeeze(0)

        for i in range(12):
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

        template1 = template
        with torch.no_grad():
            for i in range(12):
                template1, inherent_template, search1 = self.encoder.forward_train_generate_variation_token(template1, inherent_template, search1, i)

        for i in range(12):
            template, search2 = self.encoder.forward_train_fuse_vt(template, search2, i)

        out, outputs_coord_new = self.forward_box_head(search2)

        return out, outputs_coord_new

    def set_online(self, ini_it_vt):
        if ini_it_vt.dim() == 5:
            ini_it_vt = ini_it_vt.squeeze(0)

        for i in range(12):
            ini_it_vt = self.encoder.set_online(ini_it_vt, i)

    def forward_test(self, template, search):
        if template.dim() == 5:
            template = template.squeeze(0)
        if search.dim() == 5:
            search = search.squeeze(0)

        for i in range(12):
            template, search = self.encoder.forward_test(template, search, i)

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

        for i in range(12):
            template, inherent_template, search = self.encoder.forward_profile(template, inherent_template, search, i)

        self.forward_box_head(search)


def build_vit(cfg):
    encoder = build_transformer(cfg.DATA.TEMPLATE.SIZE, cfg.DATA.SEARCH.SIZE)
    head = build_box_head(cfg)
    model = ROMTrack(encoder, head, cfg.MODEL.HEAD_TYPE)
    return model