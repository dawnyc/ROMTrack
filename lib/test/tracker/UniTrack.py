from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
from lib.models.ROMTrack import build_vit
from lib.test.tracker.tracker_utils import Preprocessor_wo_mask
from lib.utils.box_ops import clip_box
from lib.test.tracker.tracker_utils import vis_attn_maps, vis_feature_maps

from lib.test.utils.hann import hann2d


class ROMTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(ROMTrack, self).__init__(params)
        network = build_vit(params.cfg)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.attn_weights = []

        self.preprocessor = Preprocessor_wo_mask()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // 16
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.frame_id = 0
        if self.debug:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, _, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)

        if self.params.vis_attn == 1:
            self.z_patch = z_patch_arr
            self.iz_patch = z_patch_arr

        template = self.preprocessor.process(z_patch_arr)
        self.template = template
        self.ini_it_vt = template

        with torch.no_grad():
            if self.params.vis_attn == 1:
                attn_weights = []
                features = []
                hooks = []
                for i in range(len(self.network.encoder.blocks)):
                    hooks.append(self.network.encoder.blocks[i].attn.attn_drop.register_forward_hook(
                        lambda self, input, output: attn_weights.append(output)
                    ))
                for i in range(len(self.network.encoder.blocks)):
                    hooks.append(self.network.encoder.blocks[i].attn.proj_drop.register_forward_hook(
                        lambda self, input, output: features.append(output)
                    ))
            self.network.set_online(self.ini_it_vt)
            if self.params.vis_attn == 1:
                for hook in hooks:
                    hook.remove()
                # size:[1, head, 64, 64] - [batch, head, it, it]
                # t-to-t
                vis_attn_maps(attn_weights, q_w=8, k_w=8, skip_len1=0, skip_len2=0, x1=self.iz_patch, x2=self.iz_patch,
                              x1_title="InherentTemplateKV", x2_title="InherentTemplateQ",
                              save_path="vis_attn/it2it/%04d" % self.frame_id, idxs=[(64, 64)])
                # size:[1, 64, 768] - [batch, size, channel]
                # t feature
                vis_feature_maps(features, 8, self.z_patch, save_path="vis_feat/%04d" % self.frame_id)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)

        visarr = []
        # if seq_name == "coin-18":
        # visarr = [1, 10, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533]
        # elif seq_name == "pig-2":
        # visarr = [1, 10, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631]
        # elif seq_name == "rabbit-10":
        # visarr = [1, 10, 2088, 2089, 2090, 2091, 2092, 2093, 2094, 2095, 2096, 2097]

        search = self.preprocessor.process(x_patch_arr)

        with torch.no_grad():
            if self.params.vis_attn == 1 and self.frame_id in visarr:
                attn_weights = []
                features = []
                hooks = []
                for i in range(len(self.network.encoder.blocks)):
                    hooks.append(self.network.encoder.blocks[i].attn.attn_drop.register_forward_hook(
                        lambda self, input, output: attn_weights.append(output)
                    ))
                for i in range(len(self.network.encoder.blocks)):
                    hooks.append(self.network.encoder.blocks[i].attn.proj_drop.register_forward_hook(
                        lambda self, input, output: features.append(output)
                    ))
            out_dict, _ = self.network.forward_test(self.template, search)
            if self.params.vis_attn == 1 and self.frame_id in visarr:
                for hook in hooks:
                    hook.remove()
                # size:[1, head, 64+256, 64+64+64+256] - [batch, head, t+s, t+it+ot+s]
                # t-to-t
                idxs_t = [(64, 64)]
                idxs_s = [(128, 128)]
                vis_attn_maps(attn_weights, q_w=8, k_w=8, skip_len1=0, skip_len2=0, x1=self.z_patch, x2=self.z_patch,
                              x1_title="TemplateKV", x2_title="TemplateQ",
                              save_path="vis_attn/t2t/%04d" % self.frame_id, idxs=idxs_t)
                # t-to-it
                vis_attn_maps(attn_weights, q_w=8, k_w=8, skip_len1=0, skip_len2=64, x1=self.iz_patch, x2=self.z_patch,
                              x1_title="InherentTemplate", x2_title="Template",
                              save_path="vis_attn/t2it/%04d" % self.frame_id, idxs=idxs_t)
                # t-to-s
                vis_attn_maps(attn_weights, q_w=8, k_w=16, skip_len1=0, skip_len2=192, x1=x_patch_arr, x2=self.z_patch,
                              x1_title="Search", x2_title="Template",
                              save_path="vis_attn/t2s/%04d" % self.frame_id, idxs=idxs_s)
                # s-to-t
                vis_attn_maps(attn_weights, q_w=16, k_w=8, skip_len1=64, skip_len2=0, x1=self.z_patch, x2=x_patch_arr,
                              x1_title="Template", x2_title="Search",
                              save_path="vis_attn/s2t/%04d" % self.frame_id, idxs=idxs_t)
                # s-to-it
                vis_attn_maps(attn_weights, q_w=16, k_w=8, skip_len1=64, skip_len2=64, x1=self.iz_patch, x2=x_patch_arr,
                              x1_title="InherentTemplate", x2_title="Search",
                              save_path="vis_attn/s2it/%04d" % self.frame_id, idxs=idxs_t)
                # s-to-s
                vis_attn_maps(attn_weights, q_w=16, k_w=16, skip_len1=64, skip_len2=192, x1=x_patch_arr, x2=x_patch_arr,
                              x1_title="SearchKV", x2_title="SearchQ",
                              save_path="vis_attn/s2s/%04d" % self.frame_id, idxs=idxs_s)
                # size:[1, 64+256, 768] - [batch, size, channel]
                # t feature
                vis_feature_maps(features, 8, self.z_patch, save_path="vis_feat/%04d" % self.frame_id)
                # s feature
                vis_feature_maps(features, 16, x_patch_arr, save_path="vis_feat/%04d" % self.frame_id)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)

        # pred_boxes = out_dict['pred_boxes'].view(-1, 4)

        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # for debug
        if self.debug:
            x1, y1, w, h = self.state
            # image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
            save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image)
        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return ROMTrack
