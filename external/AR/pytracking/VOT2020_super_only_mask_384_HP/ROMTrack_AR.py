import os
import sys

ar_path = os.path.join(os.path.dirname(__file__), '../..')
if ar_path not in sys.path:
    sys.path.insert(0, os.path.abspath(ar_path))
tracker_path = os.path.join(ar_path, '../..')
if tracker_path not in sys.path:
    sys.path.insert(0, os.path.abspath(tracker_path))

from external.AR.pytracking.VOT2020_super_only_mask_384_HP.ROMTrack_alpha_seg_class import run_vot_exp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tracker_params = {}
tracker_params['model'] = "ROMTrack_epoch0100.pth.tar"
tracker_params['vis_attn'] = 0
tracker_params['search_area_scale'] = 4.0
run_vot_exp('ROMTrack', 'baseline_stage2',
            'ARcm_coco_seg_only_mask_384', 0.6, VIS=False, tracker_params=tracker_params)
