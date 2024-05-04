import os
import sys

tracker_path = os.path.join(os.path.dirname(__file__), '../../..')
if tracker_path not in sys.path:
    sys.path.insert(0, os.path.abspath(tracker_path))

from lib.test.vot20.ROMTrack_vot20 import run_vot_exp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tracker_params = {}
tracker_params['model'] = "ROMTrack-Small-256_epoch0100.pth.tar"
tracker_params['vis_attn'] = 0
tracker_params['search_area_scale'] = 4.0
run_vot_exp('ROMTrack', 'small_256_stage2', vis=False, tracker_params=tracker_params)
