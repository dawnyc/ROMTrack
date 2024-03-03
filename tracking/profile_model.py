import argparse
import torch
import os
import time
import importlib
import _init_paths
from torch import nn
from lib.models.ROMTrack.Transformer import Attention
from thop import profile
from thop.utils import clever_format


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='ROMTrack', choices=['ROMTrack'],
                        help='training script name')
    parser.add_argument('--config', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--display_name', type=str, default='ROMTrack')
    args = parser.parse_args()

    return args


def evaluate_ROMTrack(model, template, search, display_info='ROMTrack'):
    """Compute MACs, Params and FPS"""
    macs, params = profile(model, inputs=(template, template, search))
    macs, params = clever_format([macs, params], "%.3f")
    print('==>Macs is ', macs)
    print('==>Params is ', params)

    T_w = 10
    T_t = 100
    print("testing speed ...")
    with torch.no_grad():
        # overall
        for i in range(T_w):
            _ = model(template, template, search)
        start = time.time()
        _ = model.set_online(template)
        for i in range(T_t):
            _ = model.forward_test(template, search)
        end = time.time()
        avg_lat = (end - start) / T_t
        print("\033[0;32;40m The average overall FPS of {} is {}.\033[0m".format(display_info, 1.0 / avg_lat))


def get_data(bs, sz):
    img_patch = torch.randn(bs, 3, sz, sz)
    return img_patch


if __name__ == "__main__":
    # device = "cuda:0"
    # torch.cuda.set_device(device)
    device = "cpu"
    args = parse_args()
    '''update cfg'''
    prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    yaml_fname = os.path.join(prj_dir, 'experiments/%s/%s.yaml' % (args.script, args.config))
    print("yaml_fname: {}".format(yaml_fname))
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    print("cfg: {}".format(cfg))
    '''set some values'''
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE
    '''import ROMTrack network module'''
    model_module = importlib.import_module('lib.models.ROMTrack')
    if args.script == "ROMTrack":
        model_constructor = model_module.build_vit
        model = model_constructor(cfg)
        # get the template and search
        template = get_data(bs, z_sz)
        search = get_data(bs, x_sz)
        # transfer to device
        model = model.to(device)
        template = template.to(device)
        search = search.to(device)
        # evaluate the model properties
        evaluate_ROMTrack(model, template, search, args.display_name)
