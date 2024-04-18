# There are the detailed training settings for ROMTrack.
# 1. download pretrained MAE ViT-Base model from https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth.
# 2. set the proper pretrained models path, namely, './pretrained/mae/mae_pretrain_vit_base.pth'.
# 3. follow the instructions to train corresponding trackers.

### Training ROMTrack
# Stage1:
python tracking/train.py --script ROMTrack --config baseline_stage1 --save_dir /YOUR/PATH/TO/SAVE --mode multiple --nproc_per_node 4
# Stage2: first create a folder in your working directory('vim lib/train/admin/local.py' to set your workspace_dir) named as "models-finetune",
# and put the final ckpt of Stage1 into it
python tracking/train.py --script ROMTrack --config baseline_stage2 --save_dir /YOUR/PATH/TO/SAVE --mode multiple --nproc_per_node 4

### Training ROMTrack(Only for GOT-10k)
# Stage1:
python tracking/train.py --script ROMTrack --config got_stage1 --save_dir /YOUR/PATH/TO/SAVE --mode multiple --nproc_per_node 4
# Stage2: first create a folder in your working directory('vim lib/train/admin/local.py' to set your workspace_dir) named as "models-finetune",
# and put the final ckpt of Stage1 into it
python tracking/train.py --script ROMTrack --config got_stage2 --save_dir /YOUR/PATH/TO/SAVE --mode multiple --nproc_per_node 4

### For other models(ROMTrack-384, ROMTrack-384(Only for GOT-10k), ROMTrack-Large-384, ROMTrack-Tiny-256, ROMTrack-Small-256), just modify the config
