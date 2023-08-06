# ROMTrack
[ICCV 2023] Robust Object Modeling for Visual Tracking

## News
**[August 6, 2023]**
- We release Code of ROMTrack.

**[July 14, 2023]**
- ROMTrack is accepted to **ICCV2023**.

## TODO
- [x] Code for ROMTrack
- [ ] Model Zoo and Raw Results
- [ ] Refine README

## Install the environment
Use the Anaconda
```
conda create -n romtrack python=3.6
conda activate romtrack
bash install_pytorch17.sh
```

## Data Preparation
Put the tracking datasets in ./data. It should look like:
   ```
   ${ROMTrack_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- lasot_ext
            |-- atv
            |-- badminton
            |-- cosplay
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- train2017
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
   ```
## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Train ROMTrack
Training with multiple GPUs using DDP. More details of other training settings can be found at ```tracking/train_romtrack.sh```
```
bash tracking/train_romtrack.sh
```

## Test and evaluate ROMTrack on benchmarks

- LaSOT/LaSOT_ext/GOT10k-test/TrackingNet/OTB100/UAV123/NFS30. More details of test settings can be found at ```tracking/test_romtrack.sh```
```
bash tracking/test_romtrack.sh
```

## Compute FLOPs/Params and test speed
```
python tracking/profile_model.py --config="baseline_stage1"
```

## Acknowledgments
* Thanks for [STARK](https://github.com/researchmm/Stark), [PyTracking](https://github.com/visionml/pytracking) and [MixFormer](https://github.com/MCG-NJU/MixFormer) Library, which helps us to quickly implement our ideas and test our performances.
* Our implementation of the ViT is modified from the [Timm](https://github.com/rwightman/pytorch-image-models) repo.