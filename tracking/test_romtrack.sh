# Different test settings for ROMTrack on LaSOT/LaSOT_ext/TrackingNet/GOT10k/UAV123/OTB100/NFS30
# First, put your trained ROMTrack models on SAVE_DIR/models directory. ('vim lib/test/evaluation/local.py' to set your SAVE_DIR)
# Then, uncomment the code of corresponding test settings.
# Finally, you can find the tracking results on RESULTS_PATH and the tracking plots on RESULTS_PLOT_PATH. (see 'lib/test/evaluation/local.py')


##########-------------- ROMTrack -----------------##########
### LaSOT test and evaluation
python tracking/test.py ROMTrack baseline_stage2 --dataset lasot --threads 32 --num_gpus 8 --params__model ROMTrack_epoch0100.pth.tar
python tracking/analysis_results.py --dataset_name lasot --tracker_param baseline_stage2

### LaSOT_ext test and evaluation
python tracking/test.py ROMTrack baseline_stage2 --dataset lasot_ext --threads 32 --num_gpus 8 --params__model ROMTrack_epoch0100.pth.tar
python tracking/analysis_results.py --dataset_name lasot_ext --tracker_param baseline_stage2

### TrackingNet test and pack
# python tracking/test.py ROMTrack baseline_stage2 --dataset trackingnet --threads 32 --num_gpus 8 --params__model ROMTrack_epoch0100.pth.tar
# python lib/test/utils/transform_trackingnet.py --tracker_name ROMTrack --cfg_name baseline_stage2

### ROMTrack(Only for GOT-10k) GOT10k test and pack
# python tracking/test.py ROMTrack got_stage2 --dataset got10k_test --threads 32 --num_gpus 8 --params__model ROMTrack_epoch0050.pth.tar
# python lib/test/utils/transform_got10k.py --tracker_name ROMTrack --cfg_name got_stage2

### UAV123
# python tracking/test.py ROMTrack baseline_stage2 --dataset uav --threads 32 --num_gpus 8 --params__model ROMTrack_epoch0100.pth.tar --params__search_area_scale 3.75
# python tracking/analysis_results.py --dataset_name uav --tracker_param baseline_stage2

### OTB100
# python tracking/test.py ROMTrack baseline_stage2 --dataset otb --threads 32 --num_gpus 8 --params__model ROMTrack_epoch0100.pth.tar --params__search_area_scale 3.65
# python tracking/analysis_results.py --dataset_name otb --tracker_param baseline_stage2

### NFS30
# python tracking/test.py ROMTrack baseline_stage2 --dataset nfs --threads 32 --num_gpus 8 --params__model ROMTrack_epoch0100.pth.tar --params__search_area_scale 3.95
# python tracking/analysis_results.py --dataset_name nfs --tracker_param baseline_stage2


##########-------------- ROMTrack-384 -----------------##########
### LaSOT test and evaluation
python tracking/test.py ROMTrack baseline_384_stage2 --dataset lasot --threads 32 --num_gpus 8 --params__model ROMTrack-384_epoch0100.pth.tar
python tracking/analysis_results.py --dataset_name lasot --tracker_param baseline_384_stage2

### LaSOT_ext test and evaluation
python tracking/test.py ROMTrack baseline_384_stage2 --dataset lasot_ext --threads 32 --num_gpus 8 --params__model ROMTrack-384_epoch0100.pth.tar
python tracking/analysis_results.py --dataset_name lasot_ext --tracker_param baseline_384_stage2

### TrackingNet test and pack
# python tracking/test.py ROMTrack baseline_384_stage2 --dataset trackingnet --threads 32 --num_gpus 8 --params__model ROMTrack-384_epoch0100.pth.tar --params__search_area_scale 4.5
# python lib/test/utils/transform_trackingnet.py --tracker_name ROMTrack --cfg_name baseline_384_stage2

### ROMTrack-384(Only for GOT-10k) GOT10k test and pack
# python tracking/test.py ROMTrack got_384_stage2 --dataset got10k_test --threads 32 --num_gpus 8 --params__model ROMTrack-384_epoch0050.pth.tar --params__search_area_scale 4.75
# python lib/test/utils/transform_got10k.py --tracker_name ROMTrack --cfg_name got_384_stage2
