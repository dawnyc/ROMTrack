CONFIG="baseline_stage1"
# CONFIG="baseline_384_stage1"
# CONFIG="tiny_256_stage1"
# CONFIG="small_256_stage1"
# CONFIG="large_384_stage1"

CUDA_VISIBLE_DEVICES=0 python tracking/profile_model.py --config=$CONFIG