#!/usr/local_rwth/bin/zsh


# export TF_CPP_MIN_LOG_LEVEL=1   # disable info messages
# export TF_GPU_THREAD_MODE='gpu_private'
# #export NCCL_SOCKET_NTHREADS=8   # multi-threading for NCCL communication
# # ==============================
# # ===== Set TF_CONFIG
# # ==============================
# # This step is tensorflow specific
# pr=tag_program1
# export TF_CONFIG=$(python -W ignore $pr)
# #if [[ "${RANK}" -eq "0" ]]; then
# echo "TF_CONFIG: ${TF_CONFIG}"


# run training
python -W ignore tag_program  \
--global_batch_size=tag_batch \
--augment=tag_aug \
--epoch=tag_epoch 2>&1
