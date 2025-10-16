rm -rf logs ttp_checkpoints

export HCCL_DETERMINISTIC=true
export ASCEND_LAUNCH_BLOCKING=1

export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_GLOBAL_EVENT_ENABLE=1
LOG_PATH="ascend_log"
export ASCEND_PROCESS_LOG_PATH=`pwd`/${LOG_PATH}
rm -rf ${LOG_PATH}
rm -rf output/

export MS_TFT_IP="127.0.0.1"
export MS_TFT_PORT=28030
export TTP_LOG_STDOUT=1

export DEBUG_ERROR_CODE=507053

bash scripts/msrun_launcher.sh "python3 -u run_mindformer.py \
    --config configs/llama2/pretrain_llama2_13b_bf16.yaml \
    --use_parallel True \
    --run_mode train \
    --train_dataset_dir /mnt/disk1/lirongxi/datasets/wiki4096/wiki4096.mindrecord" 4
