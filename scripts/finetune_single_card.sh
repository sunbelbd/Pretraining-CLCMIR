#!/usr/bin/env bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4
export FLAGS_check_nan_inf=True

python -u ./finetune_train_paddle.py --task finetune --logger_name finetune_paddle_logs \
  --model_name "$1" --learning_rate=.00001 \
  --cfg_ft "$2" --checkpoint_dir "$3" --mlm --mrm \
  --aux_txt_mlm --aux_t2t_recovery --i2t_recovery --auto_resume
