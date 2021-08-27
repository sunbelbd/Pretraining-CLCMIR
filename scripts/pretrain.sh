export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
export FLAGS_check_nan_inf=True

# --model_name ../checkpoint/pretrain_paddle/  when you have previous checkpoints, or 
# --model_name bert-base-multilingual-uncased
python -u train_paddle.py --task pretrain \
  --logger_name pretrain_single_paddle_logs \
  --model_name ../checkpoint/pretrain_paddle/ \
  --learning_rate=.00005 \
  --cfg cfgs/pretrain/base_prec_32x24G_fp32.yaml \
  --cfg_ft cfgs/xlretcoco/base_ret_en_16x16G_fp32.yaml \
  --mlm --mrm --cm --aux_txt_mlm \
  --fp16 --auto_resume
