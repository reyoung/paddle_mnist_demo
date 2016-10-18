set -e
log=train.log

paddle train \
  --config=trainer_config.py \
  --dot_period=10 \
  --log_period=100 \
  --test_all_data_in_one_period=1 \
  --use_gpu=0 \
  --trainer_count=4 \
  --num_passes=100 \
  2>&1 | tee $log

