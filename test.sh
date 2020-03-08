export CUDA_VISIBLE_DEVICES=0,1,2,3

python run_bert.py \
--model_name_or_path ../tianchi/chinese_wwm_ext_pytorch \
--do_test \
--data_dir ./data/ \
--output_dir ./output_base_wwm \
--max_seq_length 256 \
--eval_steps 200 \
--per_gpu_train_batch_size 16 \
--gradient_accumulation_steps 1 \
--warmup_steps 0 \
--per_gpu_eval_batch_size 64 \
--learning_rate 1e-5 \
--adam_epsilon 1e-6 \
--weight_decay 0 \
--train_steps 12500
