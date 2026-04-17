cuda=0
exp_name=hh_genarm_humor_r8_1epoch
objective=humor

lora_r=8
lora_alpha=16

beta=1e-3

epoch=1
learning_rate=5e-4
bs=32
per_device_train_batch_size=4

model_name_or_path=TinyLlama/TinyLlama-1.1B-Chat-v1.0

num_GPU=$(echo $cuda | awk -F, '{print NF}')
gradient_accumulation_steps=$(($bs/$num_GPU/$per_device_train_batch_size))
preference_dataset=HH_RLHF

output_dir=./HH-RLHF/exp_genarm_humor
if [ -d "${output_dir}" ]; then
    echo -e "\n\n"
    echo "Error: Directory "${output_dir}" already exists. Please delete it or choose a new output_dir." >&2
    exit 1
fi
echo "Output dir: $output_dir"

accelerate launch --gpu_ids $cuda --main_process_port 29502 --num_processes $num_GPU train_genarm.py \
    --preference_dataset=$preference_dataset \
    --objective=$objective \
    --lora_r=$lora_r \
    --lora_alpha=$lora_alpha \
    --model_name_or_path=$model_name_or_path \
    --beta=$beta \
    --learning_rate=$learning_rate \
    --num_train_epochs=$epoch \
    --output_dir=$output_dir \
    --run_name=$exp_name \
    --per_device_train_batch_size=$per_device_train_batch_size \
    --gradient_accumulation_steps=$gradient_accumulation_steps \
    --per_device_eval_batch_size=2 \
    --logging_steps=10 \
    --evaluation_strategy="steps" \
    --eval_steps=20 \
    --save_strategy="steps" \
    --save_steps=1000 \
    --lr_scheduler_type="cosine" \
    --warmup_steps=20 \
    --weight_decay=0.05 \
    --gradient_checkpointing=True \
    --bf16=True \
    --max_prompt_length=512 \
    --max_length=1024 \
    --report_to="none" \
    --remove_unused_columns=False

echo "Finished training $output_dir"
