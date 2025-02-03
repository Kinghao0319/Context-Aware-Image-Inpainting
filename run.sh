#!/bin/bash

# vim  ~/.cache/huggingface/accelerate/default_config.yaml

exp_dir="exp1"
accelerate launch /data/$USER/code/Context-Aware-Image-Inpainting/src/run_context_aware_inpaint.py \
    --pretrained_model_name_or_path="/data/$USER/code/Context-Aware-Image-Inpainting/pretrained/runwayml_sd_inpainting"  \
    --instance_data_dir="/data/$USER/code/Context-Aware-Image-Inpainting/${exp_dir}/merged_imgs" \
    --output_dir="/data/$USER/code/Context-Aware-Image-Inpainting/${exp_dir}/output_model" \
    --instance_prompt="a photo of asdfg bear" \
    --inference_prompt="asdfg bear on the road" \
    --subject_image_num=5 \
    --exp_dir=${exp_dir} \
    --resolution=512 \
    --mixed_precision="fp16" \
    --train_batch_size=1 \
    --learning_rate=5e-6 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=400 \
    --gradient_accumulation_steps=2 \
    --gradient_checkpointing \
    --train_text_encoder \
    --seed="0"
#   --push_to_hub