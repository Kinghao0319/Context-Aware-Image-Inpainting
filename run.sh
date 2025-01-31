accelerate launch ~/code/Context-Aware-Image-Inpainting/src/run_context_aware_inpaint.py \
    --pretrained_model_name_or_path="~/code/Context-Aware-Image-Inpainting/pretrained/runwayml_sd_inpainting"  \
    --instance_data_dir="~/code/Context-Aware-Image-Inpainting/exp/merged_imgs" \
    --output_dir="~/code/Context-Aware-Image-Inpainting/exp/output_model" \
    --instance_prompt="a photo of asdfg bear" \
    --inference_prompt="asdfg bear on the road" \
    --subject_image_num=5 \
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