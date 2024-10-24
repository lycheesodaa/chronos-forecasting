gpu_id=1
batch_size=16
output_dir=./output/aus
yaml_filename=demand_aus

# Zero-shot evaluation
python scripts/evaluation/evaluate.py \
    scripts/evaluation/configs/${yaml_filename}.yaml \
    scripts/evaluation/results/chronos-zero-shot.csv \
    --chronos-model-id "amazon/chronos-t5-base" \
    --batch-size=$batch_size \
    --device=cuda:${gpu_id} \
    --num-samples 100 \
    --run-type zero_shot \
    --output-dir $output_dir

# Fine-tune `amazon/chronos-t5-base` for 1000 steps with initial learning rate of 1e-3
# Ensure that prediction_length is greater than the largest prediction length specified in the evaluation .yaml file
CUDA_VISIBLE_DEVICES=${gpu_id} python scripts/training/train.py --config scripts/training/configs/chronos-t5-base.yaml \
    --model-id amazon/chronos-t5-base \
    --no-random-init \
    --max-steps 1000 \
    --learning-rate 0.001 \
    --prediction-length 72 \
    --output-dir $output_dir

# Fine-tuned evaluation
python scripts/evaluation/evaluate.py \
    scripts/evaluation/configs/${yaml_filename}.yaml \
    scripts/evaluation/results/chronos-finetuned.csv \
    --chronos-model-id "amazon/chronos-t5-base" \
    --batch-size=$batch_size \
    --device=cuda:${gpu_id} \
    --num-samples 100 \
    --run-type finetuned \
    --output-dir $output_dir
