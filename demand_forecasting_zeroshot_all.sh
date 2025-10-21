gpu_id=1
batch_size=16
output_dir=./output/
results_path=./results/
chronos_model=amazon/chronos-t5-small

# Zero-shot evaluation
for run_name in sg sg_daily aus aus_daily; do
python scripts/evaluation/evaluate.py \
   scripts/evaluation/configs/demand_${run_name}.yaml \
   scripts/evaluation/results/chronos-zero-shot-${run_name}.csv \
   --chronos-model-id $chronos_model \
   --batch-size=$batch_size \
   --device=cuda:${gpu_id} \
   --num-samples 100 \
   --run-type zero_shot \
   --output-dir ${output_dir}/${run_name} \
   --results-path ${results_path}/${run_name}
done
