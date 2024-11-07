#!/bin/bash

echo "Starting ELECTRA-based Text DeepFM Training Pipeline..."

# 1. 벡터 생성
echo "Step 1: Create new vectors..."
python main.py \
    -c config/config_electra_optimized.yaml \
    -m Text_DeepFM \
    -w False \
    -r ELECTRA_Text_DeepFM_Optimized_vector_create_True \
    --device cuda

# 2. 추가 실험 - 저장된 벡터 사용
# echo "Step 2: Additional training using saved vectors..."
# python main.py \
#     -c config/config_electra_optimized.yaml \
#     -m Text_DeepFM \
#     -w True \
#     -r ELECTRA_Text_DeepFM_Optimized_vector_create_False \
#     --device cuda
# 
# # 3. 앙상블을 위한 추가 실험 (다른 시드값 사용)
# echo "Step 3: Running multiple seeds for ensemble..."
# for seed in 42 123 256 789 1024
# do
#     echo "Training with seed ${seed}..."
#     python main.py \
#         -c config/config_electra_optimized.yaml \
#         -m Text_DeepFM \
#         -w True \
#         -r ELECTRA_Text_DeepFM_Seed_${seed} \
#         --seed ${seed} \
#         --device cuda
# done
# 
# # 4. 앙상블 수행
# echo "Step 4: Performing ensemble of multiple seed runs..."
# python ensemble.py \
#     --ensemble_files ELECTRA_Text_DeepFM_Seed_42,ELECTRA_Text_DeepFM_Seed_123,ELECTRA_Text_DeepFM_Seed_256,ELECTRA_Text_DeepFM_Seed_789,ELECTRA_Text_DeepFM_Seed_1024 \
#     --ensemble_strategy weighted \
#     --ensemble_weight 0.25,0.2,0.2,0.2,0.15 \
#     --result_path saved/electra_submit/

echo "Pipeline completed!"