export CUDA_VISIBLE_DEVICES=1
dataset="eurosat"
SKIP_UNTIL_CRASH_POINT=true

for PCA_LAYER in 4
do
for PCA_DIM in 8 16 32 64 128
do
for PCA_VECTORS_NUM in 8 16 32 64 128
do
for PCA_CONSISTENCY_LOSS in 0.2 0.4 0.8 1.0 1.2 1.5 2.0
do
for max_epoch in 5 6 7 8 9 10
do
    if [ "$PCA_LAYER" -eq 4 ] && \
       [ "$PCA_DIM" -eq 16 ] && \
       [ "$PCA_VECTORS_NUM" -eq 8 ] && \
       [ "$(echo "$PCA_CONSISTENCY_LOSS == 1.0" | bc -l)" -eq 1 ] && \
       [ "$max_epoch" -eq 5 ]; then
        SKIP_UNTIL_CRASH_POINT=false
    fi

    if [ "$SKIP_UNTIL_CRASH_POINT" = true ]; then
        echo "Skipping: PCA_LAYER=$PCA_LAYER, PCA_DIM=$PCA_DIM, PCA_VECTORS_NUM=$PCA_VECTORS_NUM, PCA_CONSISTENCY_LOSS=$PCA_CONSISTENCY_LOSS, max_epoch=$max_epoch"
        continue
    fi

    exp_name="tunning/layer${PCA_LAYER}_dim${PCA_DIM}_numvector${PCA_VECTORS_NUM}_loss${PCA_CONSISTENCY_LOSS}_ep${max_epoch}"
    for seed in 1 2 3
    do
        bash scripts/base2new_train_coprompt.sh $dataset $seed $exp_name \
        OPTIM.MAX_EPOCH $max_epoch \
        TRAINER.CoPrompt.PCA_LAYER $PCA_LAYER \
        TRAINER.CoPrompt.PCA_DIM $PCA_DIM \
        TRAINER.CoPrompt.PCA_VECTORS_NUM $PCA_VECTORS_NUM \
        TRAINER.CoPrompt.PCA_CONSISTENCY_LOSS $PCA_CONSISTENCY_LOSS

        bash scripts/base2new_test_coprompt.sh $dataset $seed $exp_name $max_epoch \
        OPTIM.MAX_EPOCH $max_epoch \
        TRAINER.CoPrompt.PCA_LAYER $PCA_LAYER \
        TRAINER.CoPrompt.PCA_DIM $PCA_DIM \
        TRAINER.CoPrompt.PCA_VECTORS_NUM $PCA_VECTORS_NUM \
        TRAINER.CoPrompt.PCA_CONSISTENCY_LOSS $PCA_CONSISTENCY_LOSS

    done

    python tools/tunning_result_reader.py \
        --train-base /data4t/cvpr2025/output/Output_CoPrompt_Modified/${exp_name}/train_base/$dataset \
        --results-file tunning_${dataset}.csv
    rm -r /data4t/cvpr2025/output/Output_CoPrompt_Modified/${exp_name}
done
done
done
done
done