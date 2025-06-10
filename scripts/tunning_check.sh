export CUDA_VISIBLE_DEVICES=1

dataset="eurosat"

PCA_LAYER=4
PCA_DIM=8
PCA_VECTORS_NUM=32
PCA_CONSISTENCY_LOSS=0.4
max_epoch=9

for iter in {1..10}
do
    exp_name="tunning/layer${PCA_LAYER}_dim${PCA_DIM}_numvector${PCA_VECTORS_NUM}_loss${PCA_CONSISTENCY_LOSS}_ep${max_epoch}"
    rm -r /data4t/cvpr2025/output/Output_CoPrompt_Modified/${exp_name}
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
        --results-file tunning_loop${dataset}.csv
done