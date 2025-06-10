export CUDA_VISIBLE_DEVICES=1

exp_name=$1
dataset=$2
trainer=CoPrompt

gpu=$3

if [ "$gpu" = "cuda:1" ]; then
  export CUDA_VISIBLE_DEVICES=1
else
  export CUDA_VISIBLE_DEVICES=0
fi

train_bash=scripts/base2new_train_coprompt.sh
test_bash=scripts/base2new_test_coprompt.sh

export PYTHONPATH="$PYTHONPATH:$PWD"

DATASETS=("fgvc_aircraft" "dtd" "ucf101" "eurosat" "caltech101" "oxford_pets" "oxford_flowers" "food101" "sun397")

if [ "$dataset" = "all" ]; then
  for d in "${DATASETS[@]}"
  do
    for seed in 1 2 3
    do

      bash $train_bash $d $seed $exp_name
      if [ "$d" = "food101" ]; then
        bash $test_bash $d $seed $exp_name 5
      else
        bash $test_bash $d $seed $exp_name 8
      fi
    done
  done
else
  for seed in 1 2 3
  do

    bash $train_bash $dataset $seed $exp_name
    if [ "$dataset" = "food101" ]; then
      bash $test_bash $dataset $seed $exp_name 5
    else
      bash $test_bash $dataset $seed $exp_name 8
    fi
  done
fi
