python3 imitate_episodes.py \
    --task_name insert_redness_relief_slanted \
    --ckpt_dir training_outputs \
    --policy_class ACT --kl_weight 10 --chunk_size 40 --hidden_dim 512 --batch_size 32 --dim_feedforward 3200 \
    --num_epochs 5000  --lr 2e-5 \
    --seed 0



python3 imitate_episodes.py \
    --task_name insert_redness_relief_slanted_anywhere \
    --ckpt_dir training_outputs \
    --policy_class ACT --kl_weight 10 --chunk_size 40 --hidden_dim 512 --batch_size 32 --dim_feedforward 3200 \
    --num_epochs 5000  --lr 2e-5 \
    --seed 0 \
    --run_name anywhere_slanted


python3 imitate_episodes.py \
    --task_name pick_vial \
    --ckpt_dir training_outputs \
    --policy_class ACT --kl_weight 10 --chunk_size 40 --hidden_dim 512 --batch_size 32 --dim_feedforward 3200 \
    --num_epochs 20000  --lr 2e-5 \
    --seed 0 \
    --run_name pick_vial_05_10


python3 imitate_episodes.py \
    --task_name place_vial \
    --ckpt_dir training_outputs \
    --policy_class ACT --kl_weight 10 --chunk_size 40 --hidden_dim 512 --batch_size 32 --dim_feedforward 3200 \
    --num_epochs 20000  --lr 2e-5 \
    --seed 0 \
    --run_name place_vial_5_10