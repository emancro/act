python3 imitate_episodes.py \
    --task_name insert_redness_relief_slanted \
    --ckpt_dir training_outputs \
    --policy_class ACT --kl_weight 10 --chunk_size 40 --hidden_dim 512 --batch_size 32 --dim_feedforward 3200 \
    --num_epochs 5000  --lr 2e-5 \
    --seed 0