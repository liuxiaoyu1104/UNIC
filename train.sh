
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=$((RANDOM + 10000)) \
    --use_env \
    main.py \
    --coco_path /home/liuxiaoyu/image_cropping/GAIC2\
    --output_dir  ./output/GAIC_UNIC\
    --label_class soft \
    --epochs 50 \
    --lr_drop 30 \
    --outside_ratio \
    --outpainting \
     



