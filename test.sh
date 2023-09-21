
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=$((RANDOM + 10000)) \
    --use_env \
    main.py \
    --output_dir  ./output/GAIC_outpainting_ema_1\
    --label_class soft \
    --coco_path /home/liuxiaoyu/image_cropping/GAIC2\
    --epochs 200 \
    --lr_drop 70 \
    --outside_ratio \
    --print_pic\
    --outpainting \
    --resume /home/liuxiaoyu/image_cropping/UNIC/output/checkpoint0031.pth\
    --eval \
     



