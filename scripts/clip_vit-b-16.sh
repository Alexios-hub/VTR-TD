cd src
torchrun --nproc_per_node 2 -m \
    --master_addr=127.0.0.2 --master_port=29568 \
    training.main_video \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --dataset-type webdataset \
    --train-data="/home/user/data/MSRVTT-videos/train_t_umt_preframes/train_{0..8}.tar"  \
    --train-num-samples 9000 \
    --val-data="/home/user/data/MSRVTT-videos/test_t_umt_preframes/test_{0..1}.tar"  \
    --val-num-samples 1000 \
    --warmup 0 \
    --batch-size=128 \
    --lr=1e-4 \
    --wd=0.1 \
    --epochs 32 \
    --workers=2 \
    --model ViT-B-16 \
    --pretrained datacomp_xl_s13b_b90k \
    --distill-model umt \
    --distill-pretrained l16_25m \
    --num-frames 4 \
    --distill-st-alpha 1.0 \
    --distill-ckd-alpha 1.0 \
    --distill-temporal-alpha 1.0 \
    --distill-text-fd-alpha 10.0 \


