cd src
torchrun --nproc_per_node 2 -m \
    --master_addr=127.0.0.2 --master_port=29568 \
    training.main_video \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --dataset-type webdataset \
    --train-data="/home/alex/data/MSRVTT-videos/train_t_umt_preframes_12/train_{0..8}.tar"  \
    --train-num-samples 9000 \
    --val-data="/home/alex/data/MSRVTT-videos/test_t_umt_preframes_12/test_{0..1}.tar"  \
    --val-num-samples 1000 \
    --warmup 0 \
    --batch-size=64 \
    --lr=1e-3 \
    --wd=0.2 \
    --epochs 32 \
    --workers=2 \
    --model MobileCLIP-S1 \
    --pretrained datacompdr \
    --distill-model umt \
    --distill-pretrained l16_25m \
    --num-frames 12 \
    --distill-ckd-alpha 0.0 \
    --distill-temporal-alpha 0.0 \
    --distill-text-fd-alpha 0.0 \

