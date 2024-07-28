cd src
torchrun --nproc_per_node 2 -m \
    --master_addr=127.0.0.2 --master_port=29568 \
    training.main_video \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --dataset-type webdataset \
    --train-data="/home/alex/data/msvd/train/{000000..000011}.tar::/home/alex/data/msvd/val/000000.tar"  \
    --train-num-samples 1300 \
    --val-data="/home/alex/data/msvd/test/{000000..000006}.tar"  \
    --val-num-samples 670 \
    --warmup 0 \
    --batch-size=8 \
    --lr=5e-4 \
    --wd=0.2 \
    --epochs 128 \
    --workers=4 \
    --model MobileCLIP-S1 \
    --pretrained datacompdr \
    --distill-model umt \
    --distill-pretrained l16_25m \
    --num-frames 12 \
    --distill-ckd-alpha 0.0 \
    --distill-temporal-alpha 0.0 \
    --distill-text-fd-alpha 0.0 \
    --precision bf16 \
