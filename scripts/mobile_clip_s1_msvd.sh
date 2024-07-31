cd src
torchrun --nproc_per_node 2 -m \
    --master_addr=127.0.0.2 --master_port=29568 \
    training.main_video \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --dataset-type vtr \
    --train-data="/home/alex/data/msvd/train_val_dir"  \
    --train-data-ann="/home/alex/data/msvd/ann/train_val.json" \
    --val-data="/home/alex/data/msvd/test_dir"  \
    --val-data-ann="/home/alex/data/msvd/umt_ann/msvd_ret_test.json" \
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
