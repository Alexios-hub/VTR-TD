cd src
torchrun --nproc_per_node 2 -m \
    --master_addr=127.0.0.2 --master_port=29568 \
    training.main_video \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --dataset-resampled \
    --dataset-type vtr \
    --train-data="/home/alex/data/didemo/train_dir"  \
    --train-data-ann="/home/alex/data/didemo/umt_ann/didemo_ret_trainval.json" \
    --train-num-samples 9500 \
    --val-data="/home/alex/data/didemo/test_dir"  \
    --val-data-ann="/home/alex/data/didemo/umt_ann/didemo_ret_test.json" \
    --val-num-samples 1000 \
    --warmup 0 \
    --batch-size=6 \
    --lr=5e-4 \
    --wd=0.2 \
    --epochs 32 \
    --workers=4 \
    --model MobileCLIP-S1 \
    --pretrained datacompdr \
    --distill-model umt \
    --distill-pretrained l16_25m \
    --num-frames 32 \
    --distill-ckd-alpha 0.0 \
    --distill-temporal-alpha 0.0 \
    --distill-text-fd-alpha 0.0 \
    --precision bf16 \