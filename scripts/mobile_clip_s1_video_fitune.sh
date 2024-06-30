cd src
torchrun --nproc_per_node 2 -m \
    --master_addr=127.0.0.2 --master_port=29568 \
    training.main_video \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --dataset-type webdataset \
    --train-data="/home/user/data/MSRVTT-videos/train_mae_prepro/train_{0..8}.tar"  \
    --train-num-samples 9000 \
    --val-data="/home/user/data/MSRVTT-videos/test_mae_prepro/test_{0..1}.tar"  \
    --val-num-samples 1000 \
    --warmup 1000 \
    --batch-size=32 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs 32 \
    --workers=2 \
    --model MobileCLIP-S1 \
    --pretrained datacompdr \
    --distill-model MCG-NJU/videomae-base \
    --distill-pretrained MCG-NJU/videomae-base \
    --num-frames 16 \
    --distill-alpha 0.001 \


