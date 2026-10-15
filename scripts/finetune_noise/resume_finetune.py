"""
resume_finetune.py

Continues the noise-data decoder fine-tuning from the latest checkpoint in <run_dir>/checkpoints
(decoder_mountains_finetuned_{N}_epochs.pth): restores the decoder and optimizer state and the
loss history, then trains --epochs more epochs numbered N+1, N+2, ...

Accepts the same arguments as finetune_decoder.py; pass the same --raw_dir (and --run_dir, if one was
given) as the run being resumed, e.g.:
    python scripts/finetune_noise/resume_finetune.py --raw_dir Data/noise_data/noise/awgn_f100
    python scripts/finetune_noise/resume_finetune.py --raw_dir Data/noise_data/noise/awgn_f100 --epochs 5 --lr 5e-6
"""

from finetune_decoder import run

if __name__ == "__main__":
    run(resume=True)
