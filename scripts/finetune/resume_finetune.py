"""
resume_finetune.py

Continues the Mountains decoder fine-tuning from the latest checkpoint in checkpoints/
(decoder_mountains_finetuned_{N}_epochs.pth): restores the decoder and optimizer state and the
loss history, then trains --epochs more epochs numbered N+1, N+2, ...

Accepts the same arguments as finetune_decoder.py, e.g.:
    python scripts/finetune/resume_finetune.py
    python scripts/finetune/resume_finetune.py --epochs 5 --lr 5e-6
"""

from finetune_decoder import run

if __name__ == "__main__":
    run(resume=True)
