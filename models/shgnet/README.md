# Place SHGNet-56 weights here

Required / inference:
  SHGNet-56_final.pth   # trained PyTorch (~34 MB) — synced from outputs/checkpoints after train
  SHGNet-56.onnx        # live / web (export via: python -m train.export_onnx)

Optional:
  hourglass_2stack_best.pth   # legacy 55-LM only with --from-55

Backup of pre-finetune init (if kept):
  ../../local_assets/SHGNet-56_pretrained_init.pth
