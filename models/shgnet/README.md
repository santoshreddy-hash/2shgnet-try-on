# Place SHGNet-56 weights here

Required / inference:
  SHGNet-56_final.pth   # trained PyTorch FP32 master (~34 MB) — never overwrite
  SHGNet-56.onnx        # live / web FP32 (export via: python -m train.export_onnx)

Size-reduce **.pth** exports (from FP32 separately — see docs/SIZE_REDUCE_PTH.md):
  ../../outputs/size_reduce/pth/SHGNet-56_fused_fp32.pth   # Conv+BN fused FP32
  ../../outputs/size_reduce/pth/SHGNet-56_fp16.pth         # FP16 ← FP32 (~16 MB)
  ../../outputs/size_reduce/pth/SHGNet-56_int8.pth         # INT8 ← FP32 (~7 MB; needs better calib)

Optional:
  hourglass_2stack_best.pth   # legacy 55-LM only with --from-55

Backup of pre-finetune init (if kept):
  ../../local_assets/SHGNet-56_pretrained_init.pth
