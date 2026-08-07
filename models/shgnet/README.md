# Place SHGNet-56 weights here

Required / inference:
  SHGNet-56_final.pth   # trained PyTorch FP32 master (~34 MB) — never overwrite
  SHGNet-56.onnx        # live / web FP32 (export via: python -m train.export_onnx)

Size-reduce exports (gitignored binaries; see docs/SIZE_REDUCE_PIPELINE.md):
  ../../outputs/size_reduce/SHGNet-56_fused_fp32.onnx   # Conv+BN fused FP32 (~25 MB)
  ../../outputs/size_reduce/SHGNet-56_fp16.onnx         # half precision (~13 MB) — recommended browser
  ../../outputs/size_reduce/SHGNet-56_int8.onnx         # dynamic INT8 (~7 MB) — needs calib before prod

Optional:
  hourglass_2stack_best.pth   # legacy 55-LM only with --from-55

Backup of pre-finetune init (if kept):
  ../../local_assets/SHGNet-56_pretrained_init.pth
