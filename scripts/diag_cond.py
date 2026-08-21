import os, sys, torch; sys.path.insert(0,"/iopsstor/scratch/cscs/mrohania/moshi-finetune")
from moshi.models import loaders
from moshi.run_inference import get_condition_tensors
AD="/iopsstor/scratch/cscs/mrohania/moshi_runs/S_cond_v3/checkpoints/checkpoint_000800/consolidated/lora.safetensors"
cfg=os.path.join(os.path.dirname(AD),"config.json")
ci=loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16",lora_weights=AD,config_path=cfg)
lm=ci.get_moshi(device="cuda",dtype=torch.bfloat16,fuse_lora=True)
print("conditioners:", list(lm.condition_provider.conditioners.keys()))
prev=None
for E in ["neu","hap","ang","sad"]:
    os.environ["COND_EMOTION"]=E
    ct=get_condition_tensors("moshi", lm, 1, 1.0)
    s=lm.fuser.get_sum(ct).float()
    print(f"{E}: cond_sum shape={tuple(s.shape)} norm={s.norm().item():.4f} mean={s.mean().item():.5f} head={s.flatten()[:3].tolist()}")
    if prev is not None:
        print(f"    L2 diff vs neu: {(s-prev).norm().item():.4f}")
    if E=="neu": prev=s
