import time
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError
for attempt in range(40):
    try:
        p=snapshot_download("kalbin/swb-fisher-casper-multiturn-sft-v4", repo_type="dataset",
            local_dir="/iopsstor/scratch/cscs/mrohania/datasets/swb_multiturn",
            allow_patterns=["data/*.parquet"], max_workers=3)
        print("SWB DONE",p); break
    except HfHubHTTPError as e:
        print("retry",attempt,str(e)[:80]); time.sleep(min(60*(attempt+1),600))
