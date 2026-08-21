import time
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError
for attempt in range(40):
    try:
        p=snapshot_download("kalbin/fisher-sft-v2", repo_type="dataset",
            local_dir="/iopsstor/scratch/cscs/mrohania/datasets/fisher_sft_v2",
            allow_patterns=["data/*.parquet"], max_workers=3)
        print("FISHER FULL DONE",p); break
    except HfHubHTTPError as e:
        print("retry",attempt,str(e)[:80]); time.sleep(min(60*(attempt+1),600))
