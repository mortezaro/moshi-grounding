import time
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import HfHubHTTPError
OUT="/iopsstor/scratch/cscs/mrohania/datasets/fisher_sft_v2"
# validation subset: first 5 fisher shards
for i in range(5):
    fn=f"data/train-{i:05d}-of-00296.parquet"
    for attempt in range(15):
        try:
            hf_hub_download("kalbin/fisher-sft-v2", fn, repo_type="dataset", local_dir=OUT); break
        except HfHubHTTPError as e:
            print("retry",fn,attempt,str(e)[:80]); time.sleep(min(60*(attempt+1),400))
    print("got", fn, flush=True)
print("FISHER SUBSET DONE")
