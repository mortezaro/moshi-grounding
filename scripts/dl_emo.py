import time
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError
for attempt in range(15):
    try:
        p=snapshot_download("NathanRoll/speech-emotion-dataset-english", repo_type="dataset",
            local_dir="/iopsstor/scratch/cscs/mrohania/datasets/ser_english", max_workers=2); print("EMO DONE",p); break
    except HfHubHTTPError as e:
        print("retry",attempt,str(e)[:80]); time.sleep(min(60*(attempt+1),400))
