import os
from pathlib import Path
from dotenv import load_dotenv
import gpu_acc_test
from scripts.download_model import download_models

def run():
    gpu_acc_test.gpu_cpu_time() # check having gpu or cpu
    model_id = os.getenv("MODEL_ID") # loading config
    local_path = os.getenv("LOCAL_PATH")
    if model_id and local_path:
        download_models(model_id=model_id, local_path=Path(local_path))
    else:
        print(f".env file doesn't have data")
if __name__ == "__main__":
    load_dotenv()
   
    run()
