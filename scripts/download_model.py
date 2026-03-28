import torch
from transformers import AutoModelForCausalLM,AutoTokenizer, PreTrainedModel,PreTrainedTokenizerBase
from pathlib import Path
from typing import Tuple, Optional


def download_models(model_id:str, local_path:Path) -> Tuple[PreTrainedModel | None ,PreTrainedTokenizerBase | None]:
    local_path.mkdir(parents=True, exist_ok=True) # checking local_dir exist or not
    try:
        print(f"Downloading the {model_id}")
        print("First Downloading the tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(local_path)
        print("downloading the model weight")
        model = AutoModelForCausalLM.from_pretrained(model_id,device_map="auto",low_cpu_mem_usage=True)
        model.save_pretrained(local_path)
        for f in local_path.iterdir():
            if f.is_file():
                print(f"File:{f.name} | Size:{round(f.stat().st_size/(1024*1024),4)}MB")
        return model, tokenizer
    except Exception as e:
        print(f"Failed to download model:{e}")
        return None, None

if  __name__ == "__main__":
    model_id = "HuggingFaceTB/SmolLM-135M"
    local_path = Path("./models/smollm-135m-base")
    download_models(model_id=model_id,local_path=local_path)
