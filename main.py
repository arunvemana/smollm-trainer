import os
import typer
from pathlib import Path
from dotenv import load_dotenv
import gpu_acc_test
from scripts.download_model import download_models
from scripts.model_info import ModelInfo
from scripts.dataset_parser import process_raw_data
from scripts.config import setting

app = typer.Typer()


@app.command("train-data")
def generate_test_data():
    """To generate train and validate jsonl files"""
    process_raw_data()


@app.command("start")
def run():
    gpu_acc_test.gpu_cpu_time()  # check having gpu or cpu
    model_id = setting.envpath.model_id
    local_path = setting.envpath.local_path
    if model_id and local_path:
        model, tokens = download_models(model_id=model_id, local_path=Path(local_path))
    if model:
        Model_info = ModelInfo(local_path)
    else:
        print(f".env file doesn't have data")


@app.command("test")
def test_local():
    print(setting.envpath.model_id)
    print(setting.envpath.output_dir)
    print(setting.train.max_length)


if __name__ == "__main__":
    run()
