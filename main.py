import os
import typer
from pathlib import Path
from dotenv import load_dotenv
import gpu_acc_test
from scripts.download_model import download_models
from scripts.model_info import ModelInfo
from scripts.dataset_parser import process_raw_data
from scripts.config import setting
from scripts.model_train import Train

app = typer.Typer()


@app.command("train-data")
def generate_test_data():
    """To generate train and validate jsonl files"""
    process_raw_data()


@app.command("download_check")
def download_model():
    """
    Download the model from .env and check the gpu and cpu time
    """
    model_id = setting.envpath.model_id
    local_path = setting.envpath.local_path
    if model_id and local_path:
        download_models(model_id=model_id, local_path=Path(local_path))
    else:
        raise Exception("Critical unable to download or find model path")
    gpu_acc_test.gpu_cpu_time()  # check having gpu or cpu


@app.command("start")
def run():
    """
    Gives the model info and start the narrow fine tunning
    """
    local_path = setting.envpath.local_path
    model_info = ModelInfo(local_path)
    if model_info:
        train_ = Train()
        # tokens_ = train_.set_text_tokens()
        # print(tokens_.pad_token)
        model_ = train_.run()
    else:
        raise Exception("Critical unable to load the model")


@app.command("test")
def test_local():
    """
    A simple test code block for .env variables or any code for quick
    """
    print(setting.envpath.model_id)
    print(setting.envpath.output_dir)
    print(setting.train.max_length)


if __name__ == "__main__":
    run()
