import json, os
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.columns import Columns
from transformers import AutoModelForCausalLM


class ModelInfo:
    def __init__(self, model) -> None:
        self.model_path = os.getenv("LOCAL_PATH")
        self.model = model
        self.load_config()

    def load_config(self) -> None:
        """Load the config.json of the model"""
        try:
            if self.model_path:
                with open(Path(self.model_path) / "config.json") as f:
                    config = json.load(f)
            else:
                raise Exception
        except Exception as e:
            print(f"Error at loading the model config: {e}")
            return None

        table = Table(title="=== Model Architecture info ===")
        # table.add_column("config info", style="cyan")
        table.add_row(f" Hidden Size: {config['hidden_size']}")
        table.add_row(f" Number of layers: {config['num_hidden_layers']}")
        table.add_row(f" Attention Heads: {config['num_attention_heads']}")
        model = AutoModelForCausalLM.from_pretrained(self.model)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        table1 = Table(title=" === Model info === ")
        table1.add_row(f"Total parameters:{total:,}")
        table1.add_row(f"trainable parameters: {trainable:,}")

        Console().print(Columns([table, table1]))
        table2 = Table(title=" -- model layer info --- ")
        for name, p in list(model.named_parameters())[:15]:
            table2.add_row(f" {name:60s} shape={list(p.shape)}")
        Console().print(table2)
