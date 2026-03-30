from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Pathconfig(BaseModel):
    Model_id: str = Field(default="HuggingFaceTB/SmolLM-135M")
    Local_path: Path = Field(default=Path("./models/smollm-135m-base"))
    train_file: Path = Field(default=Path("./workspace/train.jsonl"))
    val_file: Path = Field(default=Path("./workspace/validation.jsonl"))
    output_dir: Path = Field(default=Path("./workspace/SmolLM-135M-note-narrow"))
    workspace: Path = Field(default=Path("./workspace"))

    @model_validator(mode="after")
    def ensure_dirs_exist(self) -> "Pathconfig":
        # loop through class and fix any parent path is missing
        for v in self.__dict__.values():
            if isinstance(v, Path):
                folder_ = v.parent if v.suffix else v
                folder_.mkdir(parents=True, exist_ok=True)
        return self


class settings:
    pass
