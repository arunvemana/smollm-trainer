from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Pathconfig(BaseModel):
    model_id: str = Field(default="")
    local_path: Path = Field(default=Path("./models/smollm-135m-base"))
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


class TrainingConfig(BaseModel):
    """
    max_length is taken from the dataset_parser output depends on the train.jsonl data.
    epochs is calc -> 90 example / 8 batch = 12 steps per echos 12 * 12 = 144 -> in range of narrow fine tunning (100 - 200) steps.
    Lr -> learning rate is standard for fine tuning
    r=4 means 576x4 + 4x576 = 4,608 new parameters per targeted layer With 2 targeted layers x 30 transformer
    layers = ~276K trainable params That's 0.2% of the model - genuinely narrow
    """

    max_length: int = Field(default=0, ge=1)
    epochs: int = Field(default=12, gt=0)
    batch_size: int = Field(default=8, gt=0)
    learning_rate: float = 2e-4
    lora_r: int = 4
    lora_alpha: int = 8


class Settings(BaseSettings):
    """
    Master Setting combine both path and train config
    """

    train: TrainingConfig = TrainingConfig()
    envpath: Pathconfig = Pathconfig()

    model_config = SettingsConfigDict(
        env_file=".env", env_nested_delimiter="__", case_sensitive=False
    )

    @model_validator(mode="after")
    def check_env_file(self) -> "Settings":
        env_file = self.model_config.get("env_file")
        if env_file and not Path(str(env_file)).exists():
            print(Path(str(env_file)).absolute())
            raise FileNotFoundError("Critical: .env file not found")

        print(".env file loaded")
        return self


setting = Settings()
