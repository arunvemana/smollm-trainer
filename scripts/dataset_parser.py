import json
import random
from pathlib import Path
from typing import TypedDict
from pydantic import TypeAdapter, ValidationError


class RawData(TypedDict):
    note: str
    summary: str


workspace = "./workspace"


def save_to_jsonl(data: list[str], output_filepath):
    with open(output_filepath, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    print(f"Saved {len(data)} examples to {output_filepath}")


def split_and_save(formatted_data: list[str], train_ration=0.9):
    random.shuffle(formatted_data)
    split_index = int(len(formatted_data) * train_ration)
    train_data = formatted_data[:split_index]
    vali_data = formatted_data[split_index:]
    save_to_jsonl(train_data, Path(workspace) / "train.jsonl")
    save_to_jsonl(vali_data, Path(workspace) / "validation.jsonl")

def check_consistency(raw_data:list[RawData]) -> bool:
    seen_notes = {}
    duplicates,inconsistencies = 0,0
    for item in raw_data:
        note = item['note'].strip().lower()
        summary = item['summary'].strip()
        previous_summary = seen_notes.get(note)
        if previous_summary is not None:
            duplicates +=1
            if seen_notes[note] !=summary:
                print(f"Inconsistency detected")
                print(f"For Note:{note}")
                print(f"already seen summary:{seen_notes[note]}")
                print(f"next one: {summary}")
                print("-"*30)
                inconsistencies +=1
        else:
            seen_notes[note] = summary
    print(f"Total examples {len(raw_data)} \nduplicates {duplicates - inconsistencies}")
    if inconsistencies == 0:
        return True
    return False

def process_raw_data(filename: str = "raw_train_notes.json"):
    file_path: Path = Path(workspace) / filename
    if not file_path.is_file():
        print(f"Error:{filename} not found! Create raw data file")
        return
    with open(file_path, 'r', encoding="utf-8") as f:
        raw_data = json.load(f)
    adapter = TypeAdapter(list[RawData])
    try:
        v_raw_data = adapter.validate_python(raw_data)
    except ValidationError as e:
        print(f"raw data is not in proper format: {e}")
        return
    if check_consistency(v_raw_data):
        formatted_lines: list = []
        for item in v_raw_data:
            f_text: str = f"### NOTE: {item['note']} ### SUMMARY: {item['summary']}<|endoftext|>"
            formatted_lines.append(f_text)
        split_and_save(formatted_lines)


if __name__ == "__main__":
    process_raw_data()
