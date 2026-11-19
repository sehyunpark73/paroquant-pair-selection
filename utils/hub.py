import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

REPO_NAME = "sehyunpark73/paroquant-pair-selection"


def load_rows(
    hub_path: str,
) -> list[dict]:
    return json.loads(Path(hf_hub_download(REPO_NAME, f"{hub_path}.json")).read_text())


def get_row_key(
    row: dict,
) -> tuple[str, int]:
    return row["arm"], row["seed"]


def save_rows(
    hub_path: str,
    rows: list[dict],
) -> None:
    # 설정 하나가 파일 하나이고, 안의 행은 arm과 seed마다 하나
    # 같은 파일의 기존 행과 합쳐 한 commit으로 올림
    api = HfApi()
    merged = {get_row_key(row): row for row in load_rows(hub_path)} if api.file_exists(REPO_NAME, f"{hub_path}.json") else {}
    merged |= {get_row_key(row): row for row in rows}
    data = json.dumps([merged[key] for key in sorted(merged)], indent=2).encode()
    api.upload_file(path_or_fileobj=data, path_in_repo=f"{hub_path}.json", repo_id=REPO_NAME, commit_message=f"save {hub_path}.json")
    print(f"saved {hub_path}.json")


def get_hub_path(
    job: tuple[str, str, int],
) -> str:
    # job은 (hub 경로, arm, seed)
    return job[0]


def get_pending_jobs(
    jobs: list[tuple[str, str, int]],
) -> list[tuple[str, str, int]]:
    # jobs는 (hub 경로, arm, seed), 이미 저장된 행은 뺌
    files = set(HfApi().list_repo_files(REPO_NAME))
    done = set()
    for hub_path in dict.fromkeys(hub_path for hub_path, _, _ in jobs):
        if f"{hub_path}.json" in files:
            done |= {(hub_path, *get_row_key(row)) for row in load_rows(hub_path)}
    return [job for job in jobs if job not in done]


def parse_value(
    value: str,
) -> int | float | str:
    # 예시) "8" → 8, "0.01" → 0.01, "self_attn.q_proj" → "self_attn.q_proj"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def load_results(
    experiment_name: str,
) -> list[dict]:
    # 실험의 모든 결과를 받아, 경로의 모델과 설정을 행마다 붙임
    # 예시) "rotation/Qwen3-0.6B/layer=0,...,num_rotations=8.json"의 행 → {"model_name": "Qwen3-0.6B", "layer": 0, ..., "arm": "random", "loss": 0.53}
    result_dir = Path(snapshot_download(REPO_NAME, allow_patterns=[f"{experiment_name}/*/*.json"])) / experiment_name
    rows = []
    for file in sorted(result_dir.glob("*/*.json")):
        settings = {"model_name": file.parent.name}
        for setting in file.stem.split(","):
            name, value = setting.split("=", 1)
            settings[name] = parse_value(value)
        rows.extend(settings | row for row in json.loads(file.read_text()))
    return rows
