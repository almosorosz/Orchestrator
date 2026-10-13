from dataclasses import dataclass
from typing import Optional


@dataclass
class AppState:
    current_project_folder: Optional[str] = None
    experiment_count: int = 0
    running: bool = False
    current_stage: Optional[str] = None