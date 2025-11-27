# graph/state.py
from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class WorkflowState:
    vars: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)  # name -> filepath
    history: List[Dict[str, Any]] = field(default_factory=list)
