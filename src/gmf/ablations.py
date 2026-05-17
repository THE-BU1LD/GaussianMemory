
from __future__ import annotations
from pathlib import Path
from copy import deepcopy
import json

from .benchmark import run_suite
from .utils import save_json


ABLATIONS = {
    "full": {"memory": True, "controller": True, "refine": True, "dual_memory": True},
    "no_memory": {"memory": False, "controller": True, "refine": True, "dual_memory": False},
    "no_controller": {"memory": True, "controller": False, "refine": True, "dual_memory": True},
    "no_refine": {"memory": True, "controller": True, "refine": False, "dual_memory": True},
    "single_memory": {"memory": True, "controller": True, "refine": True, "dual_memory": False},
}


def ablation_plan():
    return deepcopy(ABLATIONS)
