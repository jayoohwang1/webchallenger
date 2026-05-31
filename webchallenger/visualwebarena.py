""" Re-export the classes from the visualwebarena submodule. """

import sys
from pathlib import Path

project_path = Path(__file__).parent
sys.path.append(str(project_path / "benchmarks" / "visualwebarena"))  # nopep8

from webchallenger.benchmarks.visualwebarena.browser_env.actions import *  # nopep8
from webchallenger.benchmarks.visualwebarena.browser_env.envs import *  # nopep8
from webchallenger.benchmarks.visualwebarena.browser_env.trajectory import *  # nopep8
from webchallenger.benchmarks.visualwebarena.evaluation_harness.evaluators import *
from webchallenger.benchmarks.visualwebarena.evaluation_harness.image_utils import *

sys.path = sys.path[:-1]


__all__ = [
    ScriptBrowserEnv,
    Action,
    ActionTypes,
    DetachedPage,
    create_stop_action,
    create_none_action,
    execute_action,
    Trajectory,
]
