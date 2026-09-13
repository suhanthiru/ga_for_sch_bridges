import pytest
import torch

from sb.envs import terrain as TR
from sb.envs.gen_task import GenTask
from sb.policies.demos import load_demos, make_demos


@pytest.fixture(scope="session")
def cpu():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def small_demos(tmp_path_factory, cpu):
    p = tmp_path_factory.mktemp("demos") / "demos_L1.npz"
    make_demos("L1", 16, p, cpu, seed=4242)
    return load_demos(p, cpu)


@pytest.fixture
def l1_task(cpu):
    return GenTask(TR.Layout("L1"), "none", 8, 1, cpu)
