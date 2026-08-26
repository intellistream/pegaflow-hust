"""Host-only regression tests for the performance harness."""

from argparse import Namespace
from pathlib import Path

from scripts.run_perf_base import Experiment, build_reproduce_command

REPO_ROOT = Path(__file__).resolve().parent


def _args() -> Namespace:
    return Namespace(
        cycles=1,
        requests_per_phase=1,
        pool_size="16gb",
        min_free_gb=28,
        num_instances=8,
        model="/models/Qwen3-8B",
        verify_repro=False,
        out=None,
    )


def test_reproduce_command_uses_real_runner_and_sweep_selector() -> None:
    experiment = Experiment(
        id="t3-h50",
        title="T3 50%",
        reproduce_args=["--ratios", "50"],
    )

    command = build_reproduce_command(
        experiment,
        _args(),
        REPO_ROOT,
        REPO_ROOT.parent / "vllm-hust",
        REPO_ROOT.parent / "vllm-ascend-hust",
        Path("/opt/conda"),
        "vllm-hust-dev",
        REPO_ROOT / "scripts" / "run_perf_t3_hitrate.py",
    )

    assert "scripts/run_perf_t3_hitrate.py --ratios 50" in command
    assert "run_perf_t3-h50_baseline.py" not in command
    assert "--ascend-root" in command
    assert "--conda-env vllm-hust-dev" in command


def test_reproduce_command_quotes_paths() -> None:
    args = _args()
    args.model = "/models/Qwen 3-8B"

    command = build_reproduce_command(
        Experiment(id="t1", title="T1"),
        args,
        REPO_ROOT,
        REPO_ROOT.parent / "vllm hust",
        REPO_ROOT.parent / "vllm ascend hust",
        Path("/opt/conda root"),
        "vllm dev",
        REPO_ROOT / "scripts" / "run_perf_t1_baseline.py",
    )

    assert "'/models/Qwen 3-8B'" in command
    assert "'vllm dev'" in command
