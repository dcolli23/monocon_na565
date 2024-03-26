from typing import Dict, List
from pathlib import Path
import subprocess
from pprint import pprint
import itertools
import time

import torch

# from submission import prep_for_submission

ROOT_DIR = Path(__file__).parent.resolve()


def find_final_checkpoint_file(trial_dir: Path):
    checkpoint_dir = trial_dir / "checkpoints"
    checkpoint_file_candidates = list(checkpoint_dir.glob("*_final.pth"))

    if len(checkpoint_file_candidates) > 1:
        raise RuntimeError("Found more than one 'final' checkpoint.")
    elif len(checkpoint_file_candidates) == 0:
        raise FileNotFoundError("Couldn't find 'final' checkpoint file.")

    return checkpoint_file_candidates[0]


def launch_evaluation_job(trial_dir: Path, gpu_id: int):
    """Evaluates the experiment trial recorded in `trial_dir`"""
    kwargs = {
        "config_file": (trial_dir / "config.yaml").as_posix(),
        "checkpoint_file": find_final_checkpoint_file(trial_dir).as_posix(),
        "gpu_id": str(gpu_id),
        "save_dir": (trial_dir / "test_results").as_posix()
    }

    # log_file = (trial_dir / "test_evaluation.log").as_posix()

    print("Launching submission prep job with arguments:")
    pprint(kwargs, indent=4, width=100, sort_dicts=False)
    print()

    cmd = [
        "python", "submission.py", kwargs["config_file"],
        kwargs["checkpoint_file"], "--save-dir", kwargs["save_dir"],
        "--gpu-id", kwargs["gpu_id"]
    ]
    job = subprocess.Popen(cmd)

    # Pause just to make sure nothing weird happens with launching multiple jobs.
    time.sleep(0.5)

    return job


# def increment_gpu_id(gpu_id: int, num_gpus: int):
#     return (gpu_id + 1) % num_gpus


def evaluate_all_trials(experiment_root_dir: Path,
                        trials_to_evaluate: Dict[str, List[int]],
                        gpu_ids: List[int]):
    gpu_id_generator = itertools.cycle(gpu_ids)
    jobs: List[subprocess.Popen] = []
    for experiment_name, trials in trials_to_evaluate.items():
        experiment_dir = experiment_root_dir / experiment_name

        for trial in trials:
            trial_dir = experiment_dir / f"trial_{trial}"
            gpu_id = next(gpu_id_generator)
            job = launch_evaluation_job(trial_dir, gpu_id)
            jobs.append(job)

    # finished_jobs = []
    # while True:
    #     for i, job in enumerate(jobs):
    #         if i in finished_jobs:
    #             continue

    #         poll_result = job.poll()

    #         if poll_result == 0:
    #             finished_jobs.append(i)
    #         elif poll_result > 0:
    #             print(
    #                 f"WARNING: Job #{i} failed with returncode {poll_result}")

    #     # Wait a bit so we don't continually poll jobs.
    #     time.sleep(1.0)


if __name__ == "__main__":
    experiment_root_dir = ROOT_DIR / "exps_hyperparam_tuning"
    trials_to_evaluate = {
        "double_lr_half_num_epoch": [1, 4, 5],
        "higher_weight_decay": [1, 4, 5]
    }
    gpu_ids = [i for i in range(torch.cuda.device_count())]

    evaluate_all_trials(experiment_root_dir, trials_to_evaluate, gpu_ids)
