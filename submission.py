from typing import TYPE_CHECKING, Optional
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

REPO_DIR = Path(__file__).parent
REPO_ROOT = REPO_DIR.parent
sys.path.append(REPO_ROOT.as_posix())

from engine.monocon_engine import MonoconEngine
from utils.engine_utils import (tprint, load_cfg, generate_random_seed, set_random_seed,
                                move_data_device)
from dataset.monocon_dataset import MonoConDataset
from utils.kitti_convert_utils import kitti_3d_to_file, kitti_file_to_3d
from merger import merge_detections

if TYPE_CHECKING:
    from model.detector import MonoConDetector


def set_engine_test_dataset(engine: MonoconEngine, data_dir: str, max_objs: int, batch_size: int,
                            num_workers: int):
    """Sets the MonoconEngine's test dataset to the true test dataset"""
    # Overwrite the "test" dataset loaded from the config because the config file likely used the
    # validation dataset. This is a weird way to configure things in my opinion.
    dataset = MonoConDataset(data_dir, "test", max_objs=max_objs)
    engine.test_dataset = dataset
    engine.test_loader = DataLoader(engine.test_dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    shuffle=False,
                                    collate_fn=dataset.collate_fn,
                                    drop_last=False)


@torch.no_grad()
def evaluate_minibatch(model: 'MonoConDetector', test_data, save_dir: str, device: str):
    """Evaluates and saves converted results for a minibatch"""
    test_data = move_data_device(test_data, device)
    eval_results = model.batch_eval(test_data)
    kitti_3d_to_file(eval_results, test_data["img_metas"], folder=save_dir, single_file=False)


@torch.no_grad()
def generate_raw_detections(engine: MonoconEngine, output_dir_raw: Path, device: str):
    if engine.model.training:
        engine.model.eval()
        print("Model converted to eval mode.")

    for test_data in tqdm(engine.test_loader, desc="Generating Raw Detections"):
        evaluate_minibatch(engine.model, test_data, output_dir_raw.as_posix(), device)


def main(config_file: str, checkpoint_file: str, gpu_id: int, save_dir: Optional[str] = None):
    # Load Config
    cfg = load_cfg(config_file)
    cfg.GPU_ID = gpu_id

    # Configure output paths.
    if save_dir is None:
        save_dir = "exps/test_results"
    save_dir = Path(save_dir)
    output_dir_raw = save_dir / "raw_results"
    output_file_merged = save_dir / "merged_test_normal.txt"

    device = f'cuda:{gpu_id}'

    # Set Benchmark
    # If this is set to True, it may consume more memory. (Default: True)
    if cfg.get('USE_BENCHMARK', True):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        tprint(f"CuDNN Benchmark is enabled.")

    # Set Random Seed
    seed = cfg.get('SEED', -1)
    seed = generate_random_seed(seed)
    set_random_seed(seed)
    tprint(f"Using Random Seed {seed}")

    engine = MonoconEngine(cfg, auto_resume=False, is_test=True)
    engine.load_checkpoint(checkpoint_file, verbose=True)

    set_engine_test_dataset(engine, cfg.DATA.ROOT, cfg.MODEL.HEAD.MAX_OBJS, cfg.DATA.BATCH_SIZE,
                            cfg.DATA.NUM_WORKERS)

    generate_raw_detections(engine, output_dir_raw, device)

    merge_detections(output_dir_raw, output_file_merged)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file",
                        type=str,
                        help="Path to the YAML file used to train the model.")
    parser.add_argument("checkpoint_file",
                        type=str,
                        help="Path to the PyTorch checkpoint file (.pth) you wish to test")
    parser.add_argument("--save-dir",
                        type=str,
                        default=None,
                        help="Path to where you would like test outputs to be stored.")
    parser.add_argument("--gpu-id", type=int, default=0)

    args = parser.parse_args()

    main(**vars(args))
