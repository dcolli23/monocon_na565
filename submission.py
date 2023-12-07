from typing import TYPE_CHECKING, Optional
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import pandas as pd

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


def split_merged_results(out_file_presplit: Path,
                         out_file_normal: Path,
                         out_file_bonus: Path,
                         split_frame: int = 423):
    """Splits the full evaluation into normal and bonus credit submission TXT files

    All detection results prior to `split_frame` are designated as "normal credit" detections and
    all detection results at `split_frame` and later are designated "bonus credit".
    """
    # Read all merged detections.
    df_presplit = pd.read_csv(out_file_presplit.as_posix(), delim_whitespace=True, header=None)

    # Split the file contents.
    df_normal = df_presplit[df_presplit[0] < split_frame]
    df_bonus = df_presplit[df_presplit[0] >= split_frame]

    # Write the normal credit detections.
    df_normal.to_csv(out_file_normal, sep=' ', header=False, index=False)
    print("Wrote normal credit detections to:", out_file_normal.as_posix())

    # Write the bonus credit detections.
    df_bonus.to_csv(out_file_bonus, sep=' ', header=False, index=False)
    print("Wrote extra credit detections to:", out_file_bonus.as_posix())


def main(config_file: str, checkpoint_file: str, gpu_id: int, save_dir: Optional[str] = None):
    # Load Config
    cfg = load_cfg(config_file)
    cfg.GPU_ID = gpu_id

    # Configure output paths.
    if save_dir is None:
        save_dir = "exps/test_results"
    save_dir = Path(save_dir)
    output_dir_raw = save_dir / "raw_results"
    output_file_merged_presplit = save_dir / "merged_test_pre_split.txt"
    output_file_merged_normal = save_dir / "merged_test_normal.txt"
    output_file_merged_bonus = save_dir / "merged_test_bonus.txt"

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

    merge_detections(output_dir_raw, output_file_merged_presplit)

    split_merged_results(output_file_merged_presplit, output_file_merged_normal,
                         output_file_merged_bonus)


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
