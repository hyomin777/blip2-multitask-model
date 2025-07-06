import os
import argparse
import yaml
from easydict import EasyDict
from dataset import ImageDataset
from trainer import ModelTrainer
from model import Blip2MultiTask, Mode


def load_config(path: str):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return EasyDict(config)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["sentiment", "category", "text", "all"], required=True)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def main(rank, world_size, args):
    if args.mode == "sentiment":
        mode = Mode.SENTIMENT
    elif args.mode == "category":
        mode = Mode.CATEGORY
    elif args.mode == "text":
        mode = Mode.TEXT
    else:
        mode = Mode.ALL

    config = load_config(args.config)
    trainer = ModelTrainer(config, rank, world_size, Blip2MultiTask, ImageDataset, mode, args.resume)
    trainer.train()


if __name__ == '__main__':
    args = parse_args()

    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    main(rank, world_size, args)

