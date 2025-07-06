import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter
from scheduler import CosineAnnealingWarmUpRestarts
from augmentation import Transform
from dataset import TransformedSubset
from model import Mode, Blip2MultiTask


os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ModelTrainer:
    def __init__(
            self,
            config,
            rank: int = 0,
            world_size: int = 1,
            model_cls: nn.Module = Blip2MultiTask,
            dataset_cls: Dataset = None,
            mode = Mode,
            resume = None
        ):
        self.config = config

        self.rank = rank
        self.world_size = world_size

        self.setup()

        self.is_main = self.rank == 0
        self.device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")

        self.model_cls = model_cls
        self.model = self._create_model().to(self.device)

        self.dataset_cls = dataset_cls

        self.mode = mode

        self.best_loss = float('inf')
        self.step = 1
        self.start_epoch = 1

        self.scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None

        self.writer = None
        if self.config.log_dir:
            self.writer = SummaryWriter(log_dir=Path(self.config.log_dir)) if self.is_main else None

        self.resume = resume

    def setup(self):
        if not torch.distributed.is_initialized():
            dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo",
                                   rank=self.rank, world_size=self.world_size)
        if torch.cuda.is_available():
            torch.cuda.set_device(self.rank)

    def cleanup(self):
        if torch.distributed.is_initialized():
            dist.destroy_process_group()

    def _wrap_model(self, model):
        return DDP(model, device_ids=[self.rank], find_unused_parameters=True)

    def _create_model(self):
        return self.model_cls(self.config.model)

    def _create_dataset(self):
        return self.dataset_cls(root_dir=self.config.dataset)

    def _create_optimizer(self, parameters):
        self.optimizer = torch.optim.AdamW(parameters, lr=float(self.config.optimizer.lr), weight_decay=self.config.optimizer.weight_decay)

    def _create_scheduler(self):
        return CosineAnnealingWarmUpRestarts(self.optimizer, T_0=self.config.scheduler.T_0, T_mult=self.config.scheduler.T_mult, T_up=self.config.scheduler.T_up, eta_max=float(self.config.scheduler.eta_max), gamma=self.config.scheduler.gamma)

    def _create_dataloader(self, dataset):
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True
        )
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.training.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return dataloader, sampler

    def _create_transform(self, augmentation):
        return Transform(self.config.model, augmentation)

    def _save_weights(self, model, path):
        if not self.is_main:
            return

        weights_path = Path(self.config.output_dir) / 'weights'
        weights_path.mkdir(parents=True, exist_ok=True)

        model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

        torch.save(model_state, os.path.join(weights_path, path))
        print(f"[INFO] Saved model weights to {os.path.join(weights_path, path)}")

    def _save_checkpoint(self, model, optimizer, epoch, path):
        if not self.is_main:
            return

        checkpoint_path = Path(self.config.output_dir) / 'checkpoints'
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        optimizer_state = optimizer.state_dict()

        checkpoint = {
            "model": model_state,
            "optimizer": optimizer_state,
            "step": self.step,
            "loss": self.best_loss,
            "epoch": epoch,
        }

        torch.save(checkpoint, os.path.join(checkpoint_path, path))
        print(f"[INFO] Saved checkpoint to {os.path.join(checkpoint_path, path)}")

    def _load_checkpoint(self, model, optimizer):
        if not os.path.exists(self.resume):
            if self.is_main:
                print(f"[ERROR] Checkpoint file {self.resume} not found!")
            return False

        try:
            map_location = {"cuda:0": f"cuda:{self.rank}"}
            checkpoint = torch.load(self.resume, map_location=map_location)

            if hasattr(model, "module"):
                model.module.load_state_dict(checkpoint["model"])
            else:
                model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])

            self.start_epoch = checkpoint["epoch"] + 1
            self.step = checkpoint["step"] + 1
            self.best_loss = checkpoint["loss"]

            if self.is_main:
                print(f"[INFO] Resumed from {self.resume} | epoch: {self.start_epoch} | step: {self.step} | loss: {self.best_loss}")
            return True

        except Exception as e:
            if self.is_main:
                print(f"[ERROR] Failed to load checkpoint: {e}")
            return False

    def train(self):
        parameters = self.model.get_trainable_parameters(self.mode)
        self._create_optimizer(parameters)
        self.model = self._wrap_model(self.model)
        criterion = nn.CrossEntropyLoss()
        scheduler = self._create_scheduler()

        base_dataset = self._create_dataset()
        dataset_size = len(base_dataset)
        train_size = int(dataset_size * 0.8)
        validation_size = dataset_size - train_size

        train_indices, validation_indices = random_split(
            range(len(base_dataset)),
            [train_size, validation_size],
            generator=torch.Generator().manual_seed(42)
        )

        train_transform = self._create_transform(augmentation=True)
        validation_transform = self._create_transform(augmentation=False)

        train_dataset = TransformedSubset(Subset(base_dataset, train_indices), train_transform)
        validation_dataset = TransformedSubset(Subset(base_dataset, validation_indices), validation_transform)

        train_dataloader, sampler = self._create_dataloader(train_dataset)
        validation_dataloader, _ = self._create_dataloader(validation_dataset)
        print(f'[Start] Train Dataset Size: {train_size} | Validation Dataset Size: {validation_size}')

        for epoch in range(self.start_epoch, self.config.training.num_epochs + 1):
            self.model.train()
            sampler.set_epoch(epoch)
            scheduler.step(epoch)

            total_loss = 0.0

            if self.is_main:
                pbar = tqdm(total=len(train_dataloader))

            for batch in train_dataloader:
                loss = self.train_step(batch, criterion)
                total_loss += loss.item()

                if self.is_main:
                    pbar.update(1)

            avg_loss = total_loss / len(train_dataloader)
            print(f"Train Loss: {avg_loss:.4f}")

            # Validation
            self.model.eval()
            if self.is_main:
                val_loss = 0.0
                correct = 0
                total = 0

                for batch in validation_dataloader:
                    images = batch['image'].to(self.device)
                    labels = batch['label'].to(self.device)

                    with torch.no_grad():
                        outputs = self.model(images, self.mode)
                        loss = criterion(outputs, labels)

                    val_loss += loss.item()

                    batch_correct, batch_total = self.compute_accuracy(outputs, labels)
                    correct += batch_correct
                    total += batch_total

                avg_val_loss = val_loss / len(validation_dataloader)
                val_accuracy = correct / total
                print(f"[Epoch: {epoch}] Val Loss: {avg_val_loss:.4f} | Accuracy: {val_accuracy*100:.2f}%")

                if self.best_loss > avg_val_loss:
                    self.best_loss = avg_val_loss
                    self._save_weights(self.model, f'model.pth')
                    print(f'[Save] Best Loss: {avg_val_loss:.4f}')

                if self.writer is not None:
                    self.writer.add_scalar("val_loss", avg_val_loss, epoch)
                    self.writer.add_scalar("val_accuracy", val_accuracy, epoch)
                    self.writer.add_scalar("train_loss", avg_loss, epoch)
                    self.writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'], epoch)

    def train_step(self, batch, criterion):
        images = batch['image'].to(self.device)
        labels = batch['label'].to(self.device)

        outputs = self.model(images, self.mode)
        loss = criterion(outputs, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def compute_accuracy(self, logits, labels):
        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).sum().item()
        total = labels.size(0)
        return correct, total
