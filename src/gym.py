import numpy as np
import torch
import torch.utils
import torch.utils.data
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from metrics import mse, ssim


class Gym:
    def __init__(
        self,
        model,
        optim,
        loss,
        metrics=dict(mse=mse, ssim=ssim),
        batch_size=32,
        device=None,
        random_state=42,
    ):
        self.model = model
        self.optim = optim
        self.loss = loss
        self.metrics = metrics
        self.device = device if device is not None else self._device_get()
        self.batch_size = batch_size
        self.random_state = random_state
        self.history = dict()

    def _device_get(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def _init_history(self):
        self.history = dict(train_loss=[], val_loss=[])
        for name in self.metrics.keys():
            self.history[f"train_{name}"] = []
            self.history[f"val_{name}"] = []
        return self.history

    def _display_report(self, epoch_index):
        report = f"""Epoch {epoch_index}:
  train_loss={self.history['train_loss'][-1]:.5f} \
val_loss={self.history['val_loss'][-1]:.5f}
"""
        for name in self.metrics.keys():
            train_key = f"train_{name}"
            val_key = f"val_{name}"
            report += (
                f"  {train_key}={self.history[train_key][-1]:.5f}"
                f" {val_key}={self.history[val_key][-1]:.5f}\n"
            )
        print(report)

    def _display_performance(self):
        _, axes = plt.subplots(
            len(self.history) // 2, 1, figsize=(10, 5 * len(self.metrics))
        )
        epochs = torch.arange(len(self.history["train_loss"]))
        axes[0].plot(epochs, self.history["train_loss"], label="Train Loss")
        axes[0].plot(epochs, self.history["val_loss"], label="Val Loss")
        axes[0].set_title("Train vs Validation Loss")
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        for ii, name in enumerate(self.metrics.keys()):
            axes[ii + 1].plot(
                epochs, self.history[f"train_{name}"], label=f"Train {name}"
            )
            axes[ii + 1].plot(epochs, self.history[f"val_{name}"], label=f"Val {name}")
            axes[ii + 1].set_title(f"Train vs Validation {name.title()}")
            axes[ii + 1].set_xlabel("Epochs")
            axes[ii + 1].set_ylabel("Value")
            axes[ii + 1].legend()
        plt.tight_layout()
        plt.show()

    def _tensor_to_image(self, tensor):
        return tensor.detach().squeeze().numpy()

    def _display_predictions(self, loader):
        inp, exp = next(iter(loader))
        inp = inp.to(self.device)
        exp = exp.to(self.device)
        out = self.model(inp)
        inp = inp.cpu()
        exp = exp.cpu()
        out = out.cpu()
        batch_size = loader.batch_size
        _, axes = plt.subplots(nrows=batch_size, ncols=3, figsize=(15, 5 * batch_size))
        for ii in range(batch_size):
            axes[ii, 0].imshow(self._tensor_to_image(inp[ii]), cmap="gray")
            axes[ii, 0].set_title(f"Input {ii}")
            axes[ii, 0].axis("off")
            axes[ii, 1].imshow(self._tensor_to_image(out[ii]), cmap="gray")
            axes[ii, 1].set_title(f"Predicted {ii}")
            axes[ii, 1].axis("off")
            axes[ii, 2].imshow(self._tensor_to_image(exp[ii]), cmap="gray")
            axes[ii, 2].set_title(f"Expected {ii}")
            axes[ii, 2].axis("off")
        plt.tight_layout()
        plt.show()

    def _train_one_epoch(self, loader):
        metric_values = torch.zeros(
            len(self.metrics) + 1, dtype=torch.float32, device=self.device
        )
        for inp, exp in loader:
            inp = inp.to(self.device)
            exp = exp.to(self.device)
            self.optim.zero_grad()
            out = self.model(inp)
            loss = self.loss(out, exp)
            metric_values[0] += loss.item()
            for ii, fn in enumerate(self.metrics.values()):
                metric_values[ii + 1] += fn(exp, out).item()
            loss.backward()
            self.optim.step()
        metric_values = metric_values / len(loader)
        self.history["train_loss"].append(metric_values[0].item())
        for ii, name in enumerate(self.metrics.keys()):
            self.history[f"train_{name}"].append(metric_values[ii + 1].item())
        return metric_values[0]  # train loss

    def _prepare_loaders(self, X, y, val_size=0.1):
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        X = np.expand_dims(X, axis=1)
        y = np.expand_dims(y, axis=1)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size, random_state=self.random_state
        )
        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train)
        X_val = torch.from_numpy(X_val)
        y_val = torch.from_numpy(y_val)
        loader_train = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=True,
        )
        loader_val = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=self.batch_size,
            shuffle=True,
        )
        return loader_train, loader_val

    def train(self, X, y, epochs, val_size=0.1, verbose=1):
        self._init_history()
        loader_train, loader_val = self._prepare_loaders(X, y, val_size)
        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)
        for epoch_ii in range(epochs):
            self.model.train(True)
            self._train_one_epoch(loader_train)
            self.model.eval()
            metric_values = torch.zeros(
                len(self.metrics) + 1, dtype=torch.float32, device=self.device
            )
            with torch.no_grad():
                for inp, exp in loader_val:
                    inp = inp.to(self.device)
                    exp = exp.to(self.device)
                    out = self.model(inp)
                    metric_values[0] += self.loss(out, exp).detach()
                    for jj, fn in enumerate(self.metrics.values()):
                        metric_values[jj + 1] += fn(out, exp).detach()
                metric_values = metric_values / len(loader_val)
                self.history["val_loss"].append(metric_values[0].item())
                for jj, name in enumerate(self.metrics.keys()):
                    self.history[f"val_{name}"].append(metric_values[jj + 1].item())
            if verbose > 0:
                self._display_report(epoch_ii)
