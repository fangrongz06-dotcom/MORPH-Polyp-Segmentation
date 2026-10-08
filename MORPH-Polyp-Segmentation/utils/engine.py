from __future__ import annotations

import json
import random
from pathlib import Path
import numpy as np
import torch
from utils.common import git_commit, manifest_hash, source_code_hash
import csv
import time
from utils.precision import NumericalForwardError, autocast_context
from utils.metrics import TrainingMetricAccumulator
from utils.prompts import render_dense_click_map, sample_first_click, sample_second_click
from utils.metrics import compute_metrics


"""Guard optimizer updates; replay unstable AMP batches without skipping data."""



def snapshot_forward_state(model):
    return {
        'buffers': {name: value.detach().clone() for name, value in model.named_buffers()},
        'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_forward_state(model, snapshot):
    with torch.no_grad():
        for name, value in model.named_buffers():
            value.copy_(snapshot['buffers'][name])
    random.setstate(snapshot['python_rng'])
    np.random.set_state(snapshot['numpy_rng'])
    torch.set_rng_state(snapshot['torch_rng'])
    if snapshot['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all(snapshot['cuda_rng'])


def checked_gradient_norm(model, clip_norm=None):
    parameters = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not parameters:
        raise RuntimeError('No trainable gradients were produced')
    if clip_norm is not None and (not np.isfinite(clip_norm) or clip_norm <= 0):
        raise ValueError('gradient_clip_norm must be finite and positive')
    # Finite loss does not imply finite gradients. Throw before Adam can consume
    # NaN/Inf, even when no clipping is requested by a legacy configuration.
    return torch.nn.utils.clip_grad_norm_(
        parameters, float('inf') if clip_norm is None else clip_norm,
        error_if_nonfinite=True)


def nonfinite_names(state):
    return [str(name) for name, value in state.items()
            if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex())
            and not bool(torch.isfinite(value).all())]


def invalid_running_variance_names(state):
    return [str(name) for name, value in state.items()
            if str(name).endswith('running_var') and torch.is_tensor(value)
            and bool((value < 0).any())]


def checked_batchnorm_statistics(model):
    statistics, variances = [], []
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.track_running_stats:
            statistics.extend([module.running_mean.reshape(-1), module.running_var.reshape(-1)])
            variances.append(module.running_var.reshape(-1))
    if statistics:
        # One synchronization, with small concatenated BN buffers; full model
        # parameter scans are reserved for diagnostics and checkpoint gates.
        valid = torch.isfinite(torch.cat(statistics)).all() & (torch.cat(variances) >= 0).all()
        if not bool(valid):
            bad = nonfinite_names(dict(model.named_buffers())) + invalid_running_variance_names(dict(model.named_buffers()))
            raise RuntimeError('Invalid BatchNorm running statistics: ' + ', '.join(bad[:20]))


def require_finite_checkpoint(checkpoint):
    bad = nonfinite_names(checkpoint['model']) + invalid_running_variance_names(checkpoint['model'])
    for parameter_id, state in checkpoint.get('optimizer', {}).get('state', {}).items():
        bad.extend('optimizer.' + str(parameter_id) + '.' + name for name in nonfinite_names(state))
    if bad:
        raise ValueError('Checkpoint contains non-finite state: ' + ', '.join(bad[:20]))


def record_failure(run_dir, model, batch, epoch, batch_index, stage, reason, recovered):
    identities = {}
    for key in ('dataset', 'image_id', 'image_path', 'mask_path'):
        if key in batch:
            value = batch[key]
            identities[key] = value if isinstance(value, (str, list, tuple)) else str(value)
    record = {
        'epoch': epoch, 'batch': batch_index + 1, 'stage': stage,
        'reason': str(reason), 'recovered_with_fp32': recovered,
        'samples': identities,
        'nonfinite_model_state': nonfinite_names(model.state_dict()),
        'negative_running_variance': invalid_running_variance_names(model.state_dict()),
        'nonfinite_gradients': [name for name, p in model.named_parameters()
                               if p.grad is not None and not bool(torch.isfinite(p.grad).all())],
    }
    activation_ranges = {}
    for name, module in model.named_modules():
        value = getattr(module, '_last_bn_input', None)
        if value is not None:
            finite = torch.isfinite(value)
            values = value[finite]
            details = {'shape': list(value.shape), 'dtype': str(value.dtype),
                       'finite_count': int(finite.sum()), 'total_count': value.numel(),
                       'min_finite': float(values.min()) if values.numel() else None,
                       'max_finite': float(values.max()) if values.numel() else None}
            variance = getattr(module, '_last_bn_variance', None)
            if variance is not None:
                finite_variance = variance[torch.isfinite(variance)]
                details['max_finite_batch_variance'] = float(finite_variance.max()) if finite_variance.numel() else None
            activation_ranges[name] = details
    record['batchnorm_activation_ranges'] = activation_ranges
    path = Path(run_dir) / 'nonfinite_diagnostics.jsonl'
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    return path


def save_checkpoint(path, model, optimizer, scheduler, epoch, seed, config, train_manifest, *, recovery_lineage=None):
    payload = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "epoch": int(epoch), "seed": int(seed),
        "config": config, "git_commit": git_commit(),
        "manifest_hash": manifest_hash(train_manifest),
        "source_code_sha256": source_code_hash(),
        "pytorch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "recovery_lineage": recovery_lineage,
        "rng": {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    require_finite_checkpoint(payload)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_checkpoint(path, model, optimizer=None, scheduler=None, *, restore_rng=False,
                    expected_config=None, expected_manifest=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if expected_config is not None and checkpoint.get("config") != expected_config:
        raise ValueError("Resume checkpoint config differs from the requested full config")
    if restore_rng and expected_config is not None and checkpoint.get("source_code_sha256") != source_code_hash():
        raise ValueError("Source code changed since checkpoint creation; exact resume is forbidden")
    if expected_manifest is not None and checkpoint.get("manifest_hash") != manifest_hash(expected_manifest):
        raise ValueError("Training manifest changed since checkpoint creation")
    if restore_rng and "rng" not in checkpoint:
        raise ValueError("Checkpoint lacks RNG state; exact resume cannot be claimed")
    require_finite_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None: optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None: scheduler.load_state_dict(checkpoint["scheduler"])
    if restore_rng:
        rng = checkpoint["rng"]
        random.setstate(rng["python"]); np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] is not None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA RNG checkpoint cannot be exactly resumed without CUDA")
            torch.cuda.set_rng_state_all(rng["cuda"])
    return checkpoint


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, config, device, run_dir):
        self.model, self.criterion = model, criterion
        self.optimizer, self.scheduler = optimizer, scheduler
        self.config, self.device = config, device
        self.run_dir = Path(run_dir); self.run_dir.mkdir(parents=True, exist_ok=True)
        self.last_epoch_metrics = {}
        self.recovery_lineage = None

    def train_epoch(self, loader, epoch):
        self.model.train()
        started = time.perf_counter()
        metrics = TrainingMetricAccumulator(
            threshold=float(self.config.get("execution", {}).get("training_metric_threshold", 0.5)),
            dice_smooth=float(self.config["train"].get("dice_smooth", 1e-6)))
        fp32_retries = 0
        for batch_index, batch in enumerate(loader):
            checked_batchnorm_statistics(self.model)
            execution = self.config.get("execution", {})
            nonblocking = bool(execution.get("non_blocking", False))
            image = batch["image"].to(self.device, non_blocking=nonblocking)
            target = batch["mask"].to(self.device, non_blocking=nonblocking)
            mode = self.config["train"]["prompt_mode"]
            dense = self._training_dense(image, batch["mask"], mode, epoch, batch_index)
            retry_enabled = execution.get('precision') == 'amp_bfloat16' and execution.get('nonfinite_fp32_retry', False)
            snapshot = snapshot_forward_state(self.model) if retry_enabled else None
            attempts = [execution, {**execution, 'precision': 'float32'}] if retry_enabled else [execution]
            first_failure = None
            for attempt_index, precision in enumerate(attempts):
                self.optimizer.zero_grad(set_to_none=True)
                output, loss = None, None
                failure = None
                stage = 'forward'
                try:
                    with autocast_context(precision, self.device):
                        output = self.model(image, dense_click_map=dense)
                    loss = self.criterion(output['logits'].float(), target.float())
                except NumericalForwardError as exc:
                    failure = str(exc)
                if failure is None:
                    stage = 'batchnorm_statistics'
                    try:
                        checked_batchnorm_statistics(self.model)
                    except RuntimeError as exc:
                        failure = str(exc)
                if failure is None and not bool(torch.isfinite(loss)):
                    stage = 'loss'
                    failure = 'Non-finite training loss'
                if failure is None:
                    loss.backward()
                    stage = 'gradient'
                    try:
                        checked_gradient_norm(self.model, execution.get('gradient_clip_norm'))
                    except RuntimeError as exc:
                        failure = str(exc)
                if failure is None:
                    # Check again after checkpoint recomputation in backward.
                    try:
                        checked_batchnorm_statistics(self.model)
                    except RuntimeError as exc:
                        stage = 'batchnorm_statistics'
                        failure = str(exc)
                if failure is None:
                    if attempt_index:
                        fp32_retries += 1
                        print(f'FP32 retry recovered epoch={epoch} batch={batch_index + 1}; no optimizer update was made for the failed BF16 attempt', flush=True)
                    break
                can_retry = attempt_index == 0 and retry_enabled
                diagnostic = record_failure(self.run_dir, self.model, batch, epoch, batch_index, stage, failure, False)
                if not can_retry:
                    self.optimizer.zero_grad(set_to_none=True)
                    raise RuntimeError(f'{failure} at epoch {epoch}, batch {batch_index + 1}; optimizer update blocked; see {diagnostic}')
                first_failure = (stage, failure)
                print(f'Unstable BF16 {stage} at epoch={epoch} batch={batch_index + 1}; retrying the same batch in FP32', flush=True)
                del output, loss
                self.optimizer.zero_grad(set_to_none=True)
                restore_forward_state(self.model, snapshot)
            if first_failure is not None:
                record_failure(self.run_dir, self.model, batch, epoch, batch_index, first_failure[0], first_failure[1], True)
            del snapshot
            self.optimizer.step()
            metrics.update(output["logits"], target, loss)
            interval = int(self.config.get("execution", {}).get("log_every", 0))
            if interval and (batch_index == 0 or (batch_index + 1) % interval == 0 or batch_index + 1 == len(loader)):
                running = metrics.compute()
                print(f"epoch={epoch} batch={batch_index + 1}/{len(loader)} loss={float(loss.detach()):.6f} "
                      f"train_dice_running={running['train_dice']:.6f} train_iou_running={running['train_iou']:.6f} "
                      f"seconds_per_batch={(time.perf_counter() - started)/(batch_index + 1):.3f}", flush=True)
        self.scheduler.step()
        self.last_epoch_metrics = metrics.compute()
        self.last_epoch_metrics['fp32_retry_batches'] = fp32_retries
        self.last_epoch_metrics["epoch_train_seconds"] = max(time.perf_counter() - started, 1e-12)
        self.last_epoch_metrics["train_images_per_second"] = metrics.count / self.last_epoch_metrics["epoch_train_seconds"]
        if torch.device(self.device).type == "cuda":
            self.last_epoch_metrics["peak_vram_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3
            self.last_epoch_metrics["peak_vram_reserved_gb"] = torch.cuda.max_memory_reserved() / 1024**3
        return self.last_epoch_metrics["loss"]

    def _training_dense(self, image, target, mode, epoch, batch_index):
        if mode == "none": return None
        rng = np.random.default_rng(int(self.config["seed"]) + epoch * 100000 + batch_index)
        click_lists = []
        for sample in target[:, 0].detach().cpu().numpy().astype(bool):
            if not sample.any() or (mode == "mixed" and rng.random() < 0.5):
                click_lists.append([])
            else:
                click_lists.append([sample_first_click(sample, rng)])
        dense = render_dense_click_map(click_lists, radius=int(self.config["model"]["click_radius"]), device=image.device) if any(click_lists) else None
        if mode == "iterative":
            self.model.eval()
            with torch.no_grad(), autocast_context(self.config.get("execution", {}), self.device):
                first = torch.sigmoid(self.model(image, dense)["logits"].float()).ge(0.5)
            self.model.train()
            for index, sample in enumerate(target[:, 0].detach().cpu().numpy().astype(bool)):
                second = sample_second_click(sample, first[index, 0].cpu().numpy(), rng)
                if second is not None: click_lists[index].append(second)
            dense = render_dense_click_map(click_lists, radius=int(self.config["model"]["click_radius"]), device=image.device) if any(click_lists) else None
        if mode not in {"one_click", "mixed", "iterative"}:
            raise ValueError(f"Unknown training prompt mode: {mode}")
        return dense

    def run(self, loader, start_epoch=1):
        rows = []
        log_path = self.run_dir / "train_log.csv"
        if start_epoch > 1 and log_path.is_file():
            with log_path.open(newline="", encoding="utf-8") as stream:
                # Logs may have been written just before a crash interrupted
                # checkpoint replacement. Keep only epochs committed in the
                # resume checkpoint; replay the remaining epochs once.
                rows = [row for row in csv.DictReader(stream) if int(row["epoch"]) < start_epoch]
        epochs = int(self.config["train"]["epochs"])
        for epoch in range(start_epoch, epochs + 1):
            started = time.perf_counter()
            if torch.device(self.device).type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            print(f"Starting epoch {epoch}/{epochs}", flush=True)
            learning_rate = self.optimizer.param_groups[0]["lr"]
            self.last_epoch_metrics = {}
            loss = self.train_epoch(loader, epoch)
            rows.append({"epoch": epoch, "loss": loss, **self.last_epoch_metrics, "lr": learning_rate,
                         "next_lr": self.optimizer.param_groups[0]["lr"]})
            fieldnames = list(dict.fromkeys(key for row in rows for key in row))
            with log_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)
            with (self.run_dir / "epoch_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)
            save_checkpoint(
                self.run_dir / "checkpoint_last.pth", self.model, self.optimizer, self.scheduler,
                epoch, self.config["seed"], self.config, self.config["data"]["train_manifest"],
                recovery_lineage=self.recovery_lineage,
            )
            monitoring = " ".join(f"{key}={value:.6f}" for key, value in self.last_epoch_metrics.items() if key != "loss")
            print(f"Finished epoch {epoch}/{epochs}: loss={loss:.6f} {monitoring} lr={learning_rate:.6g} elapsed={time.perf_counter() - started:.1f}s; checkpoint saved", flush=True)


@torch.no_grad()
def evaluate_sample(model, image, target, protocol, seed=1234, radius=5, threshold=0.5,
                    metric_epsilon=1e-7, boundary_width=1):
    if protocol not in {"0-m", "1-m", "2-m"}:
        raise ValueError(f"Unknown evaluation protocol: {protocol}")
    if image.shape[0] != 1 or target.shape[0] != 1:
        raise ValueError("evaluate_sample requires exactly one image")
    if threshold != 0.5:
        raise ValueError("Formal evaluation threshold must be 0.5")
    rng = np.random.default_rng(seed); gt = target[0, 0].cpu().numpy().astype(bool)
    clicks = []
    if protocol in {"1-m", "2-m"}: clicks.append(sample_first_click(gt, rng))
    dense = None if not clicks else render_dense_click_map([clicks], radius=radius, device=image.device)
    output = model(image, dense)["logits"]
    prediction = torch.sigmoid(output.float()).ge(threshold)[0, 0].cpu().numpy()
    if protocol == "2-m":
        second = sample_second_click(gt, prediction, rng)
        if second is not None:
            clicks.append(second)
            dense = render_dense_click_map([clicks], radius=radius, device=image.device)
            output = model(image, dense)["logits"]
            prediction = torch.sigmoid(output.float()).ge(threshold)[0, 0].cpu().numpy()
    return compute_metrics(prediction, gt, eps=metric_epsilon, boundary_width=boundary_width), clicks, output
