"""Single-GPU training, checkpoint recovery and evaluation."""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import Trainer, TrainingArguments, VideoMAEImageProcessor, EarlyStoppingCallback, set_seed
from prepare_data import write_json
from .models import VideoFolderDataset, VideoDataCollator, MeanPoolVideoMAEClassifier, EventLevelVideoUnderstandingModel

BACKBONE_NAME = "MCG-NJU/videomae-small-finetuned-kinetics"

# =========================
# 10) Metrics: Top@1 và Top@5
# =========================
def topk_predictions(logits, k=5):
    """Return top-k predicted class indices for each sample."""
    k = min(k, logits.shape[1])
    return np.argsort(logits, axis=1)[:, -k:][:, ::-1]


def topk_accuracy(logits, labels, k=5):
    """Top-k accuracy: correct if the true label appears in top-k predictions."""
    topk = topk_predictions(logits, k)
    return np.mean([label in topk_i for label, topk_i in zip(labels, topk)])


def topk_macro_f1(logits, labels, k=5):
    """
    Top-k macro F1 for single-label multi-class classification.
    """
    num_classes = logits.shape[1]
    k = min(k, num_classes)
    topk = topk_predictions(logits, k)

    y_true = np.zeros((len(labels), num_classes), dtype=int)
    y_true[np.arange(len(labels)), labels] = 1

    y_pred = np.zeros((len(labels), num_classes), dtype=int)
    for i, pred_indices in enumerate(topk):
        y_pred[i, pred_indices] = 1

    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # Top@1
    top1_preds = np.argmax(logits, axis=-1)
    top1_acc = accuracy_score(labels, top1_preds)
    top1_macro_f1 = f1_score(labels, top1_preds, average="macro", zero_division=0)

    # Top@5
    top5_acc = topk_accuracy(logits, labels, k=5)

    return {
        "top1_macro_f1": top1_macro_f1,
        "top1_acc": top1_acc,
        "top5_acc": top5_acc,
    }

import re
from transformers import TrainerCallback

WARMUP_KWARGS = {"warmup_ratio": 0.1}

def build_training_args(run_name, output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        remove_unused_columns=False,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        **WARMUP_KWARGS,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        label_smoothing_factor=0.0,  # Model applies smoothing exactly once.

        metric_for_best_model="top1_macro_f1",
        greater_is_better=True,

        save_total_limit=2,
        fp16=(PRECISION == "fp16"),
        bf16=(PRECISION == "bf16"),
        seed=SEED,
        data_seed=SEED,
        restore_callback_states_from_checkpoint=True,
        dataloader_drop_last=False,
        report_to="none",

        dataloader_num_workers=NUM_WORKERS,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=(NUM_WORKERS > 0),
        dataloader_prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )


class EpochProgressCallback(TrainerCallback):
    """Persist epoch metrics and mark a successfully completed training run."""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.metrics_csv = os.path.join(output_dir, "epoch_metrics.csv")
        self.done_file = os.path.join(output_dir, "TRAINING_DONE.json")

    def _dump_log_history(self, state):
        os.makedirs(self.output_dir, exist_ok=True)
        pd.DataFrame(state.log_history).to_csv(self.metrics_csv, index=False)

    def on_evaluate(self, args, state, control, **kwargs):
        self._dump_log_history(state)

    def on_train_end(self, args, state, control, **kwargs):
        self._dump_log_history(state)
        with open(self.done_file + ".tmp", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_model_checkpoint": state.best_model_checkpoint,
                    "global_step": state.global_step,
                    "epoch": state.epoch,
                },
                f,
            )
        os.replace(self.done_file + ".tmp", self.done_file)


def find_last_valid_checkpoint(output_dir):
    """Checkpoint-<step> mới nhất có đủ file để resume. Bỏ qua checkpoint ghi dở (ví dụ mất kết nối giữa chừng)."""
    if not os.path.isdir(output_dir):
        return None
    candidates = []
    for name in os.listdir(output_dir):
        m = re.fullmatch(r"checkpoint-(\d+)", name)
        path = os.path.join(output_dir, name)
        if m and os.path.isdir(path):
            candidates.append((int(m.group(1)), path))
    for _, path in sorted(candidates, reverse=True):
        has_weights = any(os.path.exists(os.path.join(path, w)) for w in ("model.safetensors", "pytorch_model.bin"))
        has_state = all(os.path.exists(os.path.join(path, f)) for f in ("optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth"))
        if has_weights and has_state:
            return path
        print("Bỏ qua checkpoint chưa đầy đủ:", path)
    return None


def load_weights_from_checkpoint(model, ckpt_dir):
    st_path = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.exists(st_path):
        from safetensors.torch import load_file
        state_dict = load_file(st_path)
    else:
        state_dict = torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"), map_location="cpu")
    model.load_state_dict(state_dict)


def train_model(model, run_name):
    output_dir = os.path.join(MODEL_OUTPUT_ROOT, run_name)
    args = build_training_args(run_name, output_dir)
    progress_cb = EpochProgressCallback(output_dir)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=VideoDataCollator(),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE),
            progress_cb,
        ],
    )

    trainer.label_names = ["labels"]
    if os.path.exists(progress_cb.done_file):
        with open(progress_cb.done_file, "r", encoding="utf-8") as f:
            done_info = json.load(f)
        if not done_info.get("best_model_checkpoint"):
            raise RuntimeError(f"{progress_cb.done_file} không có best_model_checkpoint. Xoá file này để train lại/resume.")
        best_ckpt = os.path.join(output_dir, os.path.basename(done_info["best_model_checkpoint"]))
        print(f"[{run_name}] Đã train xong trước đó ({done_info['epoch']:.0f} epoch). Nạp best checkpoint: {best_ckpt}")
        load_weights_from_checkpoint(trainer.model, best_ckpt)
    else:
        resume_ckpt = find_last_valid_checkpoint(output_dir)
        if resume_ckpt:
            print(f"[{run_name}] Tìm thấy checkpoint, TIẾP TỤC train từ: {resume_ckpt}")
        else:
            print(f"[{run_name}] Chưa có checkpoint, train từ đầu.")
        trainer.train(resume_from_checkpoint=resume_ckpt)

    val_metrics = trainer.evaluate(eval_dataset=val_dataset)

    print(f"{run_name} validation metric keys:", val_metrics.keys())

    return trainer, val_metrics

def check_gpu(precision):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable. Install a CUDA-enabled PyTorch build and check nvidia-smi.')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise RuntimeError('Use python train.py with one visible GPU; distributed training is not supported.')
    if precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('This GPU does not support bf16; use --precision fp16 or fp32.')
    print(f'GPU: {torch.cuda.get_device_name(0)}; torch={torch.__version__}', flush=True)


def run(options, prepared):
    # Globals are used by the training helpers ported from the notebook.
    global TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS
    global LEARNING_RATE, NUM_EPOCHS, WEIGHT_DECAY, NUM_WORKERS, PRECISION, SEED
    global MODEL_OUTPUT_ROOT, EARLY_STOPPING_PATIENCE, train_dataset, val_dataset
    work = options.work_dir
    config_path = work / 'training_config.json'
    previous = json.loads(config_path.read_text()) if config_path.exists() else None
    gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    batch = options.batch_size or (previous['batch_size'] if previous else (8 if gpu_gb >= 35 else 4 if gpu_gb >= 20 else 2))
    eval_batch = options.eval_batch_size or (previous['eval_batch_size'] if previous else batch)
    config = {
        'backbone': BACKBONE_NAME, 'batch_size': batch, 'eval_batch_size': eval_batch,
        'gradient_accumulation_steps': 48 // batch, 'epochs': options.epochs,
        'learning_rate': options.learning_rate, 'seed': options.seed,
        'precision': options.precision, 'patience': options.patience,
        'num_frames': 16, 'num_clips': 4, 'dropout': .25, 'weight_decay': .05,
        'label_smoothing': .05, 'freeze_backbone': False,
        'split_signature': json.loads((prepared / 'READY.json').read_text())['signature'],
    }
    if previous and previous != config:
        diff = [k for k in set(previous) | set(config) if previous.get(k) != config.get(k)]
        raise ValueError(f'Resume configuration changed: {diff}. Restore original flags or use a new --work-dir.')
    write_json(config_path, config)
    TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE = batch, eval_batch
    GRADIENT_ACCUMULATION_STEPS = 48 // batch
    LEARNING_RATE, NUM_EPOCHS, WEIGHT_DECAY = options.learning_rate, options.epochs, .05
    NUM_WORKERS, PRECISION, SEED = options.num_workers, options.precision, options.seed
    EARLY_STOPPING_PATIENCE = options.patience
    MODEL_OUTPUT_ROOT = str(work / 'checkpoints')
    final_root = work / 'final_outputs'
    final_root.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    mapping = json.loads((prepared / 'label_map.json').read_text())
    label2id = mapping['label2id']
    classes = sorted(label2id, key=label2id.get)
    processor = VideoMAEImageProcessor.from_pretrained(BACKBONE_NAME)
    datasets = {
        split: VideoFolderDataset(prepared, split, processor, label2id, num_frames=16,
                                  num_clips=4, train=(split == 'train'))
        for split in ('train', 'val', 'test')
    }
    train_dataset, val_dataset = datasets['train'], datasets['val']
    print(f'Batch {batch} x accumulation {GRADIENT_ACCUMULATION_STEPS} = 48', flush=True)
    selected = ['baseline', 'proposed'] if options.model == 'both' else [options.model]
    for name in selected:
        set_seed(SEED)
        cls = MeanPoolVideoMAEClassifier if name == 'baseline' else EventLevelVideoUnderstandingModel
        model = cls(backbone_name=BACKBONE_NAME, num_labels=len(classes), dropout=.25,
                    freeze_backbone=False, label_smoothing=.05)
        # Store task labels rather than Kinetics labels in exported config.
        model.config.id2label = {i: c for c, i in label2id.items()}
        model.config.label2id = label2id
        run_name = f'{name}_videomae_small_v8_diving48'
        trainer, val_metrics = train_model(model, run_name)
        dest = final_root / name
        dest.mkdir(parents=True, exist_ok=True)
        prediction = trainer.predict(datasets['test'])
        logits, labels = prediction.predictions, prediction.label_ids
        preds = logits.argmax(axis=-1)
        np.savez_compressed(dest / 'test_predictions.npz', logits=logits, labels=labels)
        # Preserve prediction ordering for tracing results back to clips.
        top5 = topk_predictions(logits)
        pd.DataFrame({'video_path': [s[0] for s in datasets['test'].samples],
                      'label': labels, 'prediction': preds,
                      'top5_classes': [json.dumps([classes[i] for i in row]) for row in top5],
                      'hit_top5': [bool(y in row) for y, row in zip(labels, top5)]
                      }).to_csv(dest / 'test_samples.csv', index=False)
        report = classification_report(labels, preds, labels=np.arange(len(classes)),
                                       target_names=classes, zero_division=0, output_dict=True)
        pd.DataFrame(report).transpose().to_csv(dest / 'classification_report.csv')
        matrix = confusion_matrix(labels, preds, labels=np.arange(len(classes)))
        pd.DataFrame(matrix, index=classes, columns=classes).to_csv(dest / 'confusion_matrix.csv')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(18, 16))
        ax.imshow(matrix)
        ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=5)
        ax.set_yticks(range(len(classes)), classes, fontsize=5)
        ax.set(xlabel='Predicted label', ylabel='True label', title=f'{name}: test confusion matrix')
        fig.tight_layout()
        fig.savefig(dest / 'confusion_matrix.png', dpi=150)
        plt.close(fig)
        results = {'model': name, 'validation': val_metrics,
                   'test': compute_metrics((logits, labels))}
        write_json(dest / 'metrics.json', results)
        trainer.save_model(str(dest / 'model'))
        processor.save_pretrained(dest / 'model')
        model.config.save_pretrained(dest / 'model')
        write_json(dest / 'model' / 'label_map.json', mapping)
        write_json(dest / 'model' / 'training_config.json', {**config, 'model': name})
        print(json.dumps(results, indent=2), flush=True)
        # Finish evaluation/export before allocating the second model on the GPU.
        del trainer, model, prediction
        gc.collect()
        torch.cuda.empty_cache()
    rows = []
    for name in ('baseline', 'proposed'):
        metric_path = final_root / name / 'metrics.json'
        if metric_path.exists():
            metric = json.loads(metric_path.read_text())
            rows.append({'model': name,
                         **{f'val_{k}': v for k, v in metric['validation'].items()},
                         **{f'test_{k}': v for k, v in metric['test'].items()}})
    pd.DataFrame(rows).to_csv(final_root / 'model_comparison.csv', index=False)
