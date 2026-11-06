"""VideoMAE architectures and temporal sampling ported from notebook v8."""
import random
from pathlib import Path
import av
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import VideoMAEModel
from transformers.modeling_outputs import ImageClassifierOutput
from prepare_data import VIDEO_EXTS

def is_video_file(path):
    return path.suffix.lower() in VIDEO_EXTS

# =========================
# 8) Dataset + collator
# =========================
class VideoFolderDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split,
        processor,
        label2id,
        num_frames=16,
        num_clips=4,
        train=True,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.processor = processor
        self.label2id = label2id
        self.num_frames = num_frames
        self.num_clips = num_clips
        self.train = train
        self.samples = self._gather_video_paths()

    def _gather_video_paths(self):
        samples = []
        for class_name, class_id in self.label2id.items():
            class_dir = self.root_dir / self.split / class_name
            if not class_dir.exists():
                continue
            for path in class_dir.rglob("*"):
                if path.is_file() and is_video_file(path):
                    samples.append((str(path), class_id))
        if not samples:
            raise ValueError(f"No videos found in {self.root_dir / self.split}")
        return sorted(samples)

    def __len__(self):
        return len(self.samples)

    def _read_video_frames(self, video_path):
        # Fail loudly instead of silently training on black frames.
        try:
            with av.open(video_path) as container:
                frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
            if not frames:
                raise ValueError("Empty video")
            return np.stack(frames)
        except Exception as exc:
            raise RuntimeError(f"Cannot decode {video_path}: {exc}") from exc

    def _sample_clip_indices(self, total_frames):
        if total_frames < self.num_frames:
            base = np.linspace(0, total_frames - 1, self.num_frames)
            base = np.clip(np.round(base).astype(int), 0, total_frames - 1)
            return [base for _ in range(self.num_clips)]

        segment_boundaries = np.linspace(
            0, total_frames - self.num_frames, self.num_clips + 1
        ).astype(int)

        clips = []
        for i in range(self.num_clips):
            start_min = segment_boundaries[i]
            start_max = segment_boundaries[i + 1]
            if start_max < start_min:
                start_max = start_min

            if self.train:
                start = random.randint(start_min, start_max) if start_max > start_min else start_min
            else:
                start = (start_min + start_max) // 2

            indices = np.arange(start, start + self.num_frames)
            indices = np.clip(indices, 0, total_frames - 1)
            clips.append(indices)

        return clips

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self._read_video_frames(video_path)
        total_frames = frames.shape[0]
        clip_indices_list = self._sample_clip_indices(total_frames)

        clips_pixel_values = []
        for clip_indices in clip_indices_list:
            clip_frames = [frames[i] for i in clip_indices]
            encoded = self.processor(clip_frames, return_tensors="pt")
            pixel_values = encoded["pixel_values"].squeeze(0)
            clips_pixel_values.append(pixel_values)

        pixel_values = torch.stack(clips_pixel_values, dim=0)

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label, dtype=torch.long),
            "video_path": video_path,
        }

class VideoDataCollator:
    def __call__(self, features):
        pixel_values = torch.stack([f["pixel_values"] for f in features], dim=0)
        labels = torch.stack([f["labels"] for f in features], dim=0)
        return {"pixel_values": pixel_values, "labels": labels}


# =========================
# 9) Models
# Improvements:
# - label smoothing is applied inside the model loss
# - Proposed model adds: residual + LayerNorm + clip positional embedding + 1-layer TransformerEncoder
# =========================
class MeanPoolVideoMAEClassifier(nn.Module):
    # Trainer chỉ chia loss cho gradient_accumulation_steps khi accepts_loss_kwargs=False.
    # (forward có **kwargs nên mặc định Trainer coi model tự chuẩn hoá loss -> gradient bị nhân GRADIENT_ACCUMULATION_STEPS lần)
    accepts_loss_kwargs = False

    def __init__(self, backbone_name, num_labels, dropout=0.25, freeze_backbone=False, label_smoothing=0.0):
        super().__init__()
        self.backbone = VideoMAEModel.from_pretrained(backbone_name)
        self.config = self.backbone.config
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _clip_encode(self, clip_pixel_values):
        outputs = self.backbone(pixel_values=clip_pixel_values)
        last_hidden_state = outputs.last_hidden_state
        clip_embedding = last_hidden_state.mean(dim=1)
        return clip_embedding

    def forward(self, pixel_values, labels=None, **kwargs):
        batch_size, num_clips = pixel_values.shape[:2]
        clip_embeddings = []

        for i in range(num_clips):
            clip_x = pixel_values[:, i]
            clip_emb = self._clip_encode(clip_x)
            clip_embeddings.append(clip_emb)

        clip_embeddings = torch.stack(clip_embeddings, dim=1)   # [B, C, D]
        video_repr = clip_embeddings.mean(dim=1)                # [B, D]
        logits = self.classifier(self.dropout(video_repr))

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return ImageClassifierOutput(loss=loss, logits=logits)


class EventAttentionPool(nn.Module):
    def __init__(self, hidden_size, dropout=0.25):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        # x: [B, C, D]
        weights = torch.softmax(self.score(x).squeeze(-1), dim=1)  # [B, C]
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)       # [B, D]
        return pooled, weights


class EventLevelVideoUnderstandingModel(nn.Module):
    # Trainer chỉ chia loss cho gradient_accumulation_steps khi accepts_loss_kwargs=False.
    # (forward có **kwargs nên mặc định Trainer coi model tự chuẩn hoá loss -> gradient bị nhân GRADIENT_ACCUMULATION_STEPS lần)
    accepts_loss_kwargs = False

    def __init__(
        self,
        backbone_name,
        num_labels,
        dropout=0.25,
        freeze_backbone=False,
        label_smoothing=0.05,
        max_num_clips=8,
        event_encoder_layers=1,
        event_encoder_heads=4,
    ):
        super().__init__()
        self.backbone = VideoMAEModel.from_pretrained(backbone_name)
        self.config = self.backbone.config
        hidden_size = self.backbone.config.hidden_size

        self.event_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

        # Residual normalization after event projection
        self.event_norm = nn.LayerNorm(hidden_size)

        # Learnable temporal/clip positional embedding
        self.clip_pos_embed = nn.Parameter(torch.zeros(1, max_num_clips, hidden_size))
        nn.init.trunc_normal_(self.clip_pos_embed, std=0.02)

        # Event-level temporal reasoning over clip/event tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=event_encoder_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.event_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=event_encoder_layers,
        )

        self.event_pool = EventAttentionPool(hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _clip_encode(self, clip_pixel_values):
        outputs = self.backbone(pixel_values=clip_pixel_values)
        last_hidden_state = outputs.last_hidden_state
        clip_embedding = last_hidden_state.mean(dim=1)
        return clip_embedding

    def forward(self, pixel_values, labels=None, **kwargs):
        batch_size, num_clips = pixel_values.shape[:2]
        clip_embeddings = []

        for i in range(num_clips):
            clip_x = pixel_values[:, i]
            clip_emb = self._clip_encode(clip_x)
            clip_embeddings.append(clip_emb)

        clip_embeddings = torch.stack(clip_embeddings, dim=1)  # [B, C, D]

        # Event token construction: residual projection + normalization
        event_tokens = self.event_norm(
            clip_embeddings + self.event_projection(clip_embeddings)
        )

        # Add temporal order information
        if num_clips > self.clip_pos_embed.shape[1]:
            raise ValueError(
                f"num_clips={num_clips} exceeds max_num_clips={self.clip_pos_embed.shape[1]}. "
                "Increase max_num_clips in EventLevelVideoUnderstandingModel."
            )
        event_tokens = event_tokens + self.clip_pos_embed[:, :num_clips, :]

        # Event-level reasoning
        event_tokens = self.event_encoder(event_tokens)

        # Attention-based event aggregation
        video_repr, event_weights = self.event_pool(event_tokens)
        logits = self.classifier(self.dropout(video_repr))

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        output = ImageClassifierOutput(loss=loss, logits=logits)
        return output
