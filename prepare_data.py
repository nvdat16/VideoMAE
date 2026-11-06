"""Prepare Diving48 locally without importing PyTorch or copying video data."""
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tarfile
import tempfile

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.mpeg', '.mpg'}


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def stratified_split_files(files, val_ratio=.15, test_ratio=.15, seed=42):
    # Same ordering, rounding and rare-class policy as notebook v8.
    files = sorted(files, key=lambda p: p.name if isinstance(p, Path) else p)
    random.Random(seed).shuffle(files)
    n = len(files)
    if n < 2:
        return files, [], []
    if n == 2:
        return files[:1], [], files[1:]
    n_val, n_test = max(1, round(n * val_ratio)), max(1, round(n * test_ratio))
    while n - n_val - n_test < 1:
        if n_val >= n_test and n_val > 1:
            n_val -= 1
        else:
            n_test -= 1
    return files[n_val+n_test:], files[:n_val], files[n_val:n_val+n_test]


def extract_videos(archive, destination):
    """Atomic extraction; reject traversal, links and special files."""
    destination = Path(destination)
    marker = destination / '.extracted.json'
    stat = archive.stat()
    identity = {'archive': str(archive.resolve()), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    if marker.exists() and json.loads(marker.read_text()) == identity:
        return destination
    if destination.exists():
        raise RuntimeError(f'Incomplete or different extraction at {destination}; use a new --work-dir.')
    staging = Path(tempfile.mkdtemp(prefix='extract-', dir=destination.parent))
    try:
        with tarfile.open(archive, 'r|gz') as tar:
            for member in tar:
                target = (staging / member.name).resolve()
                if not target.is_relative_to(staging.resolve()) or not (member.isdir() or member.isfile()):
                    raise ValueError(f'Unsafe archive member: {member.name}')
                if member.isfile() and target.suffix.lower() in VIDEO_EXTS:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as src, target.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
        write_json(staging / '.extracted.json', identity)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination


def prepare_dataset(data_dir, work_dir, videos_dir=None, seed=42):
    data_dir, work_dir = Path(data_dir).resolve(), Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    names = ['Diving48_vocab.json', 'Diving48_V2_train.json', 'Diving48_V2_test.json']
    for name in names:
        if not (data_dir / name).is_file():
            raise FileNotFoundError(f'Missing {data_dir / name}')
    vocab, train, test = [json.loads((data_dir / n).read_text()) for n in names]
    classes = ['_'.join(row) for row in vocab]
    if len(set(classes)) != len(classes) or any('/' in c or '\\' in c or c in {'', '.', '..'} for c in classes):
        raise ValueError('Invalid or duplicate vocabulary class names')
    labels = {}
    for entry in train + test:
        vid, label = entry['vid_name'], entry['label']
        if not isinstance(label, int) or not 0 <= label < len(classes):
            raise ValueError(f'Invalid label: {entry}')
        if vid in labels and labels[vid] != label:
            raise ValueError(f'Conflicting labels for {vid}')
        labels[vid] = label
    if videos_dir:
        video_root = Path(videos_dir).resolve()
        if not video_root.is_dir():
            raise FileNotFoundError(video_root)
    else:
        archive = data_dir / 'Diving48_rgb.tar.gz'
        if not archive.is_file():
            raise FileNotFoundError(f'Missing {archive}; or pass --videos-dir for extracted videos')
        print(f'Preparing videos from {archive}', flush=True)
        video_root = extract_videos(archive, work_dir / 'videos')
    by_id = {}
    for path in sorted(video_root.rglob('*')):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS and path.stem in labels:
            if path.stem in by_id:
                raise ValueError(f'Duplicate video ID: {path.stem}')
            by_id[path.stem] = path
    missing = sorted(set(labels) - by_id.keys())
    if missing:
        raise FileNotFoundError(f'{len(missing)} annotated videos missing; examples: {missing[:5]}')
    signature = hashlib.sha256(json.dumps({
        'seed': seed, 'ratios': [.70, .15, .15], 'vocab': vocab, 'labels': sorted(labels.items()),
        'videos': [(k, str(p), p.stat().st_size) for k, p in sorted(by_id.items())],
    }, sort_keys=True).encode()).hexdigest()
    prepared = work_dir / 'prepared_70_15_15'
    ready = prepared / 'READY.json'
    if prepared.exists():
        if not ready.exists() or json.loads(ready.read_text())['signature'] != signature:
            raise RuntimeError('Prepared split differs from current input; use a new --work-dir.')
        with (prepared / 'annotations.csv').open() as f:
            rows = list(csv.DictReader(f))
        if len(rows) != len(labels) or any(not (prepared / r['video_path']).is_file() for r in rows):
            raise RuntimeError('Prepared split is incomplete; use a new --work-dir.')
        print(f'Reusing split: {prepared}', flush=True)
        return prepared
    staging = Path(tempfile.mkdtemp(prefix='split-', dir=work_dir))
    rows = []
    try:
        for cls in sorted(classes):
            paths = [by_id[k] for k, label in labels.items() if classes[label] == cls]
            # Notebook labels are sorted class names, not original vocabulary indices.
            label_id = sorted(classes).index(cls)
            for split, files in zip(('train', 'val', 'test'), stratified_split_files(paths, seed=seed)):
                target = staging / split / cls
                target.mkdir(parents=True, exist_ok=True)
                for src in files:
                    dest = target / src.name
                    os.symlink(src, dest)
                    rows.append({'split': split, 'video_path': str(dest.relative_to(staging)),
                                 'label': cls, 'label_id': label_id})
        with (staging / 'annotations.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['split', 'video_path', 'label', 'label_id'])
            writer.writeheader()
            writer.writerows(rows)
        label2id = {cls: i for i, cls in enumerate(sorted(classes))}
        write_json(staging / 'label_map.json', {'label2id': label2id, 'id2label': {i: c for c, i in label2id.items()}})
        counts = {s: sum(r['split'] == s for r in rows) for s in ('train', 'val', 'test')}
        if not all(counts.values()):
            raise ValueError(f'Empty split: {counts}; dataset too small')
        write_json(staging / 'READY.json', {'signature': signature, 'counts': counts, 'seed': seed})
        staging.rename(prepared)
        print(f'Prepared {counts} at {prepared}', flush=True)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return prepared
