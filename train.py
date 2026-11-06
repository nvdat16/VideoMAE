#!/usr/bin/env python3
"""Local GPU entry point; --prepare-only and --help work without ML dependencies."""
import argparse
from pathlib import Path
from prepare_data import prepare_dataset


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path('../Dataset'), help='Folder shown in the dataset screenshot')
    p.add_argument('--work-dir', type=Path, default=Path('./runs/diving48_v8_small'))
    p.add_argument('--videos-dir', type=Path, help='Already extracted Diving48 videos; skips tar extraction')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--model', choices=['both', 'baseline', 'proposed'], default='both')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, help='Divisor of 48; otherwise choose by GPU memory')
    p.add_argument('--eval-batch-size', type=int)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--learning-rate', type=float, default=2e-5)
    p.add_argument('--precision', choices=['fp16', 'bf16', 'fp32'], default='fp16')
    p.add_argument('--patience', type=int, default=3)
    args = p.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.num_workers < 0 or args.learning_rate <= 0:
        p.error('epochs/patience/lr must be positive; num-workers must be nonnegative')
    if args.batch_size is not None and (args.batch_size < 1 or 48 % args.batch_size):
        p.error('--batch-size must be a positive divisor of 48')
    if args.eval_batch_size is not None and args.eval_batch_size < 1:
        p.error('--eval-batch-size must be positive')
    args.work_dir = args.work_dir.resolve()
    return args


def main():
    args = parse_args()
    if not args.prepare_only:
        from videomae_local.training import check_gpu, run
        check_gpu(args.precision)
    prepared = prepare_dataset(args.data_dir, args.work_dir, args.videos_dir, args.seed)
    if not args.prepare_only:
        run(args, prepared)


if __name__ == '__main__':
    main()
