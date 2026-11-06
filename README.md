# VideoMAE-Small / Diving48 — train trên server GPU

Chuyển từ `Diving48_VideoMAE_Small_v8_share.ipynb` sang script Python độc lập,
không cần Colab hay Google Drive. Chạy baseline mean pooling và proposed event-level
Transformer lần lượt, export kết quả rồi giải phóng GPU trước khi nạp model tiếp theo.

Giữ pipeline v8: gộp annotation train/test V2, chia theo lớp 70/15/15, seed 42;
16 frame × 4 clip/video; effective batch 48; LR 2e-5; weight decay 0.05;
dropout 0.25; label smoothing 0.05; tối đa 30 epoch; early stopping patience 3,
chọn best checkpoint theo validation macro F1. Label smoothing thực hiện một lần
trong loss của model. Giữ `accepts_loss_kwargs=False` để gradient accumulation đúng.

**Lưu ý về kết quả:** split v8 theo clip có thể đưa các clip cùng video gốc vào
train/val/test. Đây là cấu hình tái lập notebook v8, có nguy cơ rò rỉ dữ liệu;
không so sánh trực tiếp với kết quả trên split chính thức Diving48 V2.

## 1. Cài đặt (Linux, Python 3.10–3.12, NVIDIA GPU)

```bash
git clone https://github.com/nvdat16/VideoMAE.git
cd VideoMAE
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
nvidia-smi
# Ví dụ CUDA 12.4: chọn wheel CUDA phù hợp driver của server.
python -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### NVIDIA GB10 / DGX Spark (Linux ARM64)

GB10 cần bản PyTorch hỗ trợ Blackwell. Không dùng lệnh cài torch 2.6 / cu124 ở trên.
Với Python 3.10–3.12, cài wheel CUDA 13.0 chính thức trước khi cài requirements:

```bash
python -m pip install --upgrade pip
python -m pip install --upgrade 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip check
python - <<'PYCODE'
import platform
import torch
print("Architecture:", platform.machine())
print("Torch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA unavailable"
print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))
x = torch.randn(256, 256, device="cuda", requires_grad=True)
(x @ x).square().mean().backward()
torch.cuda.synchronize()
print("CUDA forward/backward OK")
PYCODE
```

`nvidia-smi` hiển thị mức CUDA driver hỗ trợ; không xác nhận phiên bản runtime
mà PyTorch đang dùng. Kiểm tra bằng `torch.version.cuda` và phép tính CUDA ở trên.
Bản torch 2.9.1 cu130 có [wheel Linux aarch64 chính thức](https://download.pytorch.org/whl/cu130/torch/).
Chưa kiểm thử train CUDA trên GB10 trong môi trường phát triển này.

Bắt đầu với batch nhỏ, tăng sau khi xác nhận mức dùng bộ nhớ thực tế:

```bash
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --data-dir /path/to/Dataset --work-dir ./runs/diving48_gb10 \
  --batch-size 2 --eval-batch-size 2 --precision bf16
```

Lần đầu cần mạng để tải checkpoint/processor công khai
`MCG-NJU/videomae-small-finetuned-kinetics` vào Hugging Face cache.
Dùng `HF_HOME=/path/to/cache` nếu muốn chọn nơi lưu cache.

## 2. Dataset đúng cấu trúc trong ảnh

```text
/path/to/Dataset/
├── .cache/                    # bỏ qua
├── Blur/                      # bỏ qua
├── england_epl/               # bỏ qua
├── Original1206_COCO/         # bỏ qua
├── Original1206-YOLO/         # bỏ qua
├── UAV-YOLO/                  # bỏ qua
├── Diving48_rgb.tar.gz
├── Diving48_V2_train.json
├── Diving48_V2_test.json
└── Diving48_vocab.json        # cần có thêm file này nếu chưa có
```

Truyền đường dẫn tới **Dataset**, không phải tới thư mục cha. Chỉ đọc ba file JSON
và archive Diving48, không quét các dataset khác. JSON annotation phải là danh sách
record có `vid_name` và `label`; vocabulary là danh sách các danh sách từ mô tả lớp,
đúng định dạng notebook gốc.

Mặc định archive được giải nén một lần vào `<work-dir>/videos/`; cần khoảng 10–12 GB
cho video, cộng dung lượng checkpoint và kết quả. Nên dành ít nhất 25 GB trống.
Script tạo symlink để chia tập, không nhân ba dữ liệu video. Dataset gốc chỉ cần quyền đọc;
`work-dir` phải ghi được và filesystem hỗ trợ symlink. Không xóa hoặc di chuyển video
đã giải nén trong lúc sử dụng split.

## 3. Train

```bash
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --data-dir /path/to/Dataset \
  --work-dir ./runs/diving48_v8_small
```

Mặc định chạy cả hai model trên **một GPU**. Chọn GPU bằng `CUDA_VISIBLE_DEVICES`,
không chạy bằng `torchrun`. Batch tự chọn 8/4/2 theo bộ nhớ GPU, accumulation tương ứng
6/12/24 để giữ effective batch = 48. Đây là heuristic; nếu OOM, dùng batch nhỏ hơn
với **work-dir mới**, ví dụ `--batch-size 1`. Có thể giảm `--eval-batch-size 1`.

```bash
# Chỉ proposed, batch 2 x accumulation 24, 8 workers
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --data-dir /path/to/Dataset --work-dir ./runs/proposed_gpu0 \
  --model proposed --batch-size 2 --num-workers 8

# Chuẩn bị dữ liệu trước, không cần CUDA hay thư viện ML
python train.py --data-dir /path/to/Dataset \
  --work-dir ./runs/diving48_v8_small --prepare-only

# Đã giải nén: không cần file tar.gz
python train.py --data-dir /path/to/Dataset \
  --videos-dir /path/to/Diving48/rgb \
  --work-dir ./runs/diving48_extracted

# Chạy nền, giữ log sau khi ngắt SSH
nohup env CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --data-dir /path/to/Dataset --work-dir ./runs/diving48_v8_small \
  > train.log 2>&1 &
tail -f train.log
```

Xem `python train.py --help` cho các tùy chọn. Mặc định fp16; GPU hỗ trợ có thể dùng
`--precision bf16`, hoặc `--precision fp32`. Giảm `--num-workers 0` khi RAM hạn chế
hoặc cần xem lỗi decode trực tiếp. Video lỗi sẽ dừng với tên file thay vì thay bằng
frame đen như notebook.

## 4. Resume và kết quả

Chạy lại **cùng lệnh, cùng work-dir** để resume checkpoint đầy đủ mới nhất. Lưu mỗi epoch,
giữ tối đa hai checkpoint (best + latest). Mất phần epoch đang dở nếu tiến trình bị ngắt.
Khôi phục optimizer, scheduler, RNG và callback early stopping; run hoàn tất sẽ nạp best
checkpoint rồi đánh giá/export lại, không train lại.

`training_config.json` khóa cấu hình train và fingerprint split; script từ chối resume
nếu đổi cấu hình. Batch tự chọn lần đầu được giữ lại. Nếu đã dùng các flag khác mặc định
(epochs, precision, learning rate, seed, patience), giữ các flag đó khi chạy lại.
Muốn thí nghiệm mới, dùng `--work-dir` mới. Chỉ chạy **một tiến trình mỗi work-dir**.

```text
runs/diving48_v8_small/
├── videos/                          # giải nén một lần
├── prepared_70_15_15/                # symlink train/val/test, annotations.csv, label_map.json
├── training_config.json
├── checkpoints/<run>/
│   ├── checkpoint-<step>/
│   ├── epoch_metrics.csv
│   └── TRAINING_DONE.json
└── final_outputs/
    ├── model_comparison.csv
    ├── baseline/
    └── proposed/
        ├── metrics.json
        ├── classification_report.csv
        ├── confusion_matrix.csv
        ├── confusion_matrix.png
        ├── test_predictions.npz       # logits và labels
        ├── test_samples.csv           # đường dẫn video theo đúng thứ tự dự đoán
        └── model/                    # weights, processor, config, label map, tham số train
```

Hai model là `torch.nn.Module` tùy chỉnh, không nạp bằng `AutoModelForVideoClassification`.
Để nạp weights đã export, khởi tạo lớp tương ứng trong `videomae_local.models` với
`num_labels` lấy từ label map, rồi `load_state_dict(safetensors.torch.load_file(...))`.
Cần cùng backbone và cấu hình model; model proposed dùng `max_num_clips=8`.

## Kiểm tra cục bộ

```bash
python -m unittest discover -s tests -v
python -m compileall -q train.py prepare_data.py videomae_local tests
```

Các test dữ liệu kiểm tra giải nén an toàn, split tất định, cache và phát hiện thiếu video.
Nếu đã cài requirements, smoke test CPU dùng VideoMAE nhỏ khởi tạo ngẫu nhiên để thử cả
hai kiến trúc, Trainer, lưu/nạp weights và resume sau một epoch, không tải model từ mạng.
Các test không xác nhận chất lượng model; train thực tế cần dataset và GPU.
Notebook gốc được giữ trong repo để đối chiếu.
