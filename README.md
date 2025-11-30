# VideoTrans - 動画文字起こしツール

ローカルの動画ファイルを読み込んで文字起こしを行い、Ollamaを使用して概要を生成するPythonスクリプトです。

## 必要な環境

- Python 3.8以上
- FFmpeg（Whisperが音声を抽出するために必要）
- Ollama（要約生成に使用、デフォルト: gemma3:1b）
- NVIDIA GPU（CUDA版を使用する場合）
- CUDA対応のNVIDIAドライバー（CUDA版を使用する場合）

## インストール

### CPU版の場合

1. 必要なパッケージをインストール:
```bash
pip install -r requirements.txt
```

### CUDA版（GPU加速）の場合

1. CUDA対応のPyTorchをインストール:
```bash
# CUDA 11.8の場合
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1の場合
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# その他のCUDAバージョンは以下を参照:
# https://pytorch.org/get-started/locally/
```

2. その他の必要なパッケージをインストール:
```bash
pip install openai-whisper numpy requests
```

### Ollamaのインストールとセットアップ

1. Ollamaをインストール:
   - Windows/macOS/Linux: https://ollama.ai/ からダウンロード

2. gemma3:1bモデルをダウンロード（初回のみ）:
```bash
ollama pull gemma3:1b
```

### FFmpegのインストール

FFmpegをインストール（まだインストールしていない場合）:
   - Windows: https://ffmpeg.org/download.html からダウンロード
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg` または `sudo yum install ffmpeg`

## 使用方法

### 方法1: コマンドライン引数で動画パスを指定

```bash
python transcribe_video.py <動画ファイルのパス>
```

例:
```bash
python transcribe_video.py video.mp4
python transcribe_video.py "C:\Users\Username\Videos\my_video.mp4"
```

### 方法2: videoフォルダに動画を配置

`video`フォルダを作成し、その中に動画ファイルを配置すると、自動的に検出されます。

```bash
# videoフォルダを作成（まだない場合）
mkdir video

# 動画ファイルをvideoフォルダに配置後
python transcribe_video.py
```

### オプション引数

- モデルサイズを指定（デフォルト: base）:
```bash
python transcribe_video.py video.mp4 small
```

利用可能なモデルサイズ:
- `tiny`: 最も高速、精度は低め
- `base`: バランス型（推奨）
- `small`: より高精度
- `medium`: 高精度
- `large`: 最高精度、処理時間が長い

- 言語を指定（デフォルト: ja）:
```bash
python transcribe_video.py video.mp4 base en
```

- デバイスを指定（デフォルト: 自動検出）:
```bash
python transcribe_video.py video.mp4 base ja cuda  # GPUを使用
python transcribe_video.py video.mp4 base ja cpu   # CPUを使用
```

- Ollamaモデルを指定（デフォルト: gemma3:1b）:
```bash
python transcribe_video.py video.mp4 base ja cuda gemma3:1b
```

- Ollamaエンドポイントを指定（デフォルト: http://localhost:11434）:
```bash
python transcribe_video.py video.mp4 base ja cuda gemma3:1b http://localhost:11434
```

環境変数でも設定可能:
```bash
# Windows
set OLLAMA_MODEL=gemma3:1b
set OLLAMA_ENDPOINT=http://localhost:11434

# Linux/macOS
export OLLAMA_MODEL=gemma3:1b
export OLLAMA_ENDPOINT=http://localhost:11434
```

## CUDA（GPU）の使用

スクリプトは自動的にCUDAが利用可能かどうかを検出します。CUDAが利用可能な場合、自動的にGPUを使用して処理速度が大幅に向上します。

CUDAが利用可能かどうかを確認するには:
```python
import torch
print(torch.cuda.is_available())  # True の場合、CUDAが利用可能
```

## 出力

文字起こし結果はOllamaを使用して要約され、以下の形式で表示されます:

- **概要**: Ollama（gemma3:1b）で生成された文字起こし結果の要約

## 注意事項

- 初回実行時、Whisperモデルが自動的にダウンロードされます
- 処理時間は動画の長さとモデルサイズによって異なります
- 長い動画の場合は処理に時間がかかる場合があります
- CUDA版を使用すると、CPU版と比べて処理速度が大幅に向上します（特に大きなモデルサイズの場合）
- CUDAが利用できない環境では、自動的にCPUが使用されます
- **Ollamaサーバーが起動している必要があります**（`ollama serve` または自動起動）
- Ollamaサーバーに接続できない場合、要約の生成に失敗します
- 全文は表示されず、要約のみが表示されます

