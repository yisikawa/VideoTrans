# VideoTrans - 動画文字起こしツール

ローカルの動画ファイルを読み込んで文字起こしを行い、LM Studioを使用して概要を生成するPythonスクリプトです。

## 必要な環境

- Python 3.8以上
- FFmpeg（Whisperが音声を抽出するために必要）
- LM Studio（要約生成に使用、デフォルトエンドポイント: http://localhost:1234）
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

### LM Studioのセットアップ

1. LM Studioをインストール:
   - Windows/macOS/Linux: https://lmstudio.ai/ からダウンロード

2. LM Studioを起動し、使用するモデルをロード

3. 「Local Server」タブでサーバーを起動（デフォルトポート: 1234）

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

- LM Studioモデルを指定（デフォルト: local-model）:
```bash
python transcribe_video.py video.mp4 base ja cuda local-model
```

- LM Studioエンドポイントを指定（デフォルト: http://localhost:1234）:
```bash
python transcribe_video.py video.mp4 base ja cuda local-model http://localhost:1234
```

環境変数でも設定可能:
```bash
# Windows
set LM_STUDIO_MODEL=local-model
set LM_STUDIO_ENDPOINT=http://localhost:1234

# Linux/macOS
export LM_STUDIO_MODEL=local-model
export LM_STUDIO_ENDPOINT=http://localhost:1234
```

## CUDA（GPU）の使用

スクリプトは自動的にCUDAが利用可能かどうかを検出します。CUDAが利用可能な場合、自動的にGPUを使用して処理速度が大幅に向上します。

CUDAが利用可能かどうかを確認するには:
```python
import torch
print(torch.cuda.is_available())  # True の場合、CUDAが利用可能
```

## 出力

文字起こし結果はLM Studioを使用して要約され、以下の形式で表示されます:

- **概要**: LM Studioで生成された文字起こし結果の要約

## 注意事項

- 初回実行時、Whisperモデルが自動的にダウンロードされます
- 処理時間は動画の長さとモデルサイズによって異なります
- 長い動画の場合は処理に時間がかかる場合があります
- CUDA版を使用すると、CPU版と比べて処理速度が大幅に向上します（特に大きなモデルサイズの場合）
- CUDAが利用できない環境では、自動的にCPUが使用されます
- **LM Studioのローカルサーバーが起動している必要があります**
- LM Studioサーバーに接続できない場合、要約の生成に失敗します
- 全文は表示されず、要約のみが表示されます
