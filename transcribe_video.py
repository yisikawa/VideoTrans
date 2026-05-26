"""
ローカルの動画ファイルを読み込んで文字起こしを行うスクリプト
"""
import whisper
import torch
import os
import sys
import json
import requests
from pathlib import Path
from typing import Optional


def transcribe_video(video_path: str, model_size: str = "base", language: str = "ja", device: str = None):
    """
    動画ファイルを文字起こしする
    
    Args:
        video_path: 動画ファイルのパス
        model_size: Whisperモデルのサイズ (tiny, base, small, medium, large)
        language: 言語コード (ja=日本語(japanese), en=英語など)
        device: 使用するデバイス (None=自動検出, "cuda", "cpu")
    
    Returns:
        文字起こし結果のテキスト
    """
    # ファイルの存在確認
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"動画ファイルが見つかりません: {video_path}")
    
    # デバイスの自動検出
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
            print(f"CUDAが利用可能です。GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print("CUDAが利用できないため、CPUを使用します。")
    else:
        if device == "cuda" and not torch.cuda.is_available():
            print("警告: CUDAが指定されましたが利用できないため、CPUを使用します。")
            device = "cpu"
    
    print(f"使用デバイス: {device}")
    print(f"Whisperモデル ({model_size}) を読み込んでいます...")
    model = whisper.load_model(model_size, device=device)
    
    print(f"文字起こし中: {os.path.basename(video_path)}")
    print("この処理には時間がかかる場合があります...")
    
    # 文字起こし実行
    result = model.transcribe(video_path, language=language)
    
    return result


def summarize_with_lmstudio(
    text: str,
    model: str = "local-model",
    lmstudio_endpoint: str = "http://localhost:1234",
    max_length: int = 400
) -> Optional[str]:
    """
    LM Studioを使用して要約を生成

    Args:
        text: 要約するテキスト
        model: LM Studioで読み込んでいるモデル名
        lmstudio_endpoint: LM Studio APIエンドポイント
        max_length: 要約の最大文字数

    Returns:
        要約されたテキスト
    """
    if not text or not text.strip():
        return None

    endpoint = lmstudio_endpoint.rstrip("/")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"以下の文章を日本語(japanese)で約{max_length}文字に要約してください。\n\n{text.strip()}"
            }
        ],
        "stream": False,
    }

    try:
        print(f"LM Studio ({model}) で要約を生成中...")
        response = requests.post(
            f"{endpoint}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        summary = data["choices"][0]["message"]["content"].strip()
        return summary if summary else None
    except requests.exceptions.ConnectionError:
        print(f"エラー: LM Studioサーバーに接続できませんでした。{endpoint} が起動しているか確認してください。")
        return None
    except Exception as exc:
        print(f"LM Studio要約エラー: {exc}")
        return None


def print_summary(result, lmstudio_model: str = "local-model", lmstudio_endpoint: str = "http://localhost:1234"):
    """
    文字起こし結果を要約してプリントアウト

    Args:
        result: Whisperの文字起こし結果
        lmstudio_model: LM Studioで読み込んでいるモデル名
        lmstudio_endpoint: LM Studio APIエンドポイント
    """
    print("\n" + "="*80)
    print("文字起こし結果の概要")
    print("="*80)

    # 文字起こしテキストを取得
    text = result.get("text", "")
    if not text or not text.strip():
        print("\n文字起こし結果が空です。")
        print("="*80)
        return

    # LM Studioで要約を生成
    summary = summarize_with_lmstudio(text, model=lmstudio_model, lmstudio_endpoint=lmstudio_endpoint)
    
    if summary:
        print("\n【概要】")
        print(summary)
    else:
        print("\n要約の生成に失敗しました。")
        print("文字起こしテキスト（最初の500文字）:")
        print(text[:500] + ("..." if len(text) > 500 else ""))
    
    print("\n" + "="*80)


def format_time(seconds: float) -> str:
    """
    秒数を時:分:秒形式に変換
    
    Args:
        seconds: 秒数
    
    Returns:
        時:分:秒形式の文字列
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main():
    """メイン処理"""
    # コマンドライン引数から動画パスを取得
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # 引数がない場合は、videoフォルダ内の動画ファイルを探す
        script_dir = Path(__file__).parent
        video_dir = script_dir / "video"
        
        # videoフォルダが存在しない場合はエラー
        if not video_dir.exists():
            print(f"エラー: videoフォルダが見つかりません: {video_dir}")
            print("使用方法:")
            print(f"  python {os.path.basename(__file__)} <動画ファイルのパス>")
            print("\nまたは、videoフォルダを作成して動画ファイルを配置してください。")
            sys.exit(1)
        
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.MP4', '.AVI', '.MOV', '.MKV']
        video_files = []
        for ext in video_extensions:
            video_files.extend(video_dir.glob(f"*{ext}"))
        
        if not video_files:
            print(f"エラー: videoフォルダ内に動画ファイルが見つかりません: {video_dir}")
            print("使用方法:")
            print(f"  python {os.path.basename(__file__)} <動画ファイルのパス>")
            print("\nまたは、videoフォルダに動画ファイルを配置してください。")
            sys.exit(1)
        
        video_path = str(video_files[0])
        print(f"動画ファイルを自動検出: {video_path}")
    
    # モデルサイズの指定（オプション）
    model_size = "base"
    if len(sys.argv) > 2:
        model_size = sys.argv[2]
    
    # 言語の指定（オプション）
    language = "ja"
    if len(sys.argv) > 3:
        language = sys.argv[3]
    
    # デバイスの指定（オプション）
    device = None
    if len(sys.argv) > 4:
        device = sys.argv[4].lower()
        if device not in ["cuda", "cpu"]:
            print(f"警告: 無効なデバイス '{device}'。自動検出を使用します。")
            device = None
    
    # LM Studioモデルの指定（オプション）
    lmstudio_model = os.getenv("LM_STUDIO_MODEL", "local-model")
    if len(sys.argv) > 5:
        lmstudio_model = sys.argv[5]

    # LM Studioエンドポイントの指定（オプション）
    lmstudio_endpoint = os.getenv("LM_STUDIO_ENDPOINT", "http://localhost:1234")
    if len(sys.argv) > 6:
        lmstudio_endpoint = sys.argv[6]

    try:
        # 文字起こし実行
        result = transcribe_video(video_path, model_size=model_size, language=language, device=device)

        # 要約をプリントアウト
        print_summary(result, lmstudio_model=lmstudio_model, lmstudio_endpoint=lmstudio_endpoint)
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

