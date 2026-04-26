# Windows local transcription with VibeVoice-ASR-HF

この手順は Hugging Face Transformers 対応版 `microsoft/VibeVoice-ASR-HF` を使い、音声ファイルをローカルPC上で文字起こしします。初回だけモデルをダウンロードします。音声ファイルを Hugging Face にアップロードして処理する方式ではありません。

モデルのダウンロード先は、このリポジトリ直下の `.hf-cache` です。このリポジトリを Gドライブに置いている場合、モデル本体も Gドライブ側に保存されます。

## 1. セットアップ

簡単に使う場合は、`Setup-VibeVoice-ASR.bat` をダブルクリックします。GPU が見つかれば CUDA 版 PyTorch、なければ CPU 版 PyTorch を入れます。

PowerShell で明示的に実行する場合は以下です。

PowerShell をリポジトリ直下で開きます。

```powershell
cd G:\Programming\AI-programming\microsoft-VibeVoice-260426-01
```

GPU/CPUを自動判定して環境を作ります。

```powershell
.\scripts\setup_vibevoice_asr_hf.ps1 -Torch auto
```

CPUのみのPCでは次を使います。

```powershell
.\scripts\setup_vibevoice_asr_hf.ps1 -Torch cpu
```

NVIDIA GPU のPCで自動判定が合わない場合は、PyTorch の CUDA wheel を明示します。

```powershell
.\scripts\setup_vibevoice_asr_hf.ps1 -Torch cu128
```

選択肢は `cpu`, `cu126`, `cu128`, `cu130` です。CUDA の対応は GPU そのものより NVIDIA ドライバと PyTorch wheel の組み合わせで決まります。うまく入らない場合は `cpu` で構築できますが、ASR-7B級モデルのため非常に遅くなります。

## 2. 文字起こし

### GUIで使う

`Start-VibeVoice-ASR-UI.bat` をダブルクリックすると簡単な画面が開きます。

画面でできること:

- 音声/動画ファイルを追加
- 出力フォルダを選択
- 固有名詞や参加者名などの補助キーワードを入力
- GPU/CPU の選択
- SRT 出力の有無を選択

PowerShell から起動する場合:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_ui.ps1
```

### コマンドで使う

```powershell
.\scripts\run_transcribe.ps1 -Audio "C:\path\meeting.wav"
```

結果は既定で `transcripts` に出力されます。

- `meeting.txt`: 本文だけ
- `meeting.json`: 生出力、本文、話者・時刻付きセグメント

話者・タイムスタンプがうまく解析できた場合に SRT も出したい場合:

```powershell
.\scripts\run_transcribe.ps1 -Audio "C:\path\meeting.wav" -WriteSrt
```

固有名詞、参加者名、専門用語を補助情報として渡す場合:

```powershell
.\scripts\run_transcribe.ps1 -Audio "C:\path\meeting.wav" -Prompt "製品名: Contoso Voice。参加者: 佐藤, Tanaka, Alice。"
```

長い補助情報は UTF-8 のテキストファイルにして渡せます。

```powershell
.\scripts\run_transcribe.ps1 -Audio "C:\path\meeting.wav" -PromptFile ".\hotwords.txt"
```

## 3. 別PCで使う場合

別PCではこのリポジトリをコピーまたは clone し、そのPCでセットアップを実行します。

```powershell
.\scripts\setup_vibevoice_asr_hf.ps1 -Torch auto
```

CPUしかないPCでは:

```powershell
.\scripts\setup_vibevoice_asr_hf.ps1 -Torch cpu
```

初回文字起こし時にモデルが Hugging Face キャッシュへ保存されます。ネットワークが制限されるPCでは、事前にモデルを明示的にダウンロードできます。

```powershell
.\.venv-vibevoice-asr\Scripts\python.exe .\local_asr\download_model.py --local-dir D:\models\VibeVoice-ASR-HF
.\scripts\run_transcribe.ps1 -Audio "C:\path\meeting.wav" -Model "D:\models\VibeVoice-ASR-HF" -Offline
```

## 4. メモリ不足時

GPUメモリ不足や CPU メモリ不足が出る場合は、トークナイザのチャンクを小さくします。

```powershell
.\scripts\run_transcribe.ps1 -Audio "C:\path\meeting.wav" -TokenizerChunkSize 64000
```

それでも足りない場合は短い音声に分割して処理してください。

`ページング ファイルが小さすぎるため、この操作を完了できません。 (os error 1455)` が出る場合は、Windows の仮想メモリ不足です。モデル読み込み時に大きな `safetensors` ファイルを扱うため、GPUメモリとは別にページングファイルも必要です。

対処:

- Windows の「仮想メモリ」をシステム管理サイズにする、または手動で大きくする
- 目安は 32768 MB 以上、余裕を見るなら 65536 MB
- 設定後に Windows を再起動する
- ブラウザ、ゲーム、動画編集、Jupyter などメモリを多く使うアプリを閉じてから実行する

## 5. 安全上の注意

- この構成では `--share` や Cloudflare tunnel は使いません。
- モデルは公式の `microsoft/VibeVoice-ASR-HF`、または自分で保存したローカルコピーだけを指定してください。
- 未信頼の音声・動画ファイルは避け、FFmpeg や PyTorch などの依存関係は定期的に更新してください。
