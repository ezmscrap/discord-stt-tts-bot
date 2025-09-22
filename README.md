# Discord Voice Caption & TTS Bot

Discord のボイスチャンネルで **テキスト読み上げ (TTS)** と **音声文字起こし (STT)** を行うボットです。gTTS と OpenAI Whisper API を組み合わせ、テキストチャンネルとボイスチャンネルを横断した運用をサポートします。

## 主な機能
- テキストチャンネルの投稿を gTTS + FFmpeg でリアルタイム読み上げ
- ボイスチャンネルの音声を録音し、Whisper API で文字起こしして投稿
- Stage チャンネル対応、字幕用スレッドの自動運用
- 読み上げ速度や声色プロファイルのカスタマイズ

## リポジトリ構成
```
.
├── docs/                    # 詳細ドキュメント
├── src/
│   └── discord_stt_tts_bot/  # ボット本体パッケージ
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## セットアップ手順 (Linux/Bash)
1. 仮想環境を作る
   ```bash
   python3 -m venv .venv
   ```
2. 仮想環境を有効化
   ```bash
   source .venv/bin/activate
   ```
3. モジュールをインストール
   ```bash
   pip install -e .
   ```
   > エディタブルインストールを行わない場合は `pip install -r requirements.txt` でも利用できます。
4. `.env` ファイルのひな形を作成
   ```bash
   cp .env.example .env
   ```
5. `.env` に Discord のトークンと OpenAI API キー（必要なら）を設定
6. 起動
   ```bash
   python -m discord_stt_tts_bot
   ```

FFmpeg が未導入の場合は別途インストールし、`ffmpeg -version` で利用可能か確認してください。詳細なセットアップやコマンド一覧は `docs/usage.md` を参照してください。

## ライセンス
MIT License
