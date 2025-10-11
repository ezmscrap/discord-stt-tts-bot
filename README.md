# Discord Voice Caption & TTS Bot

Discord のボイスチャンネルで **テキスト読み上げ (TTS)** と **音声文字起こし (STT)** を行うボットです。gTTS と OpenAI Whisper API を組み合わせ、テキストチャンネルとボイスチャンネルを横断した運用をサポートします。

## 主な機能
- テキストチャンネルの投稿を gTTS + FFmpeg でリアルタイム読み上げ
- VOICEVOX サーバーを指定した読み上げにも対応（ユーザーごとの話者ID設定をサポート）
- ボイスチャンネルの音声を録音し、Whisper API で文字起こしして投稿
- 音声認識結果をユーザーごとに色分けしてEmbed表示（16色パレット／個別指定可）
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

詳細は[使い方ガイド](./docs/usage.md)を参照。

1. **仮想環境を作る**
   ```bash
   python3 -m venv .venv
   ```
2. **仮想環境を有効化**
   ```bash
   source .venv/bin/activate
   ```
3. **モジュールをインストール**
   ```bash
   pip install -e .
   ```
   > エディタブルインストールを行わない場合は `pip install -r requirements.txt` でも利用できます。
4. **`.env` ファイルのひな形を作成**
   ```bash
   cp .env.example .env
   ```
5. `.env` に Discord のトークンと OpenAI API キー（必要なら）を設定し、`TTS_PROVIDER` を `gtts` または `voicevox` に指定
6. **起動**
   ```bash
   python -m discord_stt_tts_bot
   ```

VOICEVOX を利用する場合は `VOICEVOX_BASE_URL`（例: `http://127.0.0.1:50021`）や `VOICEVOX_DEFAULT_SPEAKER` を `.env` で設定してください。FFmpeg が未導入の場合は別途インストールし、`ffmpeg -version` で利用可能か確認してください。字幕カラーは16色パレットから自動割り当てされ、`!sttcolor` コマンドでユーザー単位に上書きできます。`!voicevoxstyles` で話者IDとスタイル名を確認し、`!voxdict` で VOICEVOX のユーザー辞書を管理できます。詳細なセットアップやコマンド一覧は `docs/usage.md` を参照してください。

## 音声合成エンジンの切り替え
- `TTS_PROVIDER=gtts` … 既定の gTTS を使用し、`!ttsspeed` や `!ttsvoice` で話速・声色を調整します。
- `TTS_PROVIDER=voicevox` … `.env` の `VOICEVOX_BASE_URL` で指定した VOICEVOX エンジンを利用します。ユーザーごとの話者IDは `!ttsspeaker` コマンドでエクスポート／インポート・個別設定が可能です。
- VOICEVOX のスタイル一覧は `!voicevoxstyles` で確認でき、字幕カラーのパレットは `!sttpalette` で参照できます。
- VOICEVOX のユーザー辞書は `!voxdict export/import/add` で管理できます。

## コマンドの詳細

基本的な使い方


1. **機能呼び出し**
   - テキストチャンネルで `!join` を実行し、ボイスチャンネルにボットを入室させます。
   - ボットは実行ユーザーが参加しているボイスチャンネルへ移動／参加します。

2. **読み上げ**
   - 今いるテキストチャンネルを対象にする: `!readon`
   - 停止: `!readoff`

3. **読み上げ音声種類の指定**
   - 自分の読み上げ音声種類の変更:  `!ttsspeaker @自分 15`
   - 読み上げ音声種類一覧の表示: `!voicevoxstyles`

4. **音声文字起こし (字幕)**
   - 開始: `!stton`（既定は VAD による自動区切り）
   - 固定区切りに切り替え: `!stton fixed 8`（8 秒ごとに強制区切り）
   - ノイズ抑圧の切り替え: `!sttset denoise off`（FFmpeg フィルタを停止）
   - 利用モデルの変更: `!sttset sttmodel gpt-4o-mini-transcribe`（日本語固定）
   - 停止: `!sttoff`

5. **音声文字起こしの色の指定**
   - 自分の文字色の変更:  `!sttcolor @自分 3`
   - 文字色一覧の表示: `!sttpalette`

4. **機能の退出**
   - 機能を停止: `!leave`

詳細は[使い方ガイド](./docs/usage.md)を参照。

## ライセンス
MIT License
