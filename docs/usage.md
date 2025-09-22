# 使い方ガイド

Discord Voice Caption & TTS Bot の詳細なセットアップ手順と運用方法をまとめています。

---

## 動作要件

- **Python** 3.10 以上推奨
- **FFmpeg**（音声の速度/ピッチ調整・再生に使用）  
  - macOS: `brew install ffmpeg`  
  - Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y ffmpeg`  
  - Windows: winget 等で `winget install Gyan.FFmpeg` 後、`ffmpeg` が PATH に通っていることを確認
- **Discord Bot のトークン**
- （任意）**OpenAI API キー**（Whisper で音声認識を行う場合）
- Discord Developer Portal で **MESSAGE CONTENT INTENT** と **SERVER MEMBERS INTENT** を有効化しておくと安定します

---

## 初期設定

1. `.env` を作成します。付属の `.env.example` をコピーして必要な値を設定してください。
   ```bash
   cp .env.example .env
   # .env を編集して DISCORD_TOKEN 等を記入
   ```

   `.env` の主なキー：
   ```dotenv
   DISCORD_TOKEN=あなたのBotトークン
   OPENAI_API_KEY=（音声認識にOpenAI Whisper APIを使う場合のみ）
   TTS_LANG=ja
   TTS_TEMPO=1.05
   TTS_PROVIDER=gtts            # gtts / voicevox を選択
   VOICEVOX_BASE_URL=http://127.0.0.1:50021
   VOICEVOX_DEFAULT_SPEAKER=2
   VOICEVOX_TIMEOUT=15
   LOG_DIR=logs
   ```

2. 仮想環境を作成して依存パッケージをインストールします。
   ```bash
   python -m venv .venv

   # macOS / Linux
   source .venv/bin/activate

   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1

   pip install -e .
   ```
   > エディタブルインストールを行わない場合は `pip install -r requirements.txt` でも利用できます。

   `.gitignore` により `.env` と `.venv` は Git 管理から除外されています。

3. FFmpeg のインストールを確認します。
   ```bash
   ffmpeg -version
   ```

4. ボットを起動します。
   ```bash
   python -m discord_stt_tts_bot
   ```

   起動ログに `Logged in as ...` が表示されれば接続成功です。

---

## 推奨権限

サーバーへ招待する際、少なくとも以下の権限を付与してください。

* **ボイス**：Connect / Speak / Use Voice Activity
* **テキスト**：View Channels / Send Messages / Manage Threads（字幕をスレッドに投稿する場合）

---

## 基本的な使い方

1. テキストチャンネルで `!join` を実行し、ボイスチャンネルにボットを入室させます。
   実行ユーザーが参加しているボイスチャンネルへ移動／参加します。

2. **読み上げ**
   - 今いるテキストチャンネルを対象にする: `!readon`
   - 停止: `!readoff`

3. **音声文字起こし (字幕)**
   - 開始: `!stton`（デフォルト 10 秒ごとに区切って文字起こし）
   - 例: `!stton 8`（8 秒ごとに区切る）
   - 停止: `!sttoff`

4. 退出: `!leave`

> 読み上げ速度は初期値でも速めに設定しています。さらに調整したい場合は管理者向けコマンドを利用してください。

---

## コマンド一覧（抜粋）

### 基本コマンド
- `!join` … 今いるボイスチャンネルへ参加
- `!leave` … 退出
- `!readon` … 現在のテキストチャンネルの新規投稿を読み上げ
- `!readoff` … 読み上げ停止
- `!readhere` … 読み上げ対象チャンネルを“今ここ”に変更

### 文字起こし（STT）
- `!stton [区切り秒数(3-60)]` … 音声認識を開始し、テキストとして投稿（OpenAI Whisper 使用）
- `!sttoff` … 音声認識停止
- `!sttset` … VAD/最小長/マージ等の調整（例）
  - `!sttset vad 0.008` … 無音判定（RMS）しきい値
  - `!sttset vaddb -46` … dBFS しきい値
  - `!sttset mindur 0.4` … 送信する最短区間（秒）
  - `!sttset merge 14` … 同一話者・近接発話のメッセージ編集マージ秒
  - `!sttset lang auto` … Whisper の言語自動判定
  - `!sttset thread on` … 字幕をスレッドに投稿（off で同じチャンネルに投稿）

> 同一話者が続けて話した場合、`merge_window`（既定 6 秒、`!sttset merge` で変更）以内なら前のメッセージを編集で追記します。

### 読み上げ（TTS）管理（サーバー管理者のみ）
- `!ttsspeed <倍率>` … サーバー全体の基準話速を変更（gTTS 利用時のみ。例: `!ttsspeed 1.35`）
- `!ttsvoice @ユーザー <半音> [テンポ]` … 話者ごとの声色/話速係数を上書き（gTTS 利用時のみ）
  - 例: `!ttsvoice @太郎 +3 1.10` / 解除: `!ttsvoice @太郎 reset`
- `!ttsconfig` … 現在の TTS 設定を表示
- `!ttsspeaker ...` … VOICEVOX 利用時の話者IDを管理（`default` / `export` / `import` / `@ユーザー <id>` 等）

> 読み上げを gTTS で行う場合は、FFmpeg フィルタでピッチ（半音）と話速を調整しています。既定で話者ごとに声色が変わるよう自動割り当てされます（ユーザーIDベース）。
> VOICEVOX を選択した場合は、`.env` で `TTS_PROVIDER=voicevox` を設定し、`VOICEVOX_BASE_URL`/`VOICEVOX_DEFAULT_SPEAKER` と `!ttsspeaker` コマンドを組み合わせて管理します。

### デバッグ／補助
- `!stttest` … gTTS→Whisper の疎通テスト
- `!rectest [2-30]` … 一時録音テスト
- `!diag` … 環境診断
- `!logs` … TTS/STT のログファイル（CSV）を取得
- `!whereami` … 現在のチャンネル情報
- `!intentcheck` … intents の動作確認
- `!help [コマンド名]` … コマンドヘルプを表示

---

## トラブルシューティング

- **「Already recording.」が出る / 録音が止まらない**
  - 競合時は 1 周スキップされます。`!sttoff` → 数秒待機 → `!stton` を試してください。
  - ステートが崩れているときはボット再起動が有効です。

- **話しているのに「skip by VAD」になる**
  - `!sttset vad 0.008` / `!sttset vaddb -48` / `!sttset mindur 0.4` などでしきい値を調整してください。

- **読み上げが遅い**
  - `!ttsspeed 1.1` のように話速を上げる、またはユーザー個別に `!ttsvoice @ユーザー +3 1.15` などでテンポ係数を上げてください。
  - `.env` の `TTS_TEMPO` を大きくすることでも速度を上げられます。

- **Stage チャンネルで話せない**
  - モデレーター承認が必要な場合があります。`!join` 後の自動リクエストが失敗した場合は手動承認をお願いします。

- **ログ出力先を変更したい**
  - `.env` の `LOG_DIR` を変更してください。

---

## セキュリティと費用

- Whisper API を有効化すると音声データが OpenAI に送信されます。組織ポリシーに従って利用してください。
- OpenAI API の利用は従量課金です。高頻度利用時はコストをご確認ください。

---

## ライセンス
MIT License
