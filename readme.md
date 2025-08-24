# Discord Voice Caption & TTS Bot

Discord のボイスチャンネルで **テキストの読み上げ**、**音声の文字起こし（字幕化）** を行うボットです。  
以下の機能を備えています：

- **ボットの呼び出し**：`!join` でボイスチャンネルに参加 / `!leave` で退出
- **書き込みの読み上げ**：指定のテキストチャンネルに投稿された新規メッセージを gTTS + FFmpeg で読み上げ
- **音声の文字起こし → 投稿**：ボイスチャンネルの発話を一定間隔で録音し、OpenAI Whisper API で文字起こししてテキストチャンネルへ投稿  
  （Whisper は任意・API キー未設定なら STT 機能は使わずに TTS のみ動作）

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

1) `.env` を作成  
   付属の `.env.example` をコピーして `.env` を作成し、必要な値を設定します。

```bash
cp .env.example .env
# .env を編集して DISCORD_TOKEN 等を記入
````

`.env` の主なキー：

```dotenv
DISCORD_TOKEN=あなたのBotトークン
OPENAI_API_KEY=（音声認識にOpenAI Whisper APIを使う場合のみ）
TTS_LANG=ja
TTS_TEMPO=1.25
```

2. 仮想環境を作成して有効化

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

3. 依存パッケージをインストール

```bash
pip install -r requirements.txt
```

> 付属の `.gitignore` により `.env` と `.venv` は Git 管理から除外されます。

4. ボットを起動

```bash
python bot.py
```

起動ログに `Logged in as ...` が出れば接続成功です。

---

## 権限（推奨）

ボットをサーバーに招待する際、少なくとも以下の権限を付与してください：

* **ボイス**：Connect / Speak / Use Voice Activity
* **テキスト**：View Channels / Send Messages / Manage Threads（字幕をスレッドに投稿する場合）

---

## 使い方（基本フロー）

1. テキストチャンネルで **`!join`** を実行して、ボイスチャンネルにボットを入室させる
   （実行ユーザーが入室しているボイスチャンネルへ移動／参加します）

2. **書き込みの読み上げ**を有効化

   * 今いるテキストチャンネルを対象：`!readon`
   * 停止：`!readoff`

3. \*\*音声の文字起こし（字幕）\*\*を開始

   * `!stton`（デフォルト 10 秒ごとに区切って文字起こし）
   * 例：`!stton 8`（8 秒ごとに区切る）
   * 停止：`!sttoff`

4. 退出：`!leave`

> 読み上げ速度は初期状態でも早めに設定済みです。さらに速くしたい・声色を変えたい場合は管理者向けコマンド（下記）をご利用ください。

---

## コマンド一覧（抜粋）

### 基本

* `!join` … 今いるボイスチャンネルへ参加
* `!leave` … 退出
* `!readon` … “今ここ”のテキストチャンネルの新規投稿を読み上げ
* `!readoff` … 読み上げ停止
* `!readhere` … 読み上げ対象チャンネルを“今ここ”に変更

### 文字起こし（STT）

* `!stton [区切り秒数(3-60)]` … 音声認識を開始し、テキストとして投稿（OpenAI Whisper 使用）
* `!sttoff` … 音声認識停止
* `!sttset` … VAD/最小長/マージ等の調整（例）

  * `!sttset vad 0.008` … 無音判定（RMS）しきい値
  * `!sttset vaddb -46` … dBFS しきい値
  * `!sttset mindur 0.4` … 送信する最短区間（秒）
  * `!sttset merge 14` … 同一話者・近接発話のメッセージ編集マージ秒
  * `!sttset lang auto` … Whisper の言語自動判定
  * `!sttset thread on` … 字幕をスレッドに投稿（off で同じチャンネルに投稿）

> 同一話者が続けて話した場合、**`merge_window`**（既定 6 秒、`!sttset merge` で変更）以内なら前のメッセージを **編集で追記** します。

### 読み上げ（TTS）管理（サーバー管理者のみ）

* `!ttsspeed <倍率>` … サーバー全体の基準話速を変更（例：`!ttsspeed 1.35`）
* `!ttsvoice @ユーザー <半音> [テンポ]` … 話者ごとの声色/話速係数を上書き
  例：`!ttsvoice @太郎 +3 1.10` / 解除：`!ttsvoice @太郎 reset`
* `!ttsconfig` … 現在の TTS 設定を表示

> 読み上げは gTTS(日本語) → FFmpeg フィルタで**ピッチ（半音）**と**話速**を調整して再生します。
> 既定で話者ごとに声色が変わるよう **自動割当** されています（ユーザーIDに基づき安定）。

### デバッグ／補助

* `!stttest` … gTTS→Whisper の疎通テスト
* `!rectest [2-30]` … 一時録音テスト
* `!diag` … 環境診断
* `!whereami` … 現在のチャンネル情報
* `!intentcheck` … intents の動作確認
* `!help [コマンド名]` … コマンドヘルプを表示（ヘルプコマンド）


---

## よくあるトラブル

* **「Already recording.」が出る / 録音が止まらない**
  内部で安全弁を入れていますが、競合時は 1 周スキップされます。問題が続く場合は `!sttoff` → 数秒待機 → `!stton` を試してください。
  （ステートが崩れているときはボット再起動も有効です）

* **話しているのに「skip by VAD」になる**
  VAD しきい値を緩めてください：
  `!sttset vad 0.008` / `!sttset vaddb -48` / `!sttset mindur 0.4` など

* **読み上げが遅い**
  `!ttsspeed 1.35` のように話速を上げる、またはユーザー個別に `!ttsvoice @ユーザー +3 1.15` などでテンポ係数を上げてください。

* **Stage チャンネルで話せない**
  モデレーター承認が必要な場合があります（`!join` 後に自動でリクエストしますが、権限により失敗することがあります）。

---

## セキュリティと費用

* Whisper API を有効化すると、**音声データが OpenAI に送信**されます。組織のポリシーに従ってご利用ください。
* OpenAI API 利用分は **従量課金** です。高頻度利用時はご注意ください。

---

## ファイル構成（主要）

```
.
├── bot.py            # ボット本体
├── requirements.txt  # 依存パッケージ
├── .env.example      # 環境変数サンプル
├── .gitignore        # .env / .venv を無視
└── readme.md         # このファイル
```

---

## ライセンス

MIT License
