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
   BOT_STATE_DIR=data          # 話者情報などを保存するディレクトリ（省略時は ./data）
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

> 読み上げ速度は初期値でも速めに設定しています。さらに調整したい場合は管理者向けコマンドを利用してください。

---

## Web API

Bot と C.C.FOLIA 等の連携システムとの間でイベントをやり取りするためのローカル API を提供しています。

- `POST /ccfolia_event` … イベントをキューへ投入し、指定チャンネルへ転送します。
- `OPTIONS /ccfolia_event` … ブラウザ向けの CORS プリフライト応答です。
- `GET /openapi.json` … OpenAPI(Swagger) 形式の仕様書を JSON で取得します。
- `GET /docs` … Swagger UI によるインタラクティブな API ドキュメントを表示します。

`CCFOLIA_POST_SECRET` を設定している場合、`POST /ccfolia_event` には `X-CCF-Token` ヘッダで同じ値を送信してください。未設定であればヘッダは不要です。IP 制限は `CCFOLIA_ACCEPT_FROM` で制御します。

### Swagger UI

Web サーバー起動後、`http://<ホスト>:<ポート>/docs` にアクセスすると Swagger UI で仕様とサンプルが確認できます。仕様の JSON は `/openapi.json` から取得できます。

### OpenAPI 抜粋

以下はトークンを利用する構成（`CCFOLIA_POST_SECRET` を設定した場合）の例です。

```yaml
openapi: 3.0.3
info:
  title: Discord STT/TTS Bot Bridge API
  version: 0.1.0
paths:
  /ccfolia_event:
    post:
      summary: テキストイベントの送信
      description: >
        C.C.FOLIA などから取得したイベントを Discord へ転送するためのキューに積みます。
        CCFOLIA_POST_SECRET を設定している場合は X-CCF-Token ヘッダで同じ値を送信してください。
      security:
        - CCFToken: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CCFoliaEventRequest'
      responses:
        '200':
          description: キュー投入に成功しました。
        '400':
          description: JSON 解析失敗や text の未指定など。
        '401':
          description: トークン不一致。
        '403':
          description: 許可されていない IP からのアクセス。
    options:
      summary: CORS プリフライト
components:
  securitySchemes:
    CCFToken:
      type: apiKey
      in: header
      name: X-CCF-Token
  schemas:
    CCFoliaEventRequest:
      type: object
      required:
        - text
      properties:
        speaker:
          type: string
          description: Discord に送信する表示名。省略時は（未指定）。
        text:
          type: string
          description: Discord に送信する本文。
        room:
          type: string
          description: 任意のルーム名やセッション名。
        ts_client:
          type: string
          description: 送信元クライアント識別子。
```

---

## コマンド一覧（抜粋）

### 基本操作
- `!join` … 参加者がいるボイスチャンネルへボットを接続します。例: `!join`
- `!leave` … ボイスチャンネルから切断します。例: `!leave`
- `!readon` … 現在のテキストチャンネルを読み上げ対象に設定します。例: `!readon`
- `!readoff` … 読み上げを停止します。例: `!readoff`
- `!readhere` … 読み上げ対象のテキストチャンネルを“今ここ”に変更します。例: `!readhere`

### 文字起こし（STT）
- `!stton [vad|fixed] [3-60]` … 文字起こしを開始します。例: `!stton`, `!stton fixed 8`
- `!sttoff` … 文字起こしを停止します。例: `!sttoff`
- `!sttset ...` … STT 設定を調整します。例: `!sttset vad 0.008`, `!sttset vadlevel 3`, `!sttset sttmodel gpt-4o-mini-transcribe`
- 小声に強くする: `!sttset gaintarget 0.07`, `!sttset gate off`
- `!sttstats` … モデル使用回数やフォールバック率・推定コストを確認します。例: `!sttstats`
- `!sttcolor ...` … 字幕カラーを管理します。例: `!sttcolor @自分 3`
- `!sttpalette` … 字幕カラーのパレットをプレビューします。例: `!sttpalette`

> 同一話者が続けて話した場合、`merge_window`（既定 6 秒、`!sttset merge` で変更）以内なら前メッセージへ追記します。
> 音声認識モデルは既定で `gpt-4o-mini-transcribe` を利用し、日本語 (`language="ja"`) 固定です。必要に応じて `!sttset sttmodel` / `sttmodel2` でモデル名を切り替えられます。

### 読み上げ（TTS）と VOICEVOX
- `!ttsspeed <倍率>` … サーバー基準の TTS 話速を設定します。例: `!ttsspeed 1.35`
- `!ttsvoice @ユーザー <半音> [テンポ]` … ユーザー別に声色を調整（gTTS 時）。例: `!ttsvoice @太郎 +3 1.10`
- `!ttsconfig` … 現在の TTS 設定を確認します。例: `!ttsconfig`
- `!ttsspeaker ...` … VOICEVOX 話者 ID を管理します。例: `!ttsspeaker default 2`, `!ttsspeaker @自分 5`
- `!voicevoxstyles` … VOICEVOX の話者/スタイル一覧を取得します。例: `!voicevoxstyles`
- `!voxdict <export|import|add>` … VOICEVOX のユーザー辞書を管理します。例: `!voxdict add テスト テスト`

> VOICEVOX を利用する場合は `.env` の `TTS_PROVIDER=voicevox` と `VOICEVOX_BASE_URL` を設定してください。

#### 話者情報指定機能の永続化
- `!ttsvoice` や `!ttsspeaker` で設定した読み上げ話者情報はギルドごとに `data/tts_profiles.json` へ自動保存され、ボットを再起動しても復元されます。
- 保存先ディレクトリは `BOT_STATE_DIR` 環境変数で変更できます（未指定時はリポジトリ直下の `data/`）。
- 設定を個別に初期化したい場合は各コマンドの `reset`/`clear` サブコマンドを利用してください。ファイルを削除すると全ギルド分のキャッシュが消去されます。

### デバッグ／ユーティリティ
- `!stttest` … gTTS→Whisper の疎通テストを実行します。例: `!stttest`
- `!rectest [2-30]` … 現在のボイスチャンネルを指定秒録音します。例: `!rectest 5`
- `!diag` … 環境診断情報を表示します。例: `!diag`
- `!logs` … STT/TTS ログ CSV を取得します。例: `!logs`
- `!whereami` … 現在のテキストチャンネル情報を表示します。例: `!whereami`
- `!intentcheck` … intents の動作状況を確認します。例: `!intentcheck`
- `!help [コマンド名]` … コマンドヘルプを表示します。例: `!help stton`

---


## バックグラウンド実行

シェルから直接起動するとターミナルが塞がるため、Linux では以下の方法が利用できます。

### systemd サービス
1. 仮想環境を有効化した状態でパスを確認します。
2. `/etc/systemd/system/discord-stt-tts-bot.service` を作成し、以下のように記述します。
   ```ini
   [Unit]
   Description=Discord STT/TTS Bot
   After=network.target

   [Service]
   Type=simple
   WorkingDirectory=/path/to/discord-stt-tts-bot
   Environment="PYTHONUNBUFFERED=1"
   ExecStart=/path/to/discord-stt-tts-bot/.venv/bin/python -m discord_stt_tts_bot
   Restart=on-failure
   RestartSec=5

   [Install]
   WantedBy=multi-user.target
   ```
3. サービスを有効化・起動します。
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now discord-stt-tts-bot.service
   sudo systemctl status discord-stt-tts-bot.service
   ```

### nohup/バックグラウンド実行
仮想環境を有効化した上で、以下のように実行するとログを `bot.log` に出力しながらバックグラウンドで動作します。
```bash
pip install -e .
source .venv/bin/activate
nohup python -m discord_stt_tts_bot > bot.log 2>&1 &
```

### tmux などを使用する
`tmux` や `screen` などを使ってセッションを維持する方法も有効です。

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
- **ノイズ抑圧が効いていない/エラーが出る**
  - `ffmpeg` コマンドが利用できるか確認してください。インストール済みであれば `!sttset denoise off` で一時的に無効化できます。
  - `ARNNDN_MODEL_PATH` を設定すると、FFmpeg `arnndn` モデルを明示できます（未設定時は自動で `afftdn` にフォールバックします）。
- **モデル切り替えの効果を見たい**
  - `logs` コマンドで `stt_metrics.csv` を取得すると、各モデルの使用状況や推定コストが確認できます。サーバー内では `!sttstats` で要約を表示できます。
- **小声が拾われにくい**
  - `!sttset gaintarget 0.07` や `!sttset gainmax 8` で自動ゲインを強められます。ノイズが増える場合は `!sttset gate on` / `gatethresh -60` でゲートを緩めて調整してください。

---

## セキュリティと費用

- Whisper API を有効化すると音声データが OpenAI に送信されます。組織ポリシーに従って利用してください。
- OpenAI API の利用は従量課金です。高頻度利用時はコストをご確認ください。

---

## ライセンス
MIT License
