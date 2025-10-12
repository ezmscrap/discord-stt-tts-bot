const TOKEN_STORAGE_KEY = "discord-sttts-gui-token";

/**
 * @typedef {Object} GuiConfig
 * @property {string} provider 使用している TTS プロバイダ。
 * @property {number} guild_id 対象ギルドの ID。
 * @property {boolean} requires_token 認証トークンが必要かどうか。
 * @property {Array<{guild_id:number,name:string}>} available_guilds 使用可能なギルド情報。
 * @property {number} [base_tempo] サーバー基準の読み上げ速度。
 * @property {number} [default_voicevox_speaker] VOICEVOX のデフォルト話者 ID。
 */

/**
 * アプリ全体の状態を保持する。
 * @type {{
 *   token: string,
 *   config: GuiConfig|null|undefined,
 *   requiresToken: boolean,
 *   loading: boolean,
 *   error: string
 * }}
 */
const state = {
  token: "",
  config: undefined,
  requiresToken: false,
  loading: false,
  error: "",
};

/**
 * localStorage に保存済みのトークンを読み出す。
 * @returns {string} 保存されたトークン。未設定の場合は空文字列。
 */
function loadStoredToken() {
  const token = window.localStorage.getItem(TOKEN_STORAGE_KEY);
  return token ? token : "";
}

/**
 * トークンを localStorage に保存する。
 * @param {string} token 保存するトークン文字列。
 * @returns {void}
 */
function saveToken(token) {
  window.localStorage.setItem(TOKEN_STORAGE_KEY, token);
}

/**
 * 保存済みトークンを削除する。
 * @returns {void}
 */
function clearStoredToken() {
  window.localStorage.removeItem(TOKEN_STORAGE_KEY);
}

/**
 * REST API に対してフェッチするヘルパー。
 * @param {string} path エンドポイントのパス。
 * @param {RequestInit} [options] fetch に渡すオプション。
 * @returns {Promise<Response>} fetch のレスポンス。
 * @throws {Error} ネットワークエラー発生時。
 */
async function apiFetch(path, options = {}) {
  const init = { ...options };
  const headers = new Headers(init.headers || {});
  if (state.token) {
    headers.set("X-Admin-Token", state.token);
  }
  init.headers = headers;
  init.credentials = init.credentials || "same-origin";

  try {
    return await fetch(path, init);
  } catch (error) {
    throw new Error(`ネットワークエラー: ${(error && error.message) || error}`);
  }
}

/**
 * 設定情報の取得を行い、状態を更新する。
 * @returns {Promise<void>}
 */
async function loadConfig() {
  if (state.loading) {
    return;
  }
  state.loading = true;
  if (!state.config) {
    render();
  }
  try {
    const response = await apiFetch("/api/gui/config");
    if (response.status === 401) {
      state.config = null;
      state.requiresToken = true;
      state.error = "管理トークンを入力してください。";
      state.token = "";
      clearStoredToken();
      return;
    }
    if (!response.ok) {
      state.config = null;
      state.error = `設定の取得に失敗しました (HTTP ${response.status})`;
      return;
    }
    const payload = await response.json();
    state.config = payload;
    state.requiresToken = Boolean(payload.requires_token);
    state.error = "";
  } catch (error) {
    state.config = null;
    state.error = error instanceof Error ? error.message : String(error);
  } finally {
    state.loading = false;
    render();
  }
}

/**
 * アプリの初期化を行う。
 * @returns {Promise<void>}
 */
async function initApp() {
  state.token = loadStoredToken();
  render();
  await loadConfig();
}

/**
 * 画面全体の再描画を行う。
 * @returns {void}
 */
function render() {
  const root = document.getElementById("app");
  if (!root) {
    throw new Error("app ルート要素が見つかりません。");
  }
  root.textContent = "";

  if (state.requiresToken && !state.token) {
    root.appendChild(renderTokenForm());
    return;
  }

  if (state.loading && !state.config) {
    root.appendChild(renderLoadingIndicator());
    return;
  }

  if (!state.config) {
    root.appendChild(renderSetupNotice());
    return;
  }

  root.appendChild(renderDashboard());
}

/**
 * トークン入力フォームを構築する。
 * @returns {HTMLElement} レンダリング用ノード。
 */
function renderTokenForm() {
  const form = document.createElement("form");
  form.className = "token-form";

  const title = document.createElement("h2");
  title.className = "token-form__title";
  title.textContent = "管理トークンを入力してください";

  const description = document.createElement("p");
  description.textContent = "環境変数 GUI_ADMIN_TOKEN に設定した値を入力します。";
  description.style.margin = "0";
  description.style.textAlign = "center";
  description.style.fontSize = "0.9rem";
  description.style.color = "rgba(255, 255, 255, 0.75)";

  const input = document.createElement("input");
  input.className = "token-form__input";
  input.type = "password";
  input.placeholder = "管理トークン";
  input.autocomplete = "off";
  input.value = state.token || "";

  const submit = document.createElement("button");
  submit.className = "token-form__submit";
  submit.type = "submit";
  submit.textContent = "接続";

  form.append(title, description, input);

  if (state.error) {
    const error = document.createElement("div");
    error.className = "token-form__error";
    error.textContent = state.error;
    form.appendChild(error);
  }

  form.appendChild(submit);

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const value = input.value.trim();
    if (!value) {
      state.error = "トークンを入力してください。";
      render();
      return;
    }
    state.token = value;
    saveToken(value);
    state.error = "";
    await loadConfig();
  });

  return form;
}

/**
 * 読み込み中インジケータを生成する。
 * @returns {HTMLElement} レンダリング用ノード。
 */
function renderLoadingIndicator() {
  const container = document.createElement("div");
  container.className = "loading-indicator";
  const spinner = document.createElement("div");
  spinner.className = "loading-indicator__spinner";
  const label = document.createElement("span");
  label.textContent = "設定を読み込み中...";
  container.append(spinner, label);
  return container;
}

/**
 * 初回セットアップの案内を表示する。
 * @returns {HTMLElement} レンダリング用ノード。
 */
function renderSetupNotice() {
  const message = document.createElement("div");
  message.className = "alert";
  message.textContent = state.error || "設定が取得できませんでした。再度お試しください。";
  return message;
}

/**
 * ダッシュボード全体のレイアウトを構築する。
 * @returns {HTMLElement} レンダリング用ノード。
 */
function renderDashboard() {
  const shell = document.createElement("div");
  shell.className = "app-shell";

  shell.appendChild(renderHeader());
  shell.appendChild(renderMainContent());
  return shell;
}

/**
 * ヘッダー部分を描画する。
 * @returns {HTMLElement} レンダリング用ノード。
 */
function renderHeader() {
  const header = document.createElement("header");
  header.className = "app-header";

  const titleRow = document.createElement("div");
  titleRow.className = "app-header__title";

  const title = document.createElement("h1");
  title.textContent = "Discord STT/TTS 管理 GUI";
  titleRow.appendChild(title);

  const controls = document.createElement("div");
  controls.className = "toolbar";

  const refreshButton = document.createElement("button");
  refreshButton.className = "button-secondary";
  refreshButton.type = "button";
  refreshButton.textContent = "設定を再取得";
  refreshButton.addEventListener("click", async () => {
    await loadConfig();
  });

  const resetTokenButton = document.createElement("button");
  resetTokenButton.className = "button-secondary";
  resetTokenButton.type = "button";
  resetTokenButton.textContent = "トークンをリセット";
  resetTokenButton.addEventListener("click", () => {
    clearStoredToken();
    state.token = "";
    state.config = null;
    state.requiresToken = true;
    state.error = "管理トークンを入力してください。";
    render();
  });

  controls.append(refreshButton, resetTokenButton);
  titleRow.appendChild(controls);

  const metaRow = document.createElement("div");
  metaRow.className = "app-header__meta";
  const providerChip = document.createElement("span");
  providerChip.className = "status-chip";
  providerChip.textContent = `TTS: ${(state.config && state.config.provider) || "unknown"}`;

  const guildLabel = document.createElement("span");
  guildLabel.textContent = resolveGuildLabel();

  metaRow.append(providerChip, guildLabel);

  if (state.error) {
    const alert = document.createElement("div");
    alert.className = "alert";
    alert.textContent = state.error;
    header.append(titleRow, metaRow, alert);
  } else {
    header.append(titleRow, metaRow);
  }

  return header;
}

/**
 * 現在選択中のギルド名を組み立てる。
 * @returns {string} ギルド表示用の文字列。
 */
function resolveGuildLabel() {
  if (!state.config) {
    return "ギルド情報なし";
  }
  const guildId = state.config.guild_id;
  const guild = (state.config.available_guilds || []).find(
    (item) => item.guild_id === guildId,
  );
  if (!guild) {
    return `ギルド ID: ${guildId}`;
  }
  return `ギルド: ${guild.name} (${guild.guild_id})`;
}

/**
 * メインコンテンツ領域を構築する。
 * @returns {HTMLElement} レンダリング用ノード。
 */
function renderMainContent() {
  const main = document.createElement("main");
  main.className = "app-main";

  main.appendChild(renderUserPanel());
  main.appendChild(renderSpeakerPanel());
  return main;
}

/**
 * ユーザリストパネルのプレースホルダーを描画する。
 * @returns {HTMLElement} レンダリング用ノード。
 */
function renderUserPanel() {
  const panel = document.createElement("section");
  panel.className = "panel";
  panel.id = "user-panel";

  const header = document.createElement("div");
  header.className = "panel__header";
  const title = document.createElement("h2");
  title.className = "panel__title";
  title.textContent = "ユーザリスト";
  header.appendChild(title);

  const content = document.createElement("div");
  content.className = "panel__content";
  content.id = "user-panel-content";
  const placeholder = document.createElement("div");
  placeholder.className = "placeholder";
  placeholder.textContent = "フェーズ3でユーザリストを実装予定です。";
  content.appendChild(placeholder);

  panel.append(header, content);
  return panel;
}

/**
 * 話者リストパネルのプレースホルダーを描画する。
 * @returns {HTMLElement} レンダリング用ノード。
 */
function renderSpeakerPanel() {
  const panel = document.createElement("section");
  panel.className = "panel";
  panel.id = "speaker-panel";

  const header = document.createElement("div");
  header.className = "panel__header";
  const title = document.createElement("h2");
  title.className = "panel__title";
  title.textContent = "話者リスト";
  header.appendChild(title);

  const content = document.createElement("div");
  content.className = "panel__content";
  content.id = "speaker-panel-content";
  const placeholder = document.createElement("div");
  placeholder.className = "placeholder";
  placeholder.textContent = "フェーズ4で話者リストと音声プレビューを実装予定です。";
  content.appendChild(placeholder);

  panel.append(header, content);
  return panel;
}

document.addEventListener("DOMContentLoaded", () => {
  initApp().catch((error) => {
    console.error("アプリ初期化中にエラーが発生しました。", error);
  });
});
