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
 *   error: string,
 *   users: Array<any>,
 *   userLoading: boolean,
 *   userError: string,
 *   userMessage: string,
 *   userQuery: string,
 *   userTotal: number,
 *   speakers: Array<any>,
 *   speakerLoading: boolean,
 *   speakerError: string,
 *   speakerQuery: string,
 *   speakerTotal: number
 * }}
 */
const state = {
  token: "",
  config: undefined,
  requiresToken: false,
  loading: false,
  error: "",
  users: [],
  userLoading: false,
  userError: "",
  userMessage: "",
  userQuery: "",
  userTotal: 0,
  speakers: [],
  speakerLoading: false,
  speakerError: "",
  speakerQuery: "",
  speakerTotal: 0,
};

/**
 * 認証エラー発生時の状態リセットを行う。
 * @param {string} message 画面に表示するメッセージ。
 * @returns {void}
 */
function handleUnauthorized(message) {
  clearStoredToken();
  state.token = "";
  state.config = null;
  state.requiresToken = true;
  state.error = message;
  state.users = [];
  state.userLoading = false;
  state.userError = "";
  state.userMessage = "";
  state.userQuery = "";
  state.userTotal = 0;
  state.speakers = [];
  state.speakerLoading = false;
  state.speakerError = "";
  state.speakerQuery = "";
  state.speakerTotal = 0;
  state.loading = false;
  render();
}

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
async function loadConfig(options = {}) {
  if (state.loading) {
    return false;
  }
  state.loading = true;
  if (!state.config) {
    render();
  }
  let success = false;
  try {
    const response = await apiFetch("/api/gui/config");
    if (response.status === 401) {
      handleUnauthorized("管理トークンを入力してください。");
      return false;
    }
    if (!response.ok) {
      state.config = null;
      state.error = `設定の取得に失敗しました (HTTP ${response.status})`;
      return false;
    }
    const payload = await response.json();
    state.config = payload;
    state.requiresToken = Boolean(payload.requires_token) && !state.token;
    state.error = "";
    success = true;
  } catch (error) {
    state.config = null;
    state.error = error instanceof Error ? error.message : String(error);
  } finally {
    if (!state.requiresToken) {
      state.loading = false;
      render();
    }
  }
  if (success && options.reloadUsers !== false) {
    await loadUsers({ refresh: true });
  }
  if (success && state.config?.provider === "voicevox") {
    await loadSpeakers({ refresh: true });
  } else {
    state.speakers = [];
    state.speakerQuery = "";
    state.speakerTotal = 0;
  }
  return success;
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
    state.users = [];
    state.userError = "";
    state.userMessage = "";
    state.userQuery = "";
    state.userTotal = 0;
    state.speakers = [];
    state.speakerError = "";
    state.speakerQuery = "";
    state.speakerTotal = 0;
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

  const toolbar = renderUserToolbar();

  const content = document.createElement("div");
  content.className = "panel__content";
  content.id = "user-panel-content";

  if (state.userMessage) {
    const notice = document.createElement("div");
    notice.className = "notice notice--success";
    notice.textContent = state.userMessage;
    content.appendChild(notice);
  }

  if (state.userError) {
    const error = document.createElement("div");
    error.className = "alert";
    error.textContent = state.userError;
    content.appendChild(error);
  }

  if (state.userLoading) {
    content.appendChild(renderLoadingIndicator());
  } else if (!state.users || state.users.length === 0) {
    const empty = document.createElement("div");
    empty.className = "placeholder";
    empty.textContent = state.userQuery
      ? "検索条件に一致するユーザは見つかりませんでした。"
      : "ログからユーザ情報をまだ取得できていません。";
    content.appendChild(empty);
  } else {
    content.appendChild(renderUserList());
  }

  panel.append(header, toolbar, content);
  return panel;
}

/**
 * ユーザリスト用のツールバーを描画する。
 * @returns {HTMLElement} レンダリング用ノード。
 */
function renderUserToolbar() {
  const toolbar = document.createElement("div");
  toolbar.className = "toolbar";
  toolbar.id = "user-toolbar";

  const searchForm = document.createElement("form");
  searchForm.className = "toolbar__group";
  searchForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const input = searchForm.querySelector("input");
    const query = input ? input.value.trim() : "";
    await loadUsers({ query, refresh: true });
  });

  const input = document.createElement("input");
  input.type = "search";
  input.placeholder = "ユーザ名・IDで検索";
  input.value = state.userQuery || "";
  input.className = "token-form__input";
  input.style.margin = "0";

  const submit = document.createElement("button");
  submit.type = "submit";
  submit.className = "button-primary";
  submit.textContent = "検索";

  searchForm.append(input, submit);

  const resetButton = document.createElement("button");
  resetButton.type = "button";
  resetButton.className = "button-secondary";
  resetButton.textContent = "絞り込み解除";
  resetButton.addEventListener("click", async () => {
    if (!state.userQuery) {
      await loadUsers({ query: "", refresh: true, clearMessage: true });
      return;
    }
    input.value = "";
    await loadUsers({ query: "", refresh: true, clearMessage: true });
  });

  toolbar.append(searchForm, resetButton);

  const summary = document.createElement("span");
  summary.className = "toolbar__summary";
  summary.textContent = `表示件数: ${state.userTotal}`;
  toolbar.appendChild(summary);

  return toolbar;
}

/**
 * ユーザカードの一覧を描画する。
 * @returns {HTMLElement} レンダリング用ノード。
 */
function renderUserList() {
  const list = document.createElement("div");
  list.className = "user-list";
  for (const user of state.users) {
    list.appendChild(createUserCard(user));
  }
  return list;
}

/**
 * ユーザカードを生成する。
 * @param {any} user ユーザ情報。
 * @returns {HTMLElement} レンダリング用ノード。
 */
function createUserCard(user) {
  const card = document.createElement("article");
  card.className = "user-card";

  const header = document.createElement("div");
  header.className = "user-card__header";

  const title = document.createElement("h3");
  title.className = "user-card__title";
  title.textContent = user.user_name || user.user_display || "(名称未設定)";
  header.appendChild(title);

  const idGroup = document.createElement("div");
  idGroup.className = "user-card__ids";
  if (user.user_id != null) {
    idGroup.appendChild(createBadge(`user_id ${user.user_id}`, "badge--id"));
  } else {
    idGroup.appendChild(createBadge("user_id 未登録", "badge"));
  }
  idGroup.appendChild(createBadge(`author_id ${user.author_id}`, "badge--author"));
  header.appendChild(idGroup);

  const meta = document.createElement("div");
  meta.className = "user-card__meta";
  if (user.user_display) {
    meta.appendChild(createMetaRow("user_display", [user.user_display], "badge--display"));
  }
  if (user.author_display) {
    meta.appendChild(createMetaRow("author_display", [user.author_display], "badge--author"));
  }
  if (typeof user.speaker_id === "number") {
    meta.appendChild(
      createMetaRow("speaker_id", [String(user.speaker_id)], "badge--voicevox"),
    );
  }
  if (user.gtts_override) {
    const override = user.gtts_override;
    meta.appendChild(
      createMetaRow(
        "gTTS",
        [
          `semitones ${Number.parseFloat(override.semitones).toFixed(1)}`,
          `tempo ${Number.parseFloat(override.tempo).toFixed(2)}`,
        ],
        "badge--gtts",
      ),
    );
  }

  const actions = document.createElement("div");
  actions.className = "user-card__actions";
  const provider = state.config?.provider || "";
  if (provider === "gtts") {
    if (user.user_id != null) {
      actions.appendChild(createGttsControl(user));
    } else {
      const note = document.createElement("div");
      note.className = "notice";
      note.textContent = "gTTS 個別設定は user_id が取得できないため利用できません。";
      actions.appendChild(note);
    }
  } else if (provider === "voicevox") {
    actions.appendChild(createVoicevoxControl(user));
  } else {
    const note = document.createElement("div");
    note.className = "notice";
    note.textContent = `現在のプロバイダ (${provider}) では個別設定を利用できません。`;
    actions.appendChild(note);
  }

  card.append(header, meta, actions);
  return card;
}

/**
 * メタ情報の1行を生成する。
 * @param {string} label 左側のラベル。
 * @param {string[]} values 表示する値リスト。
 * @param {string} badgeClass バッジのクラス。
 * @returns {HTMLElement} レンダリング用ノード。
 */
function createMetaRow(label, values, badgeClass) {
  const row = document.createElement("div");
  row.className = "user-card__meta-row";
  const labelNode = document.createElement("span");
  labelNode.className = "user-card__meta-label";
  labelNode.textContent = label;
  row.appendChild(labelNode);
  const container = document.createElement("div");
  container.className = "user-card__meta-values";
  for (const value of values) {
    container.appendChild(createBadge(value, badgeClass));
  }
  row.appendChild(container);
  return row;
}

/**
 * バッジ要素を生成する。
 * @param {string} text 表示するテキスト。
 * @param {string} [variant] 追加クラス。
 * @returns {HTMLElement} バッジノード。
 */
function createBadge(text, variant) {
  const badge = document.createElement("span");
  badge.className = variant ? `badge ${variant}` : "badge";
  badge.textContent = text;
  return badge;
}

/**
 * 話者 ID から表示名を解決する。
 * @param {number} speakerId 話者 ID。
 * @returns {string} 表示名。
 */
function speakerNameById(speakerId) {
  const targetId = Number.parseInt(speakerId, 10);
  if (Number.isNaN(targetId)) {
    return `ID ${speakerId}`;
  }
  const speaker = state.speakers.find((item) => item.speaker_id === targetId);
  return speaker ? speaker.speaker_name || `ID ${targetId}` : `ID ${targetId}`;
}

/**
 * gTTS 個別設定フォームを生成する。
 * @param {any} user 対象ユーザ情報。
 * @returns {HTMLElement} レンダリング用ノード。
 */
function createGttsControl(user) {
  const container = document.createElement("div");
  container.className = "user-card__control";

  const form = document.createElement("form");
  form.className = "gtts-form";
  form.dataset.userId = String(user.user_id);

  const info = document.createElement("div");
  info.className = "notice";
  info.textContent = `対象: user_id ${user.user_id}`;
  form.appendChild(info);

  const semitoneWrapper = document.createElement("label");
  semitoneWrapper.textContent = "半音 (semitones)";
  const semitoneInput = document.createElement("input");
  semitoneInput.type = "number";
  semitoneInput.name = "semitones";
  semitoneInput.step = "0.1";
  semitoneInput.min = "-24";
  semitoneInput.max = "24";
  semitoneInput.value = user.gtts_override
    ? Number.parseFloat(user.gtts_override.semitones).toFixed(1)
    : "0.0";
  semitoneWrapper.appendChild(semitoneInput);

  const tempoWrapper = document.createElement("label");
  tempoWrapper.textContent = "テンポ倍率";
  const tempoInput = document.createElement("input");
  tempoInput.type = "number";
  tempoInput.name = "tempo";
  tempoInput.step = "0.05";
  tempoInput.min = "0.5";
  tempoInput.max = "3.0";
  tempoInput.value = user.gtts_override
    ? Number.parseFloat(user.gtts_override.tempo).toFixed(2)
    : "1.0";
  tempoWrapper.appendChild(tempoInput);

  const actions = document.createElement("div");
  actions.className = "gtts-form__actions";

  const submit = document.createElement("button");
  submit.type = "submit";
  submit.className = "button-primary";
  submit.textContent = "保存";

  const resetButton = document.createElement("button");
  resetButton.type = "button";
  resetButton.className = "button-secondary";
  resetButton.textContent = "リセット";

  actions.append(submit, resetButton);

  form.append(semitoneWrapper, tempoWrapper, actions);

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const semitones = Number.parseFloat(semitoneInput.value);
    const tempo = Number.parseFloat(tempoInput.value);
    if (Number.isNaN(semitones) || Number.isNaN(tempo)) {
      state.userError = "半音・テンポはいずれも数値で指定してください。";
      render();
      return;
    }
    state.userMessage = "保存中...";
    render();
    await handleGttsSubmit(user, semitones, tempo);
  });

  resetButton.addEventListener("click", async () => {
    state.userMessage = "リセット中...";
    render();
    await handleGttsReset(user);
  });

  container.appendChild(form);
  return container;
}

/**
 * VOICEVOX 向けコントロールのプレースホルダーを生成する。
 * @param {any} user 対象ユーザ情報。
 * @returns {HTMLElement} レンダリング用ノード。
 */
function createVoicevoxControl(user) {
  const container = document.createElement("div");
  container.className = "user-card__control";
  const primarySpeakerId = user.speaker_id;

  const info = document.createElement("div");
  info.className = "notice";
  info.textContent = `対象: author_id ${user.author_id}`;
  container.appendChild(info);

  if (state.speakerLoading) {
    const note = document.createElement("div");
    note.className = "notice";
    note.textContent = "VOICEVOX の話者リストを読み込み中です。";
    container.appendChild(note);
    return container;
  }

  if (!state.speakers.length) {
    const note = document.createElement("div");
    note.className = "notice";
    note.textContent = "話者リストが取得できていません。右側のパネルで再取得を実行してください。";
    container.appendChild(note);
    return container;
  }

  const form = document.createElement("form");
  form.className = "voicevox-form";

  const speakerLabel = document.createElement("label");
  speakerLabel.textContent = "話者";
  const speakerSelect = document.createElement("select");
  speakerSelect.name = "speaker-id";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "話者を選択してください";
  speakerSelect.appendChild(placeholder);
  for (const speaker of state.speakers) {
    const option = document.createElement("option");
    option.value = String(speaker.speaker_id);
    option.textContent = speaker.speaker_name || `ID ${speaker.speaker_id}`;
    speakerSelect.appendChild(option);
  }
  speakerLabel.appendChild(speakerSelect);

  const actions = document.createElement("div");
  actions.className = "voicevox-form__actions";
  const submit = document.createElement("button");
  submit.type = "submit";
  submit.className = "button-primary";
  submit.textContent = "保存";
  const resetButton = document.createElement("button");
  resetButton.type = "button";
  resetButton.className = "button-secondary";
  resetButton.textContent = "リセット";
  actions.append(submit, resetButton);

  form.append(speakerLabel, actions);

  const currentInfo = document.createElement("div");
  currentInfo.className = "user-card__current-speaker";

  const selectionHint = document.createElement("div");
  selectionHint.className = "user-card__selection-hint";

  const currentSpeakerId =
    typeof primarySpeakerId === "number" && !Number.isNaN(primarySpeakerId)
      ? String(primarySpeakerId)
      : "";
  if (currentSpeakerId) {
    if (!Array.from(speakerSelect.options).some((opt) => opt.value === currentSpeakerId)) {
      const missingOption = document.createElement("option");
      missingOption.value = currentSpeakerId;
      missingOption.textContent = `${speakerNameById(Number.parseInt(currentSpeakerId, 10))} (ID: ${currentSpeakerId})`;
      speakerSelect.appendChild(missingOption);
    }
    speakerSelect.value = currentSpeakerId;
    currentInfo.textContent = `現在: ${speakerNameById(Number.parseInt(currentSpeakerId, 10))} (ID: ${currentSpeakerId})`;
  } else {
    const defaultId = state.config?.default_voicevox_speaker;
    const defaultName = typeof defaultId === "number" ? speakerNameById(defaultId) : "未設定";
    currentInfo.textContent =
      typeof defaultId === "number"
        ? `現在: デフォルト (${defaultId}) / ${defaultName}`
        : "現在: デフォルト話者 (未設定)";
  }

  speakerSelect.addEventListener("change", () => {
    if (!speakerSelect.value) {
      selectionHint.textContent = "選択中: なし";
      return;
    }
    const speakerId = Number.parseInt(speakerSelect.value, 10);
    selectionHint.textContent = `選択中: ${speakerNameById(speakerId)} (ID: ${speakerId})`;
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const speakerId = Number.parseInt(speakerSelect.value, 10);
    if (Number.isNaN(speakerId)) {
      state.userError = "設定する話者を選択してください。";
      render();
      return;
    }
    state.userMessage = "VOICEVOX 話者を保存中...";
    render();
    await handleVoicevoxSubmit(user, speakerId);
  });

  resetButton.addEventListener("click", async () => {
    state.userMessage = "VOICEVOX 話者をリセット中...";
    render();
    await handleVoicevoxReset(user);
  });

  container.append(form, currentInfo, selectionHint);
  return container;
}

/**
 * gTTS 設定の保存を実行する。
 * @param {any} user 対象ユーザ情報。
 * @param {number} semitones 半音設定。
 * @param {number} tempo テンポ設定。
 * @returns {Promise<void>}
 */
async function handleGttsSubmit(user, semitones, tempo) {
  const guildId = state.config?.guild_id ?? 0;
  try {
    const response = await apiFetch(`/api/gui/gtts/${guildId}/${user.user_id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ semitones, tempo }),
    });
    if (response.status === 401) {
      handleUnauthorized("認証が無効になりました。再度トークンを入力してください。");
      return;
    }
    if (!response.ok) {
      state.userError = await extractErrorMessage(response);
      state.userMessage = "";
      render();
      return;
    }
    state.userMessage = "設定を保存しました。";
    await loadUsers({ refresh: true, clearMessage: false });
  } catch (error) {
    state.userError = error instanceof Error ? error.message : String(error);
    state.userMessage = "";
    render();
  }
}

/**
 * gTTS 設定のリセットを実行する。
 * @param {number} userId 対象ユーザID。
 * @returns {Promise<void>}
 */
async function handleGttsReset(user) {
  const guildId = state.config?.guild_id ?? 0;
  try {
    const response = await apiFetch(`/api/gui/gtts/${guildId}/${user.user_id}`, {
      method: "DELETE",
    });
    if (response.status === 401) {
      handleUnauthorized("認証が無効になりました。再度トークンを入力してください。");
      return;
    }
    if (!response.ok) {
      state.userError = await extractErrorMessage(response);
      state.userMessage = "";
      render();
      return;
    }
    state.userMessage = "設定をリセットしました。";
    await loadUsers({ refresh: true, clearMessage: false });
  } catch (error) {
    state.userError = error instanceof Error ? error.message : String(error);
    state.userMessage = "";
    render();
  }
}

/**
 * VOICEVOX 話者設定の保存を実行する。
 * @param {any} user 対象ユーザ情報。
 * @param {number} speakerId 設定する話者ID。
 * @returns {Promise<void>}
 */
async function handleVoicevoxSubmit(user, speakerId) {
  try {
    const response = await apiFetch(
      `/api/gui/voicevox/0/${user.author_id}`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          speaker_id: speakerId,
          author_id: user.author_id,
          user_id: user.user_id,
          author_display: user.author_display || "",
          user_display: user.user_display || "",
        }),
      },
    );
    if (response.status === 401) {
      handleUnauthorized("認証が無効になりました。再度トークンを入力してください。");
      return;
    }
    if (!response.ok) {
      state.userError = await extractErrorMessage(response);
      state.userMessage = "";
      render();
      return;
    }
    state.userMessage = `VOICEVOX 話者を設定しました (${speakerNameById(speakerId)})。`;
    await loadUsers({ refresh: true, clearMessage: false });
  } catch (error) {
    state.userError = error instanceof Error ? error.message : String(error);
    state.userMessage = "";
    render();
  }
}

/**
 * VOICEVOX 話者設定のリセットを実行する。
 * @param {any} user 対象ユーザ情報。
 * @returns {Promise<void>}
 */
async function handleVoicevoxReset(user) {
  try {
    const response = await apiFetch(
      `/api/gui/voicevox/0/${user.author_id}`,
      { method: "DELETE" },
    );
    if (response.status === 401) {
      handleUnauthorized("認証が無効になりました。再度トークンを入力してください。");
      return;
    }
    if (!response.ok) {
      state.userError = await extractErrorMessage(response);
      state.userMessage = "";
      render();
      return;
    }
    state.userMessage = "VOICEVOX 話者設定をリセットしました。";
    await loadUsers({ refresh: true, clearMessage: false });
  } catch (error) {
    state.userError = error instanceof Error ? error.message : String(error);
    state.userMessage = "";
    render();
  }
}

/**
 * API レスポンスからエラーメッセージを抽出する。
 * @param {Response} response フェッチレスポンス。
 * @returns {Promise<string>} エラーメッセージ。
 */
async function extractErrorMessage(response) {
  try {
    const data = await response.json();
    if (data && typeof data === "object") {
      if (typeof data.message === "string") {
        return data.message;
      }
      if (typeof data.error === "string") {
        return data.error;
      }
    }
  } catch {
    // JSON でない場合は無視してステータステキストを利用する。
  }
  return response.statusText || "不明なエラーが発生しました。";
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

  if (state.config?.provider !== "voicevox") {
    const content = document.createElement("div");
    content.className = "panel__content";
    content.id = "speaker-panel-content";
    const placeholder = document.createElement("div");
    placeholder.className = "placeholder";
    placeholder.textContent = "VOICEVOX を利用する場合に話者リストがここに表示されます。";
    content.appendChild(placeholder);
    panel.append(header, content);
    return panel;
  }

  const toolbar = renderSpeakerToolbar();

  const content = document.createElement("div");
  content.className = "panel__content";
  content.id = "speaker-panel-content";

  if (state.speakerError) {
    const alert = document.createElement("div");
    alert.className = "alert";
    alert.textContent = state.speakerError;
    content.appendChild(alert);
  }

  if (state.speakerLoading) {
    content.appendChild(renderLoadingIndicator());
  } else if (!state.speakers.length) {
    const placeholder = document.createElement("div");
    placeholder.className = "placeholder";
    placeholder.textContent = "利用可能な話者が見つかりません。\"再取得\" を押して更新してください。";
    content.appendChild(placeholder);
  } else {
    content.appendChild(renderSpeakerList());
  }

  panel.append(header, toolbar, content);
  return panel;
}

/**
 * 話者リスト用のツールバーを描画する。
 * @returns {HTMLElement} レンダリング用ノード。
 */
function renderSpeakerToolbar() {
  const toolbar = document.createElement("div");
  toolbar.className = "toolbar";
  toolbar.id = "speaker-toolbar";

  const searchForm = document.createElement("form");
  searchForm.className = "toolbar__group";
  searchForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const input = searchForm.querySelector("input");
    const query = input ? input.value.trim() : "";
    await loadSpeakers({ query, refresh: true });
  });

  const input = document.createElement("input");
  input.type = "search";
  input.placeholder = "話者名・IDで検索";
  input.value = state.speakerQuery || "";
  input.className = "token-form__input";
  input.style.margin = "0";

  const submit = document.createElement("button");
  submit.type = "submit";
  submit.className = "button-primary";
  submit.textContent = "検索";
  searchForm.append(input, submit);

  const resetButton = document.createElement("button");
  resetButton.type = "button";
  resetButton.className = "button-secondary";
  resetButton.textContent = "絞り込み解除";
  resetButton.addEventListener("click", async () => {
    if (!state.speakerQuery) {
      await loadSpeakers({ query: "", refresh: true });
      return;
    }
    input.value = "";
    await loadSpeakers({ query: "", refresh: true });
  });

  const refreshButton = document.createElement("button");
  refreshButton.type = "button";
  refreshButton.className = "button-secondary";
  refreshButton.textContent = "再取得";
  refreshButton.addEventListener("click", async () => {
    await loadSpeakers({ query: state.speakerQuery, refresh: true });
  });

  const summary = document.createElement("span");
  summary.className = "toolbar__summary";
  summary.textContent = `話者数: ${state.speakerTotal}`;

  toolbar.append(searchForm, resetButton, refreshButton, summary);
  return toolbar;
}

/**
 * 話者カードの一覧を描画する。
 * @returns {HTMLElement} レンダリング用ノード。
 */
function renderSpeakerList() {
  const list = document.createElement("div");
  list.className = "speaker-list";
  for (const speaker of state.speakers) {
    list.appendChild(createSpeakerCard(speaker));
  }
  return list;
}

/**
 * 話者カードを生成する。
 * @param {any} speaker 話者情報。
 * @returns {HTMLElement} レンダリング用ノード。
 */
function createSpeakerCard(speaker) {
  const card = document.createElement("article");
  card.className = "speaker-card";

  const header = document.createElement("div");
  header.className = "speaker-card__header";

  const iconWrapper = document.createElement("div");
  iconWrapper.className = "speaker-card__icon";
  if (speaker.icon) {
    const img = document.createElement("img");
    img.src = speaker.icon;
    img.alt = speaker.speaker_name || "speaker icon";
    iconWrapper.appendChild(img);
  } else {
    const placeholder = document.createElement("div");
    placeholder.className = "speaker-card__icon--empty";
    placeholder.textContent = "No Icon";
    iconWrapper.appendChild(placeholder);
  }

  const info = document.createElement("div");
  info.className = "speaker-card__info";
  const name = document.createElement("h3");
  name.className = "speaker-card__name";
  name.textContent = speaker.speaker_name || "(名称未設定)";
  const meta = document.createElement("div");
  meta.className = "speaker-card__meta";
  meta.textContent = `ID: ${speaker.speaker_id} / UUID: ${speaker.speaker_uuid}`;

  info.append(name, meta);
  header.append(iconWrapper, info);

  const samples = document.createElement("div");
  samples.className = "speaker-card__samples";
  const voiceSamples = Array.isArray(speaker.voice_samples) ? speaker.voice_samples : [];
  if (voiceSamples.length) {
    for (let index = 0; index < voiceSamples.length; index += 1) {
      const sample = voiceSamples[index];
      const sampleBox = document.createElement("div");
      sampleBox.className = "speaker-card__sample";
      const label = document.createElement("span");
      label.textContent = `サンプル ${index + 1}`;
      const audio = document.createElement("audio");
      audio.controls = true;
      audio.preload = "none";
      audio.src = sample.url || sample;
      sampleBox.append(label, audio);
      samples.appendChild(sampleBox);
    }
  } else {
    const empty = document.createElement("div");
    empty.className = "speaker-card__sample speaker-card__sample--empty";
    empty.textContent = "サンプル音声は利用できません。";
    samples.appendChild(empty);
  }

  card.append(header, samples);
  return card;
}

document.addEventListener("DOMContentLoaded", () => {
  initApp().catch((error) => {
    console.error("アプリ初期化中にエラーが発生しました。", error);
  });
});
/**
 * ユーザリストを取得する。
 * @param {{query?: string, refresh?: boolean, clearMessage?: boolean}} [options] 取得オプション。
 * @returns {Promise<void>}
 */
async function loadUsers(options = {}) {
  if (!state.config) {
    return;
  }
  if (state.userLoading) {
    return;
  }
  const query = options.query !== undefined ? options.query.trim() : state.userQuery;
  const shouldClearMessage = options.clearMessage !== false;
  if (shouldClearMessage) {
    state.userMessage = "";
  }
  state.userLoading = true;
  state.userError = "";
  if (!state.requiresToken) {
    render();
  }
  const params = new URLSearchParams({ guild_id: String(state.config.guild_id) });
  if (query) {
    params.set("q", query);
  }
  if (options.refresh) {
    params.set("refresh", "1");
  }
  try {
    const response = await apiFetch(`/api/gui/users?${params.toString()}`);
    if (response.status === 401) {
      handleUnauthorized("認証が無効になりました。再度トークンを入力してください。");
      return;
    }
    if (!response.ok) {
      state.users = [];
      state.userTotal = 0;
      state.userError = `ユーザ一覧の取得に失敗しました (HTTP ${response.status})`;
      return;
    }
    const payload = await response.json();
    state.users = Array.isArray(payload.users) ? payload.users : [];
    state.userTotal = typeof payload.total === "number" ? payload.total : state.users.length;
    state.userQuery = query;
  } catch (error) {
    state.users = [];
    state.userTotal = 0;
    state.userError = error instanceof Error ? error.message : String(error);
  } finally {
    state.userLoading = false;
    render();
  }
}

/**
 * VOICEVOX の話者リストを取得する。
 * @param {{query?: string, refresh?: boolean}} [options] 取得オプション。
 * @returns {Promise<void>}
 */
async function loadSpeakers(options = {}) {
  if (!state.config || state.config.provider !== "voicevox") {
    return;
  }
  if (state.speakerLoading) {
    return;
  }
  const query =
    options.query !== undefined ? options.query.trim() : state.speakerQuery;
  state.speakerLoading = true;
  state.speakerError = "";
  if (!state.requiresToken) {
    render();
  }
  const params = new URLSearchParams();
  if (query) {
    params.set("q", query);
  }
  if (options.refresh) {
    params.set("refresh", "1");
  }
  try {
    const response = await apiFetch(`/api/gui/voicevox/speakers?${params.toString()}`);
    if (response.status === 401) {
      handleUnauthorized("認証が無効になりました。再度トークンを入力してください。");
      return;
    }
    if (!response.ok) {
      state.speakers = [];
      state.speakerTotal = 0;
      state.speakerError = `話者リストの取得に失敗しました (HTTP ${response.status})`;
      return;
    }
    const payload = await response.json();
    state.speakers = Array.isArray(payload.speakers) ? payload.speakers : [];
    state.speakerTotal =
      typeof payload.total === "number" ? payload.total : state.speakers.length;
    state.speakerQuery = query;
  } catch (error) {
    state.speakers = [];
    state.speakerTotal = 0;
    state.speakerError = error instanceof Error ? error.message : String(error);
  } finally {
    state.speakerLoading = false;
    if (!state.requiresToken) {
      render();
    }
  }
}
