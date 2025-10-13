// ==UserScript==
// @name         CCFOLIA → Local Relay (Room Only, no-UI, manual bind)
// @namespace    ccfolia-relay
// @version      1.5
// @description  UIを一切挿入しない安全版。Alt+R→チャット欄クリックで監視ルートを手動指定。例外で停止しません。
// @match        https://ccfolia.com/rooms/*
// @run-at       document-end
// @grant        GM_xmlhttpRequest
// @connect      127.0.0.1
// @connect      localhost
// @connect      192.168.68.61
// @noframes
// ==/UserScript==
(function () {
  'use strict';

  const ENDPOINT = 'http://{CCFOLIA_BRIDGE_HOST}:{CCFOLIA_BRIDGE_PORT}/ccfolia_event';
  const SECRET   = '{CCFOLIA_POST_SECRET}';

  const SS_KEY = 'ccfo_room_root_sel_v2'; // 保存キー
  let rootEl = null;
  let mo = null;
  let armed = false;

  /* --- safe helpers --- */
  const $all = (sel, root=document) => { try { return Array.from(root.querySelectorAll(sel)); } catch { return []; } };
  const fmtISO = d => { try { return new Date(d.getTime()-d.getTimezoneOffset()*60000).toISOString(); } catch { return new Date().toISOString(); } };

  function log(...a){ try{ console.log('[CCFO]', ...a);}catch{} }
  function warn(...a){ try{ console.warn('[CCFO]', ...a);}catch{} }

/* --- 既存の発言(div)を「送信済み」扱いにする（POSTはしない） --- */
function markExistingAsSeen(root) {
  try {
    const nodes = $all(
      'div[class*="MuiListItemText-root"][class*="MuiListItemText-multiline"]:not([data-ccfo-sent])',
      root || document
    );
    for (const el of nodes) {
      try { el.setAttribute('data-ccfo-sent', '1'); } catch {}
    }
    log('primed existing messages:', nodes.length);
  } catch (e) { warn('prime err', e); }
}

/* --- HTTP POST (detailed logging) --- */
function postOne(payload) {
  const url = ENDPOINT;
  const body = JSON.stringify(payload || {});
  const headers = { 'Content-Type': 'application/json', 'X-CCF-Token': SECRET };

  // Tampermonkey v4/v5 互換 (どちらか片方しか無い環境に対応)
  const req = (typeof GM_xmlhttpRequest === 'function')
                ? GM_xmlhttpRequest
                : (GM && typeof GM.xmlHttpRequest === 'function'
                      ? GM.xmlHttpRequest
                      : null);

  if (!req) {
    warn('GM_xmlhttpRequest is not available. Check @grant and inject mode.');
    return;
  }


  try {
    req({
      method: 'POST',
      url,
      headers,
      data: body,
      timeout: 15000,        // 15s
      anonymous: true,       // Cookie 等を送らない（望ましい）
      // withCredentials: false, // 既定 false、明示してもOK

      onload: (res) => {
        // ここに必ず来ます（200/401/500 等すべて）
        console.info('[CCFO] onload', {
          status: res.status,
          statusText: res.statusText,
          responseText: (res.responseText || '').slice(0, 200)
        });
        // 401 などサーバ側の応答もここで確認できます
      },

      onerror: (err) => {
        // ネットワーク層のエラー（疎通不可など）
        console.error('[CCFO] onerror', err);
      },

      ontimeout: () => {
        console.error('[CCFO] timeout (15s) to', url);
      },

      onabort: () => {
        console.warn('[CCFO] aborted');
      },

      onreadystatechange: (res) => {
        // ステート変化を細かく見たい場合
        // 4 (= DONE) 以外はノイズになりがちなので必要ならコメント解除
        // console.debug('[CCFO] readyState', res.readyState, res.status);
      }
    });
  } catch (e) {
    warn('POST exception', e);
  }
}

  /* --- 行パース（簡易・堅牢） --- */
function parseLine(div){
  try {
    if (!div) return null;
    // スピーカー：h6（MuiListItemText-primary）。末尾の時間<span>は除外。
    const h6 =
      div.querySelector('h6[class*="MuiListItemText-primary"]') ||
      div.querySelector('h6');
    let speaker = '（未指定）';
    if (h6) {
      // h6の直テキストノードを連結（子spanのテキストは除外：時間が混ざるため）
      const parts = [];
      h6.childNodes.forEach(n => {
        if (n.nodeType === Node.TEXT_NODE) {
          const s = String(n.textContent || '').trim();
          if (s) parts.push(s);
        }
      });
      // 直テキストが取れない場合は、全体テキストから時間っぽい部分を落とす
      if (parts.length) {
        speaker = parts.join(' ').trim();
      } else {
        const raw = (h6.textContent || '').trim();
        // 「名前 - 今日 8:36」の「 - 以降」を削る簡易処理
        speaker = raw.replace(/\s*[-−ー・]\s*.*$/, '').trim() || raw || '（未指定）';
      }
    }
    // 本文：p（MuiListItemText-secondary）
    const p =
      div.querySelector('p[class*="MuiListItemText-secondary"]') ||
      div.querySelector('p');
    const text = (p && p.textContent) ? p.textContent.trim() : '';
    if (!text) return null;
    return { speaker, text };
  } catch {
    return null;
  }
}

  /* --- セレクタ生成（CSS.escape 非依存） --- */
  function cssPath(el){
    try {
      if (!(el instanceof Element)) return null;
      const parts = [];
      let cur = el, depth = 0;
      while (cur && cur.nodeType === 1 && depth < 8) {
        let name = cur.nodeName.toLowerCase();
        if (cur.id && /^[A-Za-z][\w\-\:\.]*$/.test(cur.id)) { // 簡易条件
          name += '#' + cur.id;
          parts.unshift(name);
          break;
        }
        // クラス1個だけなら採用（簡易）
        const cls = (cur.classList && cur.classList.length === 1) ? '.' + cur.classList[0] : '';
        if (cls) {
          parts.unshift(name + cls);
        } else {
          // nth-of-type
          let i = 1, sib = cur;
          while ((sib = sib.previousElementSibling)) if (sib.nodeName === cur.nodeName) i++;
          parts.unshift(name + `:nth-of-type(${i})`);
        }
        cur = cur.parentElement;
        depth++;
      }
      return parts.length ? parts.join(' > ') : null;
    } catch { return null; }
  }

  /* --- 初回スキャン＆監視 --- */
  function scanAndSend(scope){
    try {
      if (!rootEl) return;
      const base = scope || rootEl;
      const roomName = document.title || 'CCFOLIA';
      //const lines = $all('p:not([data-ccfo-sent])', base);
      //const lines = $all('p:not([data-ccfo-sent])', base);
      // 1発言の親div単位で拾う（クラス名の一部一致でMUIのhash変動に耐性）
      const lines = $all(
          'div[class*="MuiListItemText-root"][class*="MuiListItemText-multiline"]:not([data-ccfo-sent])',
          base
      );
      for (const p of lines) {
        try {
          const parsed = parseLine(p);
          if (!parsed) continue;
          postOne({ ...parsed, room: roomName, ts_client: fmtISO(new Date()) });
          try { p.setAttribute('data-ccfo-sent', '1'); } catch {}
        } catch {}
      }
    } catch (e) { warn('scan err', e); }
  }

  function bindRoot(el){
    try {
      if (!el || el === rootEl) return;
      rootEl = el;
      //scanAndSend(el);//bind直後は発言済みにしｍ発言しないように変更
      markExistingAsSeen(el);
      try { mo && mo.disconnect(); } catch {}
      mo = new MutationObserver(muts => {
        try {
          for (const m of muts) {
            m.addedNodes && m.addedNodes.forEach(n => {
              try {
                if (n.nodeType !== 1) return;
                if (n.tagName === 'P') scanAndSend(n);
                else scanAndSend(n);
              } catch {}
            });
          }
        } catch (e) { warn('MO tick err', e); }
      });
      mo.observe(el, { childList:true, subtree:true });
      log('bound to', el);
    } catch (e) { warn('bind err', e); }
  }

  /* --- 自動推定（シンプル版：スクロール可能＆<p>多め） --- */
  function autoFindRoot(){
    try {
      const cands = $all('div').filter(el => {
        try {
          const st = getComputedStyle(el);
          if (!(st.overflowY === 'auto' || st.overflowY === 'scroll')) return false;
          if (el.clientHeight < 120) return false;
          return el.querySelectorAll('p').length >= 5;
        } catch { return false; }
      });
      // <p> が多い順
      cands.sort((a,b)=>b.querySelectorAll('p').length - a.querySelectorAll('p').length);
      return cands[0] || null;
    } catch { return null; }
  }

  /* --- 手動バインド：Alt+R → クリック --- */
  function armManualBind(){
    try {
      if (armed) return;
      armed = true;
      log('manual bind armed: click chat area within 5s');
      const timer = setTimeout(()=>{ armed=false; log('manual bind timeout'); }, 5000);

      const onClick = (ev) => {
        if (!armed) return;
        ev.preventDefault(); ev.stopPropagation();
        armed = false; clearTimeout(timer);

        try {
          let best = null, bestScore = 0;
          let cur = ev.target; let steps = 0;
          while (cur && steps < 10) {
            if (cur instanceof HTMLElement) {
              const st = getComputedStyle(cur);
              const scrollable = (st.overflowY === 'auto' || st.overflowY === 'scroll');
              const ps = scrollable ? cur.querySelectorAll('p').length : 0;
              const score = (scrollable ? 10 : 0) + Math.min(ps, 50);
              if (score > bestScore) { bestScore = score; best = cur; }
            }
            cur = cur.parentElement; steps++;
          }
          const root = best || ev.target;
          bindRoot(root);
          const sel = cssPath(root);
          if (sel) { try { sessionStorage.setItem(SS_KEY, sel); } catch {} }
          log('manual bound selector =', sel || '(none)');
        } catch (e) {
          warn('manual bind err', e);
        }
        window.removeEventListener('click', onClick, true);
      };
      window.addEventListener('click', onClick, true);
    } catch (e) { warn('arm err', e); }
  }

  /* --- 起動 --- */
  function onReady(fn){
    const go = () => { try { fn(); } catch (e) { warn('init err', e); } };
    if (document.readyState === 'complete' || document.readyState === 'interactive') setTimeout(go, 0);
    else window.addEventListener('DOMContentLoaded', go, { once:true });
  }

  onReady(()=>{
    try {
      // 保存復元
      const saved = sessionStorage.getItem(SS_KEY);
      if (saved) {
        const el = document.querySelector(saved);
        if (el) bindRoot(el);
      }
      // 自動推定
      if (!rootEl) {
        const auto = autoFindRoot();
        if (auto) bindRoot(auto);
      }
      // 変化待ち（後から生成される場合）
      if (!rootEl) {
        const obs = new MutationObserver(()=>{
          if (rootEl) { try{obs.disconnect();}catch{} return; }
          const a = autoFindRoot();
          if (a) { try{obs.disconnect();}catch{} bindRoot(a); }
        });
        obs.observe(document.body || document.documentElement, { childList:true, subtree:true });
        // 10秒で諦め（以後手動）
        setTimeout(()=>{ try{obs.disconnect();}catch{} }, 10000);
      }
      // タブ切替などで root が消えたら再探索
      const lifeline = new MutationObserver(()=>{
        if (rootEl && !document.contains(rootEl)) {
          rootEl = null;
          const again = autoFindRoot();
          if (again) bindRoot(again);
        }
      });
      lifeline.observe(document.body || document.documentElement, { childList:true, subtree:true, attributes:true });

      // Alt+R で手動バインド
      window.addEventListener('keydown', (ev)=>{
        try {
          if (ev.altKey && !ev.shiftKey && !ev.ctrlKey && !ev.metaKey && (ev.key==='r' || ev.key==='R')) {
            armManualBind();
          }
        } catch {}
      }, true);

      log('started. Press Alt+R then click chat area if auto-binding fails.');
    } catch (e) { warn('main err', e); }
  });
})();
