from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from src.agent.chat_agent import ChatAgent
from src.agent.tools import AgentToolbox
from src.app import build_agent
from src.llm_client import UnavailableLLMClient
from src.memory.models import MemoryOperation


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Memory Agent Test UI</title>
  <style>
    :root {
      --bg: #f5f7f9;
      --panel: #ffffff;
      --panel-2: #eef3f6;
      --border: #cad4dc;
      --text: #172026;
      --muted: #63717d;
      --accent: #0f766e;
      --accent-2: #155e75;
      --danger: #b42318;
      --warning: #92400e;
      --shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; }

    /* ── Viewport lock: nothing on the page scrolls except designated areas ── */
    html {
      height: 100%;
      overflow: hidden;
    }
    body {
      height: 100%;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      background: var(--bg);
      color: var(--text);
    }

    /* ── Header: fixed strip ── */
    header {
      flex: 0 0 58px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 18px;
      background: #ffffff;
      border-bottom: 1px solid var(--border);
    }
    h1 { font-size: 18px; font-weight: 680; }
    .status-line {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
    }
    .dot {
      width: 9px; height: 9px;
      border-radius: 999px;
      background: var(--danger);
    }
    .dot.ready { background: var(--accent); }

    /* ── Main 3-col grid: fills remaining height exactly ── */
    main {
      flex: 1 1 0;
      min-height: 0;          /* critical: allow grid to shrink below content */
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr) 380px;
      grid-template-rows: 1fr;
      gap: 12px;
      padding: 12px;
      overflow: hidden;
    }

    /* ── Every grid cell: white card, can never grow past its slot ── */
    section, aside {
      background: var(--panel);
      border: 1px solid var(--border);
      box-shadow: var(--shadow);
      min-height: 0;
      min-width: 0;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }

    /* ── Shared panel pieces ── */
    .panel-head {
      flex: 0 0 auto;
      padding: 12px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }
    .panel-head h2 { font-size: 14px; font-weight: 680; }
    .panel-body {
      flex: 1 1 0;
      min-height: 0;
      padding: 12px;
      overflow-y: auto;
    }

    /* ── Form elements ── */
    label {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin: 10px 0 5px;
    }
    input, select, textarea {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 8px 9px;
      font: inherit;
      background: #ffffff;
      color: var(--text);
    }
    textarea { min-height: 74px; resize: vertical; }
    button {
      border: 1px solid var(--border);
      background: #ffffff;
      color: var(--text);
      border-radius: 6px;
      padding: 8px 10px;
      font: inherit;
      cursor: pointer;
    }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: white;
      font-weight: 650;
    }
    button.danger { border-color: #f3b4ae; color: var(--danger); }
    button:disabled { opacity: 0.55; cursor: not-allowed; }
    .row { display: flex; gap: 8px; align-items: center; }
    .row > * { flex: 1; }

    /* ── Chat column: header / scrollable messages / fixed composer ── */
    .chat {
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .messages {
      flex: 1 1 0;
      min-height: 0;
      overflow-y: auto;
      overflow-x: hidden;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      background: #fbfcfd;
    }
    .msg {
      flex: 0 0 auto;          /* never shrink: this is the scroll-fix */
      max-width: 82%;
      padding: 10px 11px;
      border-radius: 8px;
      white-space: pre-wrap;
      overflow-wrap: break-word;
      word-break: break-word;
      line-height: 1.38;
      border: 1px solid var(--border);
      background: #ffffff;
    }
    .msg.user {
      align-self: flex-end;
      background: #e7f5f1;
      border-color: #b7dfd3;
    }
    .msg.assistant { align-self: flex-start; }
    .msg.error {
      align-self: stretch;
      max-width: 100%;
      border-color: #f2b8b5;
      background: #fff1f0;
      color: var(--danger);
    }
    .msg-text { white-space: pre-wrap; }
    .msg-annotation {
      margin-top: 6px;
      font-size: 11.5px;
      color: var(--accent-2);
      cursor: pointer;
      user-select: none;
    }
    .msg-annotation:hover { text-decoration: underline; }
    .msg-annotation-details {
      display: none;
      margin-top: 4px;
      padding: 6px 8px;
      background: rgba(0,0,0,0.03);
      border-radius: 5px;
      font-size: 11.5px;
      color: var(--muted);
      line-height: 1.45;
    }
    .msg-annotation-details.open { display: block; }
    .msg-annotation-details div { padding: 2px 0; }
    .msg-annotation-details .ann-add { color: var(--accent); }
    .msg-annotation-details .ann-del { color: var(--danger); }
    .msg-annotation-details .ann-sup { color: var(--warning); }
    .composer {
      flex: 0 0 auto;
      padding: 12px;
      border-top: 1px solid var(--border);
      background: #ffffff;
    }
    .composer textarea {
      min-height: 50px;
      max-height: 120px;
      resize: none;
    }

    /* ── Right panel lists ── */
    .memory-list, .session-list, .history-list { display: grid; gap: 8px; }
    .memory {
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 9px;
      background: #ffffff;
    }
    .memory.inactive { background: #f7f8fa; color: var(--muted); }
    .meta { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 6px; }
    .pill {
      font-size: 11px; line-height: 1;
      padding: 4px 6px; border-radius: 999px;
      background: var(--panel-2); color: var(--muted);
    }
    .pill.active { color: var(--accent); background: #dff3ee; }
    .pill.deleted, .pill.superseded { color: var(--warning); background: #fff7ed; }
    .memory p, .item p { margin: 0; line-height: 1.35; }
    .tiny { font-size: 12px; color: var(--muted); word-break: break-all; }
    .stats {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px; margin-bottom: 12px;
    }
    .stat { background: var(--panel-2); border-radius: 6px; padding: 8px; }
    .stat strong { display: block; font-size: 18px; }

    /* ── Tabs bar ── */
    .tabs {
      flex: 0 0 auto;
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      border-bottom: 1px solid var(--border);
    }
    .tab {
      border: 0;
      border-right: 1px solid var(--border);
      border-radius: 0;
      padding: 10px 6px;
      background: #ffffff;
    }
    .tab.active { background: var(--panel-2); font-weight: 650; }
    .hidden { display: none !important; }
    .item { border-bottom: 1px solid var(--border); padding: 8px 0; }
    .item:last-child { border-bottom: 0; }

    /* ── Mobile: stack vertically, each panel gets a fixed height ── */
    @media (max-width: 1080px) {
      body { overflow-y: auto; }
      main {
        flex: 0 0 auto;
        grid-template-columns: 1fr;
        grid-template-rows: auto;
        overflow: visible;
      }
      section, aside { min-height: 0; height: 500px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Memory Agent Test UI</h1>
    <div class="status-line"><span id="chatDot" class="dot"></span><span id="chatStatus">Checking status</span></div>
  </header>
  <main>
    <aside class="left">
      <div class="panel-head"><h2>Run Context</h2><button id="newSessionBtn" title="Create a new session id">New Session</button></div>
      <div class="panel-body">
        <label for="userId">User ID</label>
        <input id="userId" value="default" autocomplete="off">
        <label for="sessionId">Session ID</label>
        <input id="sessionId" autocomplete="off">
        <div class="row" style="margin-top: 12px;">
          <button id="refreshBtn">Refresh</button>
          <button id="resetBtn" class="danger">Reset DB</button>
        </div>
        <hr style="border: 0; border-top: 1px solid var(--border); margin: 16px 0;">
        <h2 style="font-size: 14px; margin: 0 0 6px;">Manual Memory</h2>
        <label for="memoryContent">Content</label>
        <textarea id="memoryContent" placeholder="User prefers Java examples for coding questions."></textarea>
        <label for="memoryType">Type</label>
        <select id="memoryType">
          <option value="preference">preference</option>
          <option value="working_style">working_style</option>
          <option value="project_decision">project_decision</option>
          <option value="fact">fact</option>
          <option value="temporary">temporary</option>
        </select>
        <button id="addMemoryBtn" class="primary" style="width: 100%; margin-top: 10px;">Add Memory</button>
        <p id="toolMessage" class="tiny"></p>
      </div>
    </aside>

    <section class="chat">
      <div class="panel-head">
        <h2>Conversation</h2>
        <span id="dbPath" class="tiny"></span>
      </div>
      <div id="messages" class="messages"></div>
      <div class="composer">
        <label for="messageInput">Message</label>
        <textarea id="messageInput" placeholder="Remember that I prefer Python examples and concise explanations."></textarea>
        <div class="row" style="margin-top: 8px;">
          <button id="sampleBtn">Insert Sample</button>
          <button id="sendBtn" class="primary">Send</button>
        </div>
      </div>
    </section>

    <aside class="right">
      <div class="tabs">
        <button class="tab active" data-tab="memories">Memories</button>
        <button class="tab" data-tab="sessions">Sessions</button>
        <button class="tab" data-tab="history">History</button>
      </div>
      <div id="memoriesTab" class="panel-body">
        <div class="stats">
          <div class="stat"><span class="tiny">Total</span><strong id="statTotal">0</strong></div>
          <div class="stat"><span class="tiny">Active</span><strong id="statActive">0</strong></div>
          <div class="stat"><span class="tiny">Inactive</span><strong id="statInactive">0</strong></div>
        </div>
        <label for="searchInput">Search Active Memories</label>
        <div class="row">
          <input id="searchInput" placeholder="Java examples">
          <button id="searchBtn">Search</button>
        </div>
        <div id="memoryList" class="memory-list" style="margin-top: 12px;"></div>
      </div>
      <div id="sessionsTab" class="panel-body hidden">
        <div id="sessionList" class="session-list"></div>
      </div>
      <div id="historyTab" class="panel-body hidden">
        <div id="historyList" class="history-list"></div>
      </div>
    </aside>
  </main>
  <script>
    const state = {
      sessionId: localStorage.getItem('memoryAgentSession') || `ui-${Math.random().toString(16).slice(2, 10)}`,
      userId: localStorage.getItem('memoryAgentUser') || 'default',
      activeTab: 'memories'
    };

    const $ = (id) => document.getElementById(id);

    async function init() {
      $('sessionId').value = state.sessionId;
      $('userId').value = state.userId;
      localStorage.setItem('memoryAgentSession', state.sessionId);
      bindEvents();
      try {
        const query = new URLSearchParams({ user_id: state.userId, session_id: state.sessionId });
        const payload = await api(`/api/state?${query.toString()}`);
        renderState(payload);
        if (payload.history && payload.history.length) {
          loadChatFromHistory(payload.history);
        } else {
          addMessage('assistant', 'Use this UI to test chat, persistence, memory search, manual memory seeding, deletion, sessions, and history.');
        }
      } catch (error) {
        addMessage('assistant', 'Use this UI to test chat, persistence, memory search, manual memory seeding, deletion, sessions, and history.');
      }
    }

    function bindEvents() {
      $('sendBtn').addEventListener('click', sendMessage);
      $('messageInput').addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) sendMessage();
      });
      $('refreshBtn').addEventListener('click', refreshState);
      $('resetBtn').addEventListener('click', resetDb);
      $('newSessionBtn').addEventListener('click', () => {
        state.sessionId = `ui-${Math.random().toString(16).slice(2, 10)}`;
        $('sessionId').value = state.sessionId;
        localStorage.setItem('memoryAgentSession', state.sessionId);
        $('messages').replaceChildren();
        addMessage('assistant', `Started new session ${state.sessionId}.`);
        refreshState();
      });
      $('sampleBtn').addEventListener('click', () => {
        const samples = [
          'Remember that I prefer Python examples and concise explanations.',
          'Show me a stack implementation.',
          'Actually, use Java examples going forward, not Python.',
          'Forget that I prefer concise explanations.',
          'How should we test the Ledger project?'
        ];
        $('messageInput').value = samples[Math.floor(Math.random() * samples.length)];
        $('messageInput').focus();
      });
      $('addMemoryBtn').addEventListener('click', addManualMemory);
      $('searchBtn').addEventListener('click', searchMemories);
      $('userId').addEventListener('change', syncContext);
      $('sessionId').addEventListener('change', syncContext);
      document.querySelectorAll('.tab').forEach((button) => {
        button.addEventListener('click', () => switchTab(button.dataset.tab));
      });
    }

    function syncContext() {
      state.userId = $('userId').value.trim() || 'default';
      state.sessionId = $('sessionId').value.trim() || state.sessionId;
      localStorage.setItem('memoryAgentUser', state.userId);
      localStorage.setItem('memoryAgentSession', state.sessionId);
      refreshState();
    }

    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: { 'Content-Type': 'application/json' },
        ...options
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok || payload.ok === false) {
        throw new Error(payload.error || payload.message || `HTTP ${response.status}`);
      }
      return payload;
    }

    async function refreshState() {
      syncValuesOnly();
      try {
        const query = new URLSearchParams({ user_id: state.userId, session_id: state.sessionId });
        const payload = await api(`/api/state?${query.toString()}`);
        renderState(payload);
      } catch (error) {
        addMessage('error', error.message);
      }
    }

    function syncValuesOnly() {
      state.userId = $('userId').value.trim() || 'default';
      state.sessionId = $('sessionId').value.trim() || state.sessionId;
      localStorage.setItem('memoryAgentUser', state.userId);
      localStorage.setItem('memoryAgentSession', state.sessionId);
    }

    function renderState(payload) {
      $('dbPath').textContent = payload.db_path;
      $('chatStatus').textContent = payload.chat_available ? 'DeepSeek chat available' : 'Memory tools only: set DEEPSEEK_API_KEY for chat';
      $('chatDot').classList.toggle('ready', payload.chat_available);
      const stats = payload.stats || { total: 0, by_status: {} };
      const active = stats.by_status?.active || 0;
      $('statTotal').textContent = stats.total || 0;
      $('statActive').textContent = active;
      $('statInactive').textContent = Math.max(0, (stats.total || 0) - active);
      renderMemories(payload.memories || []);
      renderSessions(payload.sessions || []);
      renderHistory(payload.history || []);
    }

    function renderMemories(memories) {
      const list = $('memoryList');
      list.replaceChildren();
      if (!memories.length) {
        list.append(emptyItem('No memories stored.'));
        return;
      }
      for (const memory of memories) list.append(memoryCard(memory));
    }

    function memoryCard(memory) {
      const card = document.createElement('div');
      card.className = `memory ${memory.status === 'active' ? '' : 'inactive'}`;
      const meta = document.createElement('div');
      meta.className = 'meta';
      for (const label of [memory.status, memory.type, `i:${memory.importance}`]) {
        const pill = document.createElement('span');
        pill.className = `pill ${label}`;
        pill.textContent = label;
        meta.append(pill);
      }
      const content = document.createElement('p');
      content.textContent = memory.content;
      const id = document.createElement('div');
      id.className = 'tiny';
      id.textContent = memory.id;
      const actions = document.createElement('div');
      actions.className = 'row';
      actions.style.marginTop = '8px';
      const copy = document.createElement('button');
      copy.textContent = 'Copy ID';
      copy.addEventListener('click', () => navigator.clipboard?.writeText(memory.id));
      const forget = document.createElement('button');
      forget.textContent = 'Forget';
      forget.className = 'danger';
      forget.disabled = memory.status !== 'active';
      forget.addEventListener('click', () => forgetMemory(memory.id));
      actions.append(copy, forget);
      card.append(meta, content, id, actions);
      return card;
    }

    function renderSessions(sessions) {
      const list = $('sessionList');
      list.replaceChildren();
      if (!sessions.length) {
        list.append(emptyItem('No sessions stored.'));
        return;
      }
      for (const session of sessions) {
        const item = document.createElement('div');
        item.className = 'item';
        const p = document.createElement('p');
        p.textContent = `${session.session_id} · ${session.turn_count} turns`;
        const tiny = document.createElement('div');
        tiny.className = 'tiny';
        tiny.textContent = session.last_turn_at || '';
        const button = document.createElement('button');
        button.textContent = 'Use Session';
        button.style.marginTop = '6px';
        button.addEventListener('click', async () => {
          state.sessionId = session.session_id;
          $('sessionId').value = session.session_id;
          localStorage.setItem('memoryAgentSession', session.session_id);
          try {
            const query = new URLSearchParams({ user_id: state.userId, session_id: state.sessionId });
            const payload = await api(`/api/state?${query.toString()}`);
            renderState(payload);
            loadChatFromHistory(payload.history);
          } catch (error) {
            addMessage('error', error.message);
          }
          switchTab('history');
        });
        item.append(p, tiny, button);
        list.append(item);
      }
    }

    function renderHistory(turns) {
      const list = $('historyList');
      list.replaceChildren();
      const ordered = [...turns].reverse();
      if (!ordered.length) {
        list.append(emptyItem('No turns stored for this session.'));
        return;
      }
      for (const turn of ordered) {
        const item = document.createElement('div');
        item.className = 'item';
        const p = document.createElement('p');
        p.textContent = `${turn.role}: ${turn.content}`;
        const tiny = document.createElement('div');
        tiny.className = 'tiny';
        tiny.textContent = turn.created_at;
        item.append(p, tiny);
        list.append(item);
      }
    }

    async function sendMessage() {
      syncValuesOnly();
      const message = $('messageInput').value.trim();
      if (!message) return;
      const userNode = addMessage('user', message);
      $('messageInput').value = '';
      const assistantNode = addMessage('assistant', '');
      setBusy(true);
      try {
        const response = await fetch('/api/chat_stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_id: state.userId, session_id: state.sessionId, message })
        });
        if (!response.ok || !response.body) {
          const payload = await response.json().catch(() => ({}));
          throw new Error(payload.error || `HTTP ${response.status}`);
        }
        await readChatStream(response, assistantNode, userNode);
      } catch (error) {
        if (!(assistantNode._textEl || assistantNode).textContent) assistantNode.remove();
        addMessage('error', error.message);
      } finally {
        setBusy(false);
      }
    }

    async function readChatStream(response, assistantNode, userNode) {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          if (!line.trim()) continue;
          const event = JSON.parse(line);
          if (event.type === 'chunk') {
            assistantNode._textEl.textContent += event.content;
            $('messages').scrollTop = $('messages').scrollHeight;
          } else if (event.type === 'done') {
            renderState(event.state);
            if (event.related_memories && event.related_memories.length) {
              const items = event.related_memories.map(m => m.content);
              annotateMessage(userNode, `context: ${items.length} memories`, items);
            }
            pollForMemoryUpdates(event.state, assistantNode);
          } else if (event.type === 'error') {
            throw new Error(event.error);
          }
        }
      }
    }

    function pollForMemoryUpdates(baseState, assistantNode) {
      const baseIds = new Set((baseState.memories || []).map(m => m.id));
      let attempts = 0;
      const delays = [1000, 2000, 3000, 5000];
      function poll() {
        if (attempts >= delays.length) return;
        setTimeout(async () => {
          try {
            const query = new URLSearchParams({ user_id: state.userId, session_id: state.sessionId });
            const payload = await api(`/api/state?${query.toString()}`);
            renderState(payload);
            const newMems = payload.memories || [];
            const changes = diffMemories(baseIds, baseState.memories || [], newMems);
            if (changes.length) {
              if (assistantNode) {
                const labels = changes.map(c => c.label);
                const classes = changes.map(c => c.css);
                annotateMemoryChanges(assistantNode, changes);
              }
              return;
            }
          } catch (e) {}
          attempts++;
          poll();
        }, delays[attempts]);
      }
      poll();
    }

    function diffMemories(baseIds, oldMems, newMems) {
      const changes = [];
      const oldById = {};
      for (const m of oldMems) oldById[m.id] = m;
      const newById = {};
      for (const m of newMems) newById[m.id] = m;
      for (const m of newMems) {
        if (!baseIds.has(m.id) && m.status === 'active') {
          changes.push({ type: 'added', label: '+ ' + m.content, css: 'ann-add' });
        }
      }
      for (const m of oldMems) {
        if (m.status === 'active' && newById[m.id] && newById[m.id].status === 'deleted') {
          changes.push({ type: 'deleted', label: '- ' + m.content, css: 'ann-del' });
        }
        if (m.status === 'active' && newById[m.id] && newById[m.id].status === 'superseded') {
          changes.push({ type: 'superseded', label: '~ ' + m.content + ' (superseded)', css: 'ann-sup' });
        }
      }
      return changes;
    }

    function annotateMemoryChanges(node, changes) {
      if (!changes.length) return;
      const ann = document.createElement('div');
      ann.className = 'msg-annotation';
      ann.textContent = 'memory: ' + changes.length + ' change' + (changes.length > 1 ? 's' : '');
      const details = document.createElement('div');
      details.className = 'msg-annotation-details';
      for (const c of changes) {
        const row = document.createElement('div');
        row.className = c.css;
        row.textContent = c.label;
        details.append(row);
      }
      ann.addEventListener('click', () => {
        details.classList.toggle('open');
        $('messages').scrollTop = $('messages').scrollHeight;
      });
      node.append(ann, details);
      $('messages').scrollTop = $('messages').scrollHeight;
    }

    async function addManualMemory() {
      syncValuesOnly();
      const content = $('memoryContent').value.trim();
      if (!content) return;
      try {
        const payload = await api('/api/memories', {
          method: 'POST',
          body: JSON.stringify({
            user_id: state.userId,
            content,
            type: $('memoryType').value
          })
        });
        $('toolMessage').textContent = payload.message;
        $('memoryContent').value = '';
        renderState(payload.state);
      } catch (error) {
        $('toolMessage').textContent = error.message;
      }
    }

    async function searchMemories() {
      syncValuesOnly();
      const query = $('searchInput').value.trim();
      if (!query) {
        refreshState();
        return;
      }
      try {
        const payload = await api('/api/search', {
          method: 'POST',
          body: JSON.stringify({ user_id: state.userId, query })
        });
        renderMemories(payload.memories);
      } catch (error) {
        addMessage('error', error.message);
      }
    }

    async function forgetMemory(memoryId) {
      syncValuesOnly();
      try {
        const payload = await api('/api/forget', {
          method: 'POST',
          body: JSON.stringify({ memory_id: memoryId, user_id: state.userId, session_id: state.sessionId })
        });
        $('toolMessage').textContent = payload.message;
        renderState(payload.state);
      } catch (error) {
        $('toolMessage').textContent = error.message;
      }
    }

    async function resetDb() {
      if (!confirm('Delete all turns and memories in this UI database?')) return;
      syncValuesOnly();
      try {
        const payload = await api('/api/reset', {
          method: 'POST',
          body: JSON.stringify({ user_id: state.userId, session_id: state.sessionId })
        });
        $('messages').replaceChildren();
        addMessage('assistant', payload.message);
        renderState(payload.state);
      } catch (error) {
        addMessage('error', error.message);
      }
    }

    function loadChatFromHistory(history) {
      $('messages').replaceChildren();
      if (!history || !history.length) return;
      const chronological = [...history].reverse();
      for (const turn of chronological) {
        addMessage(turn.role, turn.content);
      }
    }

    function addMessage(role, text) {
      const node = document.createElement('div');
      node.className = `msg ${role}`;
      const textEl = document.createElement('div');
      textEl.className = 'msg-text';
      textEl.textContent = text;
      node.append(textEl);
      node._textEl = textEl;
      $('messages').append(node);
      $('messages').scrollTop = $('messages').scrollHeight;
      return node;
    }

    function annotateMessage(node, label, items, cssClass) {
      if (!items || !items.length) return;
      const ann = document.createElement('div');
      ann.className = 'msg-annotation';
      ann.textContent = label;
      const details = document.createElement('div');
      details.className = 'msg-annotation-details';
      for (const item of items) {
        const row = document.createElement('div');
        if (cssClass) row.className = cssClass;
        row.textContent = item;
        details.append(row);
      }
      ann.addEventListener('click', () => {
        details.classList.toggle('open');
        $('messages').scrollTop = $('messages').scrollHeight;
      });
      node.append(ann, details);
      $('messages').scrollTop = $('messages').scrollHeight;
    }

    function emptyItem(text) {
      const item = document.createElement('div');
      item.className = 'item tiny';
      item.textContent = text;
      return item;
    }

    function setBusy(busy) {
      $('sendBtn').disabled = busy;
      $('sendBtn').textContent = busy ? 'Sending...' : 'Send';
    }

    function switchTab(tab) {
      state.activeTab = tab;
      document.querySelectorAll('.tab').forEach((button) => button.classList.toggle('active', button.dataset.tab === tab));
      $('memoriesTab').classList.toggle('hidden', tab !== 'memories');
      $('sessionsTab').classList.toggle('hidden', tab !== 'sessions');
      $('historyTab').classList.toggle('hidden', tab !== 'history');
    }

    init();
  </script>
</body>
</html>
"""


class MemoryAgentUiServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], agent: ChatAgent):
        super().__init__(server_address, MemoryAgentUiHandler)
        self.agent = agent
        self.toolbox = AgentToolbox(store=agent.store, retriever=agent.retriever)

    @property
    def chat_available(self) -> bool:
        return not isinstance(self.agent.llm_client, UnavailableLLMClient)


class MemoryAgentUiHandler(BaseHTTPRequestHandler):
    server: MemoryAgentUiServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(INDEX_HTML)
            return
        if parsed.path == "/api/state":
            query = parse_qs(parsed.query)
            user_id = _first(query, "user_id", "default")
            session_id = _first(query, "session_id", "")
            self._send_json({"ok": True, **self._state(user_id=user_id, session_id=session_id)})
            return
        self._send_json({"ok": False, "error": "Not found."}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            if parsed.path == "/api/chat":
                self._handle_chat(payload)
            elif parsed.path == "/api/chat_stream":
                self._handle_chat_stream(payload)
            elif parsed.path == "/api/memories":
                self._handle_add_memory(payload)
            elif parsed.path == "/api/search":
                self._handle_search(payload)
            elif parsed.path == "/api/forget":
                self._handle_forget(payload)
            elif parsed.path == "/api/reset":
                self._handle_reset(payload)
            else:
                self._send_json({"ok": False, "error": "Not found."}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: Any) -> None:
        logging.getLogger("http").info(format, *args)

    def _handle_chat(self, payload: dict[str, Any]) -> None:
        user_id = str(payload.get("user_id") or "default")
        session_id = str(payload.get("session_id") or f"ui-{uuid.uuid4().hex[:8]}")
        message = str(payload.get("message") or "").strip()
        if not message:
            raise ValueError("message is required.")
        response = self.server.agent.chat(message, session_id=session_id, user_id=user_id)
        self._send_json(
            {
                "ok": True,
                "response": response,
                "state": self._state(user_id=user_id, session_id=session_id),
            }
        )

    def _handle_chat_stream(self, payload: dict[str, Any]) -> None:
        user_id = str(payload.get("user_id") or "default")
        session_id = str(payload.get("session_id") or f"ui-{uuid.uuid4().hex[:8]}")
        message = str(payload.get("message") or "").strip()
        if not message:
            raise ValueError("message is required.")

        related_memories = self.server.agent.retriever.retrieve(user_id, message, k=5)

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        try:
            for chunk in self.server.agent.chat_stream(message, session_id=session_id, user_id=user_id):
                self._send_ndjson({"type": "chunk", "content": chunk})
            self._send_ndjson(
                {
                    "type": "done",
                    "state": self._state(user_id=user_id, session_id=session_id),
                    "related_memories": [memory_to_public_dict(m) for m in related_memories],
                    "memory_update_pending": True,
                }
            )
        except Exception as exc:
            self._send_ndjson({"type": "error", "error": str(exc)})

    def _handle_add_memory(self, payload: dict[str, Any]) -> None:
        user_id = str(payload.get("user_id") or "default")
        session_id = str(payload.get("session_id") or "")
        content = str(payload.get("content") or "").strip()
        if not content:
            raise ValueError("content is required.")
        operation = MemoryOperation(
            op="ADD",
            content=content,
            type=str(payload.get("type") or "preference"),
            importance=int(payload.get("importance") or 3),
            confidence=float(payload.get("confidence") or 0.8),
            reason="Added manually through UI.",
        )
        changed = self.server.agent.apply_memory_operations([operation], user_id=user_id)
        if not changed:
            raise ValueError("Memory was rejected by policy or already exists.")
        self._send_json(
            {
                "ok": True,
                "message": f"Added memory {changed[0]}.",
                "state": self._state(user_id=user_id, session_id=session_id),
            }
        )

    def _handle_search(self, payload: dict[str, Any]) -> None:
        user_id = str(payload.get("user_id") or "default")
        query = str(payload.get("query") or "").strip()
        if not query:
            raise ValueError("query is required.")
        result = self.server.toolbox.search_memories(user_id, query, limit=10)
        self._send_json({"ok": True, "memories": [memory_to_public_dict(memory) for memory in result.data]})

    def _handle_forget(self, payload: dict[str, Any]) -> None:
        user_id = str(payload.get("user_id") or "default")
        session_id = str(payload.get("session_id") or "")
        memory_id = str(payload.get("memory_id") or "").strip()
        if not memory_id:
            raise ValueError("memory_id is required.")
        result = self.server.toolbox.forget_memory(memory_id)
        if not result.ok:
            raise ValueError(result.message)
        self._send_json(
            {
                "ok": True,
                "message": result.message,
                "state": self._state(user_id=user_id, session_id=session_id),
            }
        )

    def _handle_reset(self, payload: dict[str, Any]) -> None:
        user_id = str(payload.get("user_id") or "default")
        session_id = str(payload.get("session_id") or "")
        self.server.agent.store.reset_all()
        self._send_json(
            {
                "ok": True,
                "message": "All turns and memories removed from this database.",
                "state": self._state(user_id=user_id, session_id=session_id),
            }
        )

    def _state(self, *, user_id: str, session_id: str) -> dict[str, Any]:
        memories = self.server.toolbox.list_memories(user_id, include_inactive=True).data
        stats = self.server.toolbox.memory_stats(user_id).data
        sessions = self.server.toolbox.list_sessions(limit=30).data
        history = self.server.toolbox.session_history(session_id, limit=30).data if session_id else []
        return {
            "chat_available": self.server.chat_available,
            "db_path": self.server.agent.store.db_path,
            "memories": [memory_to_public_dict(memory) for memory in memories],
            "stats": stats,
            "sessions": sessions,
            "history": history,
        }

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length == 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object.")
        return data

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_ndjson(self, payload: dict[str, Any]) -> None:
        self.wfile.write(json.dumps(payload).encode("utf-8") + b"\n")
        self.wfile.flush()


def _first(query: dict[str, list[str]], key: str, default: str) -> str:
    values = query.get(key)
    if not values:
        return default
    return values[0] or default


def memory_to_public_dict(memory: Any) -> dict[str, Any]:
    return {
        "id": memory.id,
        "user_id": memory.user_id,
        "type": memory.type,
        "content": memory.content,
        "status": memory.status,
        "importance": memory.importance,
        "confidence": memory.confidence,
        "source_turn_id": memory.source_turn_id,
        "supersedes_id": memory.supersedes_id,
        "created_at": memory.created_at,
        "updated_at": memory.updated_at,
        "valid_until": memory.valid_until,
        "metadata": memory.metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Memory Agent browser UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--db", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    agent = build_agent(args.db)
    server = MemoryAgentUiServer((args.host, args.port), agent)
    url = f"http://{args.host}:{server.server_address[1]}"
    print(f"Memory Agent UI running at {url}")
    if not server.chat_available:
        print("Chat calls are disabled until DEEPSEEK_API_KEY is set; memory tools still work.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
