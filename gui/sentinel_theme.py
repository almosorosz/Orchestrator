"""CrystalLoop styles applied only to the embedded Sentinel dashboard."""

DASHBOARD_STYLESHEET = """
:root {
    --r: 12px;
    --primary: #2563eb;
    --focus: #3b82f6;
    --sans: "Segoe UI", system-ui, sans-serif;
    --mono: Consolas, ui-monospace, monospace;
}

:root[data-crystalloop-theme="dark"] {
    color-scheme: dark;
    --bg: #07111f;
    --panel: #111d31;
    --input: #0b1728;
    --ink: #e5e7eb;
    --muted: #b4c1d4;
    --faint: #94a3b8;
    --line: #16263d;
    --line-hi: #2e435e;
    --hover: #10203a;
    --signal: #93c5fd;
    --live: #34d399;
    --amber: #fbbf24;
    --coral: #fca5a5;
    --warning-bg: #30230e;
}

:root[data-crystalloop-theme="light"] {
    color-scheme: light;
    --bg: #f8fafc;
    --panel: #ffffff;
    --input: #ffffff;
    --ink: #1e293b;
    --muted: #475569;
    --faint: #64748b;
    --line: #d8dee8;
    --line-hi: #a5b4c8;
    --hover: #e8f0fe;
    --signal: #1d4ed8;
    --live: #15803d;
    --amber: #92400e;
    --coral: #b91c1c;
    --warning-bg: #fef3c7;
}

body {
    font-family: var(--sans);
    scrollbar-color: var(--line-hi) transparent;
}

body, .detail { background: var(--bg); }
body::before, .quad::after { display: none; }

.quad, .sensor-card, .core-face, .history-status > div, .history-settings {
    background: var(--panel);
    box-shadow: none;
}

.quad, .sensor-card { border-radius: var(--r); }

.quad:hover, .quad:active, .sensor-card:hover {
    border-color: var(--focus);
    box-shadow: none;
    transform: none;
}

.q-title, .info-list dd, .s-name, .s-val, #clockTime, #wxTemp,
.detail-title h2, .field input, .field textarea, .sensor-name,
.sensor-val, .probe-current .big, .empty h3, .log-name {
    color: var(--ink);
}

.info-list dd.empty-val { color: var(--faint); }

.mono, .s-val, .sensor-id, .empty code, .log-time { font-family: var(--mono); }

.q-ico, .field input, .field textarea, .btn, .back,
.history-toolbar button, .history-toolbar select, .history-toolbar a,
.history-toolbar input, .history-settings input {
    border-radius: 8px;
}

.back:hover, .history-toolbar button:hover, .history-toolbar a:hover {
    background: var(--hover);
    border-color: var(--focus);
}

.btn {
    background: var(--primary);
    border-color: var(--focus);
    color: #fff;
}

.field input:focus, .field textarea:focus {
    border-color: var(--focus);
    box-shadow: 0 0 0 3px #3b82f633;
}

.dot.live {
    background: var(--live);
    animation: none;
}

.badge-demo {
    background: var(--warning-bg);
    border-color: var(--amber);
}

.log-row.warn .log-tag { border-color: var(--amber); }
.log-row.alert .log-tag { border-color: var(--coral); }
.cam-fallback { color: #cbd5e1; }
"""
