/**
 * DigitalTwin.ai - Core Frontend Application Controller
 * Handles WebSocket lifecycle, view navigation, and simulation speed controls.
 */

let ws = null;
let currentView = 'supervisor';
let isPaused = false;
let currentSpeed = 1.0;

document.addEventListener('DOMContentLoaded', () => {
    initWebSocket();
    initWaveformChart();
    initPlantManagerView();
    initLeadershipView();
});

function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host || 'localhost:8000';
    const wsUrl = `${protocol}//${host}/ws/live`;

    const statusDot = document.getElementById('wsStatusDot');
    const statusText = document.getElementById('wsStatusText');

    try {
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            if (statusDot) statusDot.className = 'w-2 h-2 rounded-full bg-emerald-400 animate-pulse';
            if (statusText) statusText.innerText = 'Stream: Live WebSocket';
            console.log('[WebSocket] Connected to DigitalTwin.ai stream.');
        };

        ws.onmessage = (event) => {
            try {
                const tickData = JSON.parse(event.data);
                handleLiveTick(tickData);
            } catch (e) {
                console.error('[WebSocket] Parse error:', e);
            }
        };

        ws.onclose = () => {
            if (statusDot) statusDot.className = 'w-2 h-2 rounded-full bg-amber-400';
            if (statusText) statusText.innerText = 'Stream: Reconnecting...';
            setTimeout(initWebSocket, 2500);
        };

        ws.onerror = (err) => {
            console.warn('[WebSocket] Error:', err);
            ws.close();
        };
    } catch (e) {
        console.error('[WebSocket] Init failed:', e);
    }
}

function handleLiveTick(tickData) {
    if (!tickData) return;

    // Header metadata
    const timeEl = document.getElementById('headerSimTime');
    const nameEl = document.getElementById('headerLineName');
    if (timeEl) timeEl.innerText = `Tick #${tickData.tick} (${tickData.timestamp.split('T')[1].slice(0, 8)})`;
    if (nameEl && tickData.line_name) nameEl.innerText = tickData.line_name;

    // Update current active view
    if (currentView === 'supervisor') {
        renderSupervisorView(tickData);
    }
}

function switchView(viewName) {
    currentView = viewName;

    // Update Nav Tabs
    document.querySelectorAll('.nav-tab').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.view-container').forEach(view => view.classList.remove('active'));

    if (viewName === 'supervisor') {
        document.getElementById('tabSupervisor').classList.add('active');
        document.getElementById('viewSupervisor').classList.add('active');
    } else if (viewName === 'plant_manager') {
        document.getElementById('tabPlantManager').classList.add('active');
        document.getElementById('viewPlantManager').classList.add('active');
        initPlantManagerView();
    } else if (viewName === 'leadership') {
        document.getElementById('tabLeadership').classList.add('active');
        document.getElementById('viewLeadership').classList.add('active');
        updateLeadershipCalculations();
    }
}

async function setSimSpeed(speed) {
    currentSpeed = speed;
    ['speed1Btn', 'speed2Btn', 'speed5Btn', 'speed10Btn'].forEach(id => {
        const btn = document.getElementById(id);
        if (btn) {
            btn.className = 'px-2 py-1 rounded-lg text-xs font-medium text-slate-400 hover:text-white transition';
        }
    });

    const activeId = speed === 1.0 ? 'speed1Btn' : speed === 2.0 ? 'speed2Btn' : speed === 5.0 ? 'speed5Btn' : 'speed10Btn';
    const activeBtn = document.getElementById(activeId);
    if (activeBtn) activeBtn.className = 'px-2 py-1 rounded-lg text-xs font-bold text-white bg-blue-600 transition';

    await fetch('/api/sim/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ speed: speed })
    });
}

async function togglePause() {
    isPaused = !isPaused;
    const icon = document.getElementById('pauseIcon');
    if (icon) {
        icon.className = isPaused ? 'fa-solid fa-play text-emerald-400' : 'fa-solid fa-pause';
    }

    await fetch('/api/sim/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pause: isPaused })
    });
}

async function stepSimulation() {
    await fetch('/api/sim/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ step: true })
    });
}

async function changeLineConfig(configPath) {
    await fetch('/api/sim/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config_file: configPath })
    });
}
