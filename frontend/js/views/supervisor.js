/**
 * DigitalTwin.ai - Floor Supervisor (Real-Time Operations) View
 * Renders the live station flow topology, pulsing anomaly nodes,
 * explainable AI alerts, and 1-click PLC dispatch action controls.
 */

let selectedStationId = 8;
let currentActiveAlert = null;

function renderSupervisorView(tickData) {
    if (!tickData || !tickData.stations) return;

    // Update KPI Bar
    const kpiThroughput = document.getElementById('kpiThroughput');
    const kpiHealth = document.getElementById('kpiHealth');
    const kpiBacklog = document.getElementById('kpiBacklog');
    const kpiAlertCount = document.getElementById('kpiAlertCount');

    if (kpiThroughput) kpiThroughput.innerText = tickData.plant_kpis.average_throughput_uph.toFixed(1);
    if (kpiHealth) kpiHealth.innerText = `${tickData.plant_kpis.plant_health_index.toFixed(1)}%`;
    if (kpiBacklog) kpiBacklog.innerText = tickData.plant_kpis.total_line_backlog_units;
    if (kpiAlertCount) kpiAlertCount.innerText = tickData.plant_kpis.active_alert_count;

    // Group stations by Zone
    const zones = [
        { id: "Body Construction", name: "Zone 1: Body Construction", color: "blue", border: "border-blue-500/30", badge: "bg-blue-500/20 text-blue-300" },
        { id: "Paint & Coat", name: "Zone 2: Paint & Coat", color: "purple", border: "border-purple-500/30", badge: "bg-purple-500/20 text-purple-300" },
        { id: "Final Assembly", name: "Zone 3: Final Assembly", color: "emerald", border: "border-emerald-500/30", badge: "bg-emerald-500/20 text-emerald-300" }
    ];

    const container = document.getElementById('stationNodesContainer');
    if (!container) return;

    let html = '';

    zones.forEach(zone => {
        const zoneStations = tickData.stations.filter(s => s.zone === zone.id);
        if (zoneStations.length === 0) return;

        html += `
        <div class="glass-card p-3 rounded-xl border ${zone.border}">
            <div class="flex items-center justify-between mb-2">
                <span class="text-xs font-bold ${zone.badge} px-2.5 py-0.5 rounded-full uppercase tracking-wider">${zone.name}</span>
                <span class="text-[10px] text-slate-400 font-mono">${zoneStations.length} Active Stations</span>
            </div>
            <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 xl:grid-cols-6 gap-2.5">
        `;

        zoneStations.forEach(st => {
            const isSelected = st.station_id === selectedStationId;
            let statusClass = "station-nominal";
            let statusBadge = `<span class="text-[10px] text-emerald-400 font-mono font-bold"><i class="fa-solid fa-check"></i> Nominal</span>`;

            if (st.is_alert || st.defect_risk_pct > 50 || st.anomaly_score > 60) {
                statusClass = "station-alert glow-rose";
                statusBadge = `<span class="text-[10px] text-rose-400 font-mono font-bold animate-pulse"><i class="fa-solid fa-triangle-exclamation"></i> Alert (${st.defect_risk_pct}%)</span>`;
            } else if (st.anomaly_score > 35 || st.queue_len >= 3) {
                statusClass = "station-warning glow-amber";
                statusBadge = `<span class="text-[10px] text-amber-400 font-mono font-bold"><i class="fa-solid fa-clock"></i> Warning</span>`;
            }

            const proxyBadge = !st.sensor_rich
                ? `<span class="text-[9px] font-mono px-1.5 py-0.2 rounded bg-purple-950/80 text-purple-300 border border-purple-500/30" title="Proxy Inferenced from RFID Dwell & Power">Proxy: ${st.data_confidence.split(' ')[0]}</span>`
                : `<span class="text-[9px] font-mono px-1.5 py-0.2 rounded bg-blue-950/80 text-blue-300 border border-blue-500/30">Direct IoT</span>`;

            const queueBadge = st.queue_len > 0
                ? `<span class="px-1.5 py-0.2 rounded font-mono text-[10px] ${st.queue_len > 3 ? 'bg-rose-500/30 text-rose-300 border border-rose-500/50' : 'bg-slate-800 text-slate-300'}">Q:${st.queue_len}</span>`
                : '';

            html += `
            <div onclick="selectStation(${st.station_id})" class="station-node glass-card p-2.5 rounded-xl border ${statusClass} ${isSelected ? 'ring-2 ring-blue-500 shadow-lg shadow-blue-500/30' : ''}">
                <div class="flex items-center justify-between text-[10px] text-slate-400 mb-1">
                    <span class="font-mono font-bold text-slate-300">St ${st.station_id < 10 ? '0' + st.station_id : st.station_id}</span>
                    <div class="flex items-center gap-1">${queueBadge}${proxyBadge}</div>
                </div>
                <div class="text-xs font-bold text-white truncate" title="${st.name}">${st.name}</div>
                <div class="flex items-center justify-between text-[11px] font-mono mt-2 pt-1.5 border-t border-slate-800">
                    <span class="text-slate-300">${st.cycle_time}s</span>
                    ${statusBadge}
                </div>
            </div>
            `;
        });

        html += `
            </div>
        </div>
        `;
    });

    container.innerHTML = html;

    // Update Prescriptive AI Alert Card
    updateAlertCard(tickData);

    // Fetch and update selected station telemetry waveform
    fetchStationHistory(selectedStationId);
}

function selectStation(stationId) {
    selectedStationId = stationId;
    fetchStationHistory(stationId);
}

function updateAlertCard(tickData) {
    const alertCard = document.getElementById('aiActionCard');
    const alertTarget = document.getElementById('alertTargetBadge');
    const alertTitle = document.getElementById('alertTitle');
    const alertDesc = document.getElementById('alertDesc');
    const alertImpact = document.getElementById('alertImpact');
    const featureList = document.getElementById('featureImportanceList');

    if (!alertCard) return;

    if (tickData.active_alerts && tickData.active_alerts.length > 0) {
        currentActiveAlert = tickData.active_alerts[0];
        const rec = currentActiveAlert.recommendation;

        alertTarget.innerText = `Station ${currentActiveAlert.station_id < 10 ? '0' + currentActiveAlert.station_id : currentActiveAlert.station_id}`;
        alertTitle.innerText = rec.action_title;
        alertDesc.innerText = rec.details;
        alertImpact.innerText = `Impact: ${rec.impact}`;

        // Render Explainability Feature Importances
        if (featureList && currentActiveAlert.feature_importances) {
            let featHtml = '';
            const sortedFeats = Object.entries(currentActiveAlert.feature_importances).sort((a, b) => b[1] - a[1]).slice(0, 4);
            sortedFeats.forEach(([name, pct]) => {
                featHtml += `
                <div class="space-y-0.5">
                    <div class="flex justify-between text-[10px] text-slate-300 font-mono">
                        <span class="truncate max-w-[190px]">${name}</span>
                        <span class="text-purple-400 font-bold">${pct}%</span>
                    </div>
                    <div class="w-full bg-slate-800 h-1 rounded-full overflow-hidden">
                        <div class="bg-gradient-to-r from-blue-500 to-purple-500 h-full rounded-full" style="width: ${Math.min(100, pct * 1.8)}%;"></div>
                    </div>
                </div>
                `;
            });
            featureList.innerHTML = featHtml;
        }

        alertCard.classList.remove('hidden');
    } else {
        // Fallback nominal state
        alertTarget.innerText = `All Nominal`;
        alertTitle.innerText = `Assembly Flow Operating Within Takt`;
        alertDesc.innerText = `Zero active anomalies detected. DigitalTwin.ai continuously monitoring 36 stations.`;
        alertImpact.innerText = `Line Efficiency: 98.4% (Takt nominal: 60s)`;
        if (featureList) {
            featureList.innerHTML = `<div class="text-[10px] text-slate-500 italic py-1">All multivariate channels within 3-sigma bounds.</div>`;
        }
    }
}

async function fetchStationHistory(stId) {
    try {
        const resp = await fetch(`/api/stations/${stId}/history`);
        if (resp.ok) {
            const data = await resp.json();
            const stationName = `Station ${stId < 10 ? '0' + stId : stId}`;
            updateWaveformChart(data.data, stationName);
        }
    } catch (e) {
        console.error('Error fetching station history:', e);
    }
}

async function dispatchSupervisorAction() {
    if (!currentActiveAlert) {
        showToast("Action Logged", "Routine check acknowledged by supervisor.");
        return;
    }

    try {
        const payload = {
            station_id: currentActiveAlert.station_id,
            station_name: currentActiveAlert.station_name,
            action_type: currentActiveAlert.recommendation.action_type,
            description: currentActiveAlert.recommendation.action_title
        };

        const resp = await fetch('/api/action/execute', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (resp.ok) {
            showToast("Prescriptive Action Deployed to PLC", `Dispatched '${currentActiveAlert.recommendation.action_title}' to Siemens S7-1500 controller.`);
            dismissCurrentAlert();
        }
    } catch (e) {
        console.error("Action execution error:", e);
    }
}

function dismissCurrentAlert() {
    currentActiveAlert = null;
    const alertCard = document.getElementById('aiActionCard');
    if (alertCard) {
        const alertTarget = document.getElementById('alertTargetBadge');
        const alertTitle = document.getElementById('alertTitle');
        const alertDesc = document.getElementById('alertDesc');
        if (alertTarget) alertTarget.innerText = "Acknowledged";
        if (alertTitle) alertTitle.innerText = "Alert Acknowledged by Floor Supervisor";
        if (alertDesc) alertDesc.innerText = "Intervention queued for next scheduled buffer cycle.";
    }
}

function showToast(title, desc) {
    const toast = document.getElementById('plcToast');
    const toastDesc = document.getElementById('plcToastDesc');
    if (toast && toastDesc) {
        toastDesc.innerText = desc;
        toast.classList.add('show');
        setTimeout(() => {
            toast.classList.remove('show');
        }, 4000);
    }
}
