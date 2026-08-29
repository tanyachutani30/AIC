/**
 * DigitalTwin.ai - Chart.js Component Visualizations
 * Handles telemetry waveforms, threshold tradeoff curves, and root-cause Pareto charts.
 */

let waveformChart = null;
let tradeoffChart = null;
let paretoChart = null;

function initWaveformChart() {
    const ctx = document.getElementById('stationWaveformChart');
    if (!ctx) return;

    waveformChart = new Chart(ctx.getContext('2d'), {
        type: 'line',
        data: {
            labels: Array.from({ length: 20 }, (_, i) => `-${20 - i}t`),
            datasets: [
                {
                    label: 'Cycle Time (s)',
                    data: Array(20).fill(58.0),
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 2,
                    tension: 0.3,
                    fill: false,
                    pointRadius: 0
                },
                {
                    label: 'Torque / Power Proxy',
                    data: Array(20).fill(50.0),
                    borderColor: '#a855f7',
                    borderWidth: 2,
                    tension: 0.3,
                    fill: false,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                legend: {
                    display: true,
                    labels: { color: '#94a3b8', font: { size: 10, family: 'Inter' }, boxWidth: 10 }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#64748b', font: { size: 9, family: 'JetBrains Mono' } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#64748b', font: { size: 9, family: 'JetBrains Mono' } }
                }
            }
        }
    });
}

function updateWaveformChart(historyRecords, stationName) {
    if (!waveformChart || !historyRecords || historyRecords.length === 0) return;

    const labels = historyRecords.map((_, i) => `-${historyRecords.length - i}t`);
    const ctData = historyRecords.map(r => r.cycle_time);
    const secondaryData = historyRecords.map(r => r.torque_nm !== null && r.torque_nm !== undefined ? r.torque_nm : (r.power_kw ? r.power_kw * 15.0 : 50.0));

    waveformChart.data.labels = labels;
    waveformChart.data.datasets[0].data = ctData;
    waveformChart.data.datasets[1].data = secondaryData;
    waveformChart.data.datasets[1].label = historyRecords[0].torque_nm !== null && historyRecords[0].torque_nm !== undefined ? 'Torque (Nm)' : 'Power Proxy (kW x15)';
    waveformChart.update();

    const titleEl = document.getElementById('waveformStationTitle');
    if (titleEl) titleEl.innerText = `${stationName} Telemetry Waveform`;

    const summaryEl = document.getElementById('waveformSummaryMetric');
    const latest = historyRecords[historyRecords.length - 1];
    if (summaryEl && latest) {
        if (latest.torque_nm !== null && latest.torque_nm !== undefined) {
            summaryEl.innerText = `CT: ${latest.cycle_time}s | Torque: ${latest.torque_nm} Nm | Vib: ${latest.vibration_rms || 1.2}g`;
        } else {
            summaryEl.innerText = `CT: ${latest.cycle_time}s | RFID Dwell: ${latest.rfid_dwell_time_sec}s | Power: ${latest.power_kw} kW`;
        }
    }
}

function initTradeoffChart(curveData) {
    const ctx = document.getElementById('tradeoffChart');
    if (!ctx) return;

    if (tradeoffChart) {
        tradeoffChart.destroy();
    }

    const defaultCurve = curveData && curveData.length > 0 ? curveData : [
        { threshold: 0.1, precision: 0.45, recall: 1.0, false_alarm_rate: 0.12 },
        { threshold: 0.3, precision: 0.68, recall: 0.98, false_alarm_rate: 0.04 },
        { threshold: 0.5, precision: 0.81, recall: 1.0, false_alarm_rate: 0.007 },
        { threshold: 0.7, precision: 0.92, recall: 0.88, false_alarm_rate: 0.002 },
        { threshold: 0.9, precision: 0.98, recall: 0.72, false_alarm_rate: 0.000 }
    ];

    const labels = defaultCurve.map(d => d.threshold.toFixed(2));
    const prec = defaultCurve.map(d => (d.precision * 100).toFixed(1));
    const rec = defaultCurve.map(d => (d.recall * 100).toFixed(1));
    const far = defaultCurve.map(d => (d.false_alarm_rate * 100).toFixed(1));

    tradeoffChart = new Chart(ctx.getContext('2d'), {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Precision (%)',
                    data: prec,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 2.5,
                    tension: 0.3,
                    fill: false
                },
                {
                    label: 'Recall (%)',
                    data: rec,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 2.5,
                    tension: 0.3,
                    fill: false
                },
                {
                    label: 'False Alarm Rate (%)',
                    data: far,
                    borderColor: '#f43f5e',
                    backgroundColor: 'rgba(244, 63, 94, 0.1)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    tension: 0.3,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    labels: { color: '#94a3b8', font: { size: 11, family: 'Inter' } }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Alert Confidence Threshold', color: '#64748b', font: { size: 10 } },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8', font: { size: 10, family: 'JetBrains Mono' } }
                },
                y: {
                    title: { display: true, text: 'Metric %', color: '#64748b', font: { size: 10 } },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8', font: { size: 10, family: 'JetBrains Mono' } },
                    min: 0,
                    max: 105
                }
            }
        }
    });
}

function initParetoChart(importances) {
    const ctx = document.getElementById('paretoChart');
    if (!ctx) return;

    if (paretoChart) {
        paretoChart.destroy();
    }

    const defaultImp = importances && Object.keys(importances).length > 0 ? importances : {
        "Anomaly Score (IForest)": 49.2,
        "Power Draw Proxy": 18.1,
        "Tool Operating Age": 9.7,
        "RFID Dwell Deviation": 9.2,
        "Instantaneous CT": 7.8,
        "Vibration Level": 5.6
    };

    const labels = Object.keys(defaultImp);
    const values = Object.values(defaultImp);

    paretoChart = new Chart(ctx.getContext('2d'), {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Relative Contribution (%)',
                data: values,
                backgroundColor: [
                    'rgba(59, 130, 246, 0.8)',
                    'rgba(168, 85, 247, 0.8)',
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(245, 158, 11, 0.8)',
                    'rgba(244, 63, 94, 0.8)',
                    'rgba(14, 165, 233, 0.8)'
                ],
                borderRadius: 6
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8', font: { size: 10, family: 'JetBrains Mono' } }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#cbd5e1', font: { size: 10, family: 'Inter' } }
                }
            }
        }
    });
}
