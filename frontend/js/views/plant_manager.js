/**
 * DigitalTwin.ai - Plant Manager (Planning & Trends) View
 * Displays rolling validation metrics, tunable threshold trade-offs,
 * SPC vs ML benchmark comparisons, and root-cause Pareto analytics.
 */

let validationReportCache = null;

async function initPlantManagerView() {
    try {
        const resp = await fetch('/api/metrics/validation');
        if (resp.ok) {
            const data = await resp.json();
            renderValidationMetrics(data.metrics);
            initTradeoffChart(data.tradeoff_curve);
        }

        const reportResp = await fetch('/api/reports/evaluation');
        if (reportResp.ok) {
            validationReportCache = await reportResp.json();
            if (validationReportCache.models_summary && validationReportCache.models_summary.random_forest_defect_classifier) {
                initParetoChart(validationReportCache.models_summary.random_forest_defect_classifier.feature_importances);
            }
        }
    } catch (e) {
        console.error("Error initializing Plant Manager view:", e);
    }
}

function renderValidationMetrics(metricsData) {
    if (!metricsData) return;

    const def = metricsData.defect_classifier;
    const pEl = document.getElementById('pmPrecision');
    const rEl = document.getElementById('pmRecall');
    const fEl = document.getElementById('pmFar');

    if (pEl && def) pEl.innerText = `${(def.precision * 100).toFixed(1)}%`;
    if (rEl && def) rEl.innerText = `${(def.recall * 100).toFixed(1)}%`;
    if (fEl && def) fEl.innerText = `${(def.false_alarm_rate * 100).toFixed(1)}%`;
}

async function updateConfidenceThreshold(newVal) {
    const textEl = document.getElementById('thresholdValueText');
    if (textEl) textEl.innerText = parseFloat(newVal).toFixed(2);

    try {
        const resp = await fetch('/api/threshold', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ threshold: parseFloat(newVal) })
        });

        if (resp.ok) {
            const mResp = await fetch(`/api/metrics/validation?threshold=${newVal}`);
            if (mResp.ok) {
                const data = await mResp.json();
                renderValidationMetrics(data.metrics);
            }
        }
    } catch (e) {
        console.error("Error updating threshold:", e);
    }
}
