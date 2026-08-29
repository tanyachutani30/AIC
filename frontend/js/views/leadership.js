/**
 * DigitalTwin.ai - Leadership (Business Case & ROI) View
 * Handles the "Rule of Ten" defect economics, realized financial returns,
 * and the interactive Global Enterprise Scaling Simulator.
 */

async function initLeadershipView() {
    try {
        const resp = await fetch('/api/metrics/roi');
        if (resp.ok) {
            const data = await resp.json();
            // Initial render with default parameters
            updateLeadershipCalculations();
        }
    } catch (e) {
        console.error("Error loading ROI metrics:", e);
    }
}

function updateLeadershipCalculations() {
    const plantsSlider = document.getElementById('scalePlantsSlider');
    const linesSlider = document.getElementById('scaleLinesSlider');
    const sensorSlider = document.getElementById('scaleSensorSlider');

    if (!plantsSlider || !linesSlider || !sensorSlider) return;

    const plants = parseInt(plantsSlider.value);
    const lines = parseInt(linesSlider.value);
    const sensorPct = parseInt(sensorSlider.value);

    // Update label text
    const pVal = document.getElementById('scalePlantsVal');
    const lVal = document.getElementById('scaleLinesVal');
    const sVal = document.getElementById('scaleSensorVal');

    if (pVal) pVal.innerText = `${plants} ${plants === 1 ? 'Plant' : 'Plants'}`;
    if (lVal) lVal.innerText = `${lines} ${lines === 1 ? 'Line' : 'Lines'}/Plant`;
    if (sVal) sVal.innerText = `${sensorPct}% (${sensorPct < 60 ? 'Sparse / Proxy Heavy' : sensorPct < 85 ? 'Mixed Brownfield' : 'Dense Gigafactory'})`;

    // Calculate economics
    // Base annual value per assembly line: ~$2,380,000
    // Sensor efficiency factor: even at 30% sensor coverage, our proxy inference captures 72% of value!
    const proxyFactor = 0.60 + 0.40 * (sensorPct / 100.0);
    const totalLines = plants * lines;
    const baseSavingsPerLine = 2380000;
    
    const totalAnnualSavings = Math.round(totalLines * baseSavingsPerLine * proxyFactor);
    const scrapSavings = Math.round(totalAnnualSavings * 0.27);
    const downtimeSavings = totalAnnualSavings - scrapSavings;

    const roiTotalEl = document.getElementById('roiTotalAnnual');
    const roiScrapEl = document.getElementById('roiScrapDollars');
    const roiDowntimeEl = document.getElementById('roiDowntimeDollars');

    if (roiTotalEl) roiTotalEl.innerText = `$${totalAnnualSavings.toLocaleString()}`;
    if (roiScrapEl) roiScrapEl.innerText = `$${scrapSavings.toLocaleString()}`;
    if (roiDowntimeEl) roiDowntimeEl.innerText = `$${downtimeSavings.toLocaleString()}`;
}
