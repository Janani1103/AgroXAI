// Global Chart Instances
let shapSummaryChart, shapWaterfallChart;

// Section Management
function showSection(sectionId) {
    document.querySelectorAll('section').forEach(section => {
        section.classList.remove('active');
    });
    const target = document.getElementById(sectionId);
    if (target) {
        target.classList.add('active');
        
        // Reset multi-step form if entering input section
        if (sectionId === 'input') {
            nextStep(1);
        }

        // Update nav active state
        document.querySelectorAll('.nav-links a').forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('onclick') && link.getAttribute('onclick').includes(sectionId)) {
                link.classList.add('active');
            }
        });
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
}

// Multi-Step Form Logic
function nextStep(step) {
    const activeStep = document.querySelector('.form-step.active');
    const currentStepNum = activeStep ? parseInt(activeStep.id.split('-')[1]) : 1;

    // Only validate if moving FORWARD
    if (step > currentStepNum) {
        const inputs = activeStep.querySelectorAll('input, select');
        let valid = true;
        
        inputs.forEach(input => {
            if (input.hasAttribute('required') && !input.value) {
                valid = false;
                input.style.borderColor = '#ef4444';
            } else {
                input.style.borderColor = '#E0E4E0';
            }
        });

        if (!valid) {
            alert("Please complete all required fields in this step.");
            return;
        }
    }

    // Hide all steps
    document.querySelectorAll('.form-step').forEach(s => s.classList.remove('active'));
    // Show target step
    const target = document.getElementById(`step-${step}`);
    if (target) target.classList.add('active');
    
    // Update Indicators
    document.querySelectorAll('.step').forEach((s, idx) => {
        if (idx + 1 <= step) {
            s.classList.add('active');
        } else {
            s.classList.remove('active');
        }
    });
}

// XAI Tab Logic
function switchXaiTab(tabId) {
    // Update Buttons
    document.querySelectorAll('.xai-tab-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.getAttribute('onclick').includes(tabId)) {
            btn.classList.add('active');
        }
    });
    
    // Update Content
    document.querySelectorAll('.xai-tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(tabId).classList.add('active');
}

// Form Handling
document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Get form data
    const formData = {
        rainfall: document.getElementById('rainfall').value,
        temperature: document.getElementById('temperature').value,
        weatherCondition: document.getElementById('weatherCondition').value,
        daysToHarvest: document.getElementById('daysToHarvest').value,
        region: document.getElementById('region').value,
        soilType: document.getElementById('soilType').value,
        fertilizerUsed: document.getElementById('fertilizerUsed').value,
        irrigationUsed: document.getElementById('irrigationUsed').value
    };

    // Show loading state
    const btn = document.getElementById('predict-btn');
    const originalText = btn.innerHTML;
    btn.innerHTML = '<i class="fa-solid fa-seedling fa-spin"></i> Analyzing...';
    btn.disabled = true;

    try {
        const response = await fetch('http://localhost:5000/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        if (!response.ok) throw new Error('Backend error');
        
        const result = await response.json();
        if (result.success) {
            displayResults(result);
            showSection('xai');
        } else {
            alert("Prediction failed: " + result.error);
        }

    } catch (error) {
        console.warn('Backend not reached, using simulated prediction for demonstration.');
        showMockResults(formData);
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
});

function displayResults(data) {
    const yieldValue = data.prediction.toFixed(2);
    
    // Update numerical results in XAI view
    const resYield = document.getElementById('res-yield');
    if (resYield) resYield.innerText = yieldValue;
    
    const resConfidence = document.getElementById('res-confidence');
    if (resConfidence && data.confidence) {
        resConfidence.innerText = data.confidence.toFixed(1);
    }
    
    // Update numerical results in Input Preview (if visible)
    const previewValue = document.querySelector('#output-preview .prediction-value');
    if (previewValue) previewValue.innerHTML = `${yieldValue} <span style="font-size: 1.5rem;">tons/ha</span>`;
    
    // Show output preview on input page
    const preview = document.getElementById('output-preview');
    if (preview) preview.style.display = 'block';

    // Update XAI Text Interpretation
    const topFeatures = data.shap_values || [];
    const pos = topFeatures.filter(f => f.impact > 0).sort((a,b) => b.impact - a.impact).map(f => f.feature.replace('_', ' '));
    const neg = topFeatures.filter(f => f.impact < 0).sort((a,b) => a.impact - b.impact).map(f => f.feature.replace('_', ' '));
    
    let explanation = "The model predicts a harvest of " + yieldValue + " tons per hectare. ";
    if (pos.length > 0) {
        explanation += `The strongest growth drivers are <strong>${pos.slice(0, 2).join(' and ')}</strong>. `;
    }
    if (neg.length > 0) {
        explanation += `However, <strong>${neg.slice(0, 1).join(', ')}</strong> slightly reduced the potential yield.`;
    }
    
    const xaiText = document.getElementById('xai-text');
    if (xaiText) xaiText.innerHTML = explanation || "Factors are balanced for this prediction.";

    // Update Feature Impact Cards
    const featuresToTrack = [
        { key: 'Rainfall', id: 'impact-rainfall', card: 'card-rainfall' },
        { key: 'Temperature', id: 'impact-temperature', card: 'card-temp' },
        { key: 'Fertilizer', id: 'impact-fertilizer', card: 'card-fert' }
    ];

    featuresToTrack.forEach(item => {
        const feat = topFeatures.find(f => f.feature.includes(item.key));
        const impactEl = document.getElementById(item.id);
        const cardEl = document.getElementById(item.card);
        
        if (impactEl && feat) {
            impactEl.innerText = feat.impact > 0 ? '🔼 Positive' : '🔽 Negative';
            impactEl.style.color = feat.impact > 0 ? 'var(--primary-green)' : '#ef4444';
            if (cardEl) {
                cardEl.style.borderLeftColor = feat.impact > 0 ? 'var(--primary-green)' : '#ef4444';
            }
        }
    });

    // Initialize Charts
    initShapCharts(topFeatures, data.base_value || 5.2, data.prediction);
    
    // Reset to first tab
    switchXaiTab('xai-summary');
}

function showMockResults(formData) {
    const rainfall = parseFloat(formData.rainfall) || 800;
    const temp = parseFloat(formData.temperature) || 28;
    const fert = formData.fertilizerUsed === 'True';
    
    let prediction = 5.2; 
    let rImp = (rainfall - 700) / 800;
    let tImp = (temp > 24 && temp < 31) ? 0.4 : -0.3;
    let fImp = fert ? 0.6 : -0.2;
    
    prediction += (rImp + tImp + fImp);

    const mockData = {
        success: true,
        prediction: Math.max(1.5, prediction),
        confidence: 92.4,
        base_value: 5.2,
        shap_values: [
            { feature: "Rainfall_mm", impact: rImp },
            { feature: "Fertilizer_Used", impact: fImp },
            { feature: "Temperature_Celsius", impact: tImp },
            { feature: "Days_to_Harvest", impact: 0.1 }
        ]
    };
    
    displayResults(mockData);
    showSection('xai');
}

function initShapCharts(shapValues, baseValue, prediction) {
    if (shapSummaryChart) shapSummaryChart.destroy();
    if (shapWaterfallChart) shapWaterfallChart.destroy();

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false },
            tooltip: {
                backgroundColor: 'rgba(27, 27, 27, 0.9)',
                padding: 12,
                titleFont: { family: 'Outfit', size: 14 },
                bodyFont: { family: 'Inter', size: 13 }
            }
        },
        scales: {
            x: { grid: { display: false } },
            y: { grid: { color: 'rgba(0,0,0,0.05)' } }
        }
    };

    // Summary Chart
    const summaryCtx = document.getElementById('shapSummaryChart')?.getContext('2d');
    if (summaryCtx) {
        shapSummaryChart = new Chart(summaryCtx, {
            type: 'bar',
            data: {
                labels: shapValues.map(v => v.feature.replace('_', ' ')),
                datasets: [{
                    data: shapValues.map(v => v.impact),
                    backgroundColor: shapValues.map(v => v.impact >= 0 ? '#2E7D32' : '#ef4444'),
                    borderRadius: 8
                }]
            },
            options: {
                ...chartOptions,
                indexAxis: 'y'
            }
        });
    }

    // Waterfall Chart
    const waterfallCtx = document.getElementById('shapWaterfallChart')?.getContext('2d');
    if (waterfallCtx) {
        const labels = ['Average', ...shapValues.map(v => v.feature.split('_')[0]), 'Final'];
        const values = [baseValue, ...shapValues.map(v => v.impact), prediction];
        
        shapWaterfallChart = new Chart(waterfallCtx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    data: values,
                    backgroundColor: values.map((v, i) => {
                        if (i === 0 || i === labels.length - 1) return '#F4C430';
                        return v >= 0 ? '#2E7D32' : '#ef4444';
                    }),
                    borderRadius: 8
                }]
            },
            options: chartOptions
        });
    }
}
