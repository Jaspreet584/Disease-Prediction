let performanceMetrics = null;
let activeDisease = 'heart';
let activeModel = 'random_forest';
let charts = {};

// Circle Progress Ring Config
const circle = document.querySelector('.progress-ring-fill');
const radius = circle.r.baseVal.value;
const circumference = radius * 2 * Math.PI;
circle.style.strokeDasharray = `${circumference} ${circumference}`;
circle.style.strokeDashoffset = circumference;

function setProgress(percent) {
    const offset = circumference - (percent / 100 * circumference);
    circle.style.strokeDashoffset = offset;
}

// Disease Features Configurations
const FEATURE_SCHEMAS = {
    heart: [
        { id: 'age', label: 'Age', min: 29, max: 80, step: 1, val: 55, unit: 'years' },
        { id: 'sex', label: 'Gender', min: 0, max: 1, step: 1, val: 1, unit: '', labels: { 0: 'Female', 1: 'Male' } },
        { id: 'chest_pain_type', label: 'Chest Pain Type', min: 0, max: 3, step: 1, val: 1, unit: '', labels: { 0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-Anginal', 3: 'Asymptomatic' } },
        { id: 'resting_blood_pressure', label: 'Resting Blood Pressure', min: 90, max: 200, step: 1, val: 130, unit: 'mmHg' },
        { id: 'cholesterol', label: 'Serum Cholesterol', min: 120, max: 450, step: 1, val: 240, unit: 'mg/dl' },
        { id: 'fasting_blood_sugar', label: 'Fasting Blood Sugar', min: 0, max: 1, step: 1, val: 0, unit: '', labels: { 0: '<= 120 mg/dl', 1: '> 120 mg/dl' } },
        { id: 'resting_ecg', label: 'Resting ECG', min: 0, max: 2, step: 1, val: 1, unit: '', labels: { 0: 'Normal', 1: 'ST-T Wave Abnormality', 2: 'LV Hypertrophy' } },
        { id: 'max_heart_rate', label: 'Max Heart Rate', min: 70, max: 202, step: 1, val: 150, unit: 'bpm' },
        { id: 'exercise_angina', label: 'Exercise Induced Angina', min: 0, max: 1, step: 1, val: 0, unit: '', labels: { 0: 'No', 1: 'Yes' } },
        { id: 'st_depression', label: 'ST Depression (oldpeak)', min: 0.0, max: 6.2, step: 0.1, val: 1.0, unit: 'mm' }
    ],
    diabetes: [
        { id: 'pregnancies', label: 'Pregnancies', min: 0, max: 17, step: 1, val: 3, unit: 'times' },
        { id: 'glucose', label: 'Glucose Tolerance Test', min: 45, max: 200, step: 1, val: 115, unit: 'mg/dL' },
        { id: 'blood_pressure', label: 'Diastolic Blood Pressure', min: 38, max: 122, step: 1, val: 70, unit: 'mmHg' },
        { id: 'skin_thickness', label: 'Triceps Skin Fold Thickness', min: 0, max: 99, step: 1, val: 20, unit: 'mm' },
        { id: 'insulin', label: '2-Hour Serum Insulin', min: 0, max: 800, step: 5, val: 80, unit: 'mu U/ml' },
        { id: 'bmi', label: 'Body Mass Index (BMI)', min: 16.0, max: 67.0, step: 0.1, val: 31.2, unit: 'kg/m²' },
        { id: 'diabetes_pedigree_function', label: 'Diabetes Pedigree Function', min: 0.08, max: 2.42, step: 0.01, val: 0.47, unit: '' },
        { id: 'age', label: 'Age', min: 21, max: 81, step: 1, val: 33, unit: 'years' }
    ],
    breast_cancer: [
        { id: 'radius_mean', label: 'Cell Nucleus Radius', min: 6.9, max: 28.1, step: 0.1, val: 14.1, unit: 'mm' },
        { id: 'texture_mean', label: 'Texture (Gray-Scale SD)', min: 9.7, max: 39.2, step: 0.1, val: 19.3, unit: 'px' },
        { id: 'perimeter_mean', label: 'Nucleus Perimeter', min: 40.0, max: 190.0, step: 0.1, val: 92.0, unit: 'mm' },
        { id: 'area_mean', label: 'Nucleus Area', min: 140.0, max: 2500.0, step: 10, val: 650, unit: 'mm²' },
        { id: 'smoothness_mean', label: 'Smoothness (Local Rad Var)', min: 0.05, max: 0.16, step: 0.001, val: 0.096, unit: '' },
        { id: 'compactness_mean', label: 'Compactness', min: 0.01, max: 0.34, step: 0.001, val: 0.104, unit: '' },
        { id: 'concavity_mean', label: 'Concavity (Severity of Contours)', min: 0.0, max: 0.42, step: 0.001, val: 0.088, unit: '' },
        { id: 'concave_points_mean', label: 'Concave Points', min: 0.0, max: 0.20, step: 0.001, val: 0.048, unit: '' },
        { id: 'symmetry_mean', label: 'Symmetry', min: 0.1, max: 0.3, step: 0.001, val: 0.181, unit: '' },
        { id: 'fractal_dimension_mean', label: 'Fractal Dimension', min: 0.05, max: 0.097, step: 0.001, val: 0.062, unit: '' }
    ]
};

document.addEventListener('DOMContentLoaded', () => {
    setProgress(0);
    initApp();
    setupEventListeners();
});

async function initApp() {
    buildDynamicSliders();
    await fetchPerformance();
    triggerPrediction();
}

function setupEventListeners() {
    // 1. Target disease selection
    const diseaseSelect = document.getElementById('disease-select');
    diseaseSelect.addEventListener('change', async (e) => {
        activeDisease = e.target.value;
        buildDynamicSliders();
        await fetchPerformance();
        triggerPrediction();
        showToast(`Switched diagnostic target to ${e.target.options[e.target.selectedIndex].text}`);
    });

    // 2. Machine learning model selection
    const algoSelect = document.getElementById('algorithm-select');
    algoSelect.addEventListener('change', (e) => {
        activeModel = e.target.value;
        triggerPrediction();
        showToast(`Switched active classifier model to ${e.target.options[e.target.selectedIndex].text}`);
    });
}

// Render input range sliders dynamically based on active disease schema
function buildDynamicSliders() {
    const formGrid = document.getElementById('dynamic-form-grid');
    formGrid.innerHTML = ''; // Clear existing inputs
    
    const schemaList = FEATURE_SCHEMAS[activeDisease];
    
    schemaList.forEach(field => {
        const formGroup = document.createElement('div');
        formGroup.className = 'form-group';
        
        // Label containing display text value
        const label = document.createElement('label');
        label.setAttribute('for', `input-${field.id}`);
        
        const labelText = document.createTextNode(`${field.label}: `);
        const valueSpan = document.createElement('span');
        valueSpan.id = `val-${field.id}`;
        valueSpan.className = 'val-display';
        
        // Initial value text
        valueSpan.innerText = formatFieldValue(field, field.val);
        
        label.appendChild(labelText);
        label.appendChild(valueSpan);
        
        // Range Input element
        const input = document.createElement('input');
        input.type = 'range';
        input.id = `input-${field.id}`;
        input.min = field.min;
        input.max = field.max;
        input.step = field.step;
        input.value = field.val;
        input.className = 'slider';
        
        // Input Event Listener
        input.addEventListener('input', () => {
            const currentVal = parseFloat(input.value);
            valueSpan.innerText = formatFieldValue(field, currentVal);
            
            // Physical correlation rule for Breast Cancer radius/perimeter/area
            if (activeDisease === 'breast_cancer' && field.id === 'radius_mean') {
                syncBreastCancerRadius(currentVal);
            }
            
            // Execute predictive assessment on change
            triggerPrediction();
        });
        
        formGroup.appendChild(label);
        formGroup.appendChild(input);
        formGrid.appendChild(formGroup);
    });
}

// Synced updates to keep physical dimensions realistic in breast cancer dataset
function syncBreastCancerRadius(radiusValue) {
    const perimeterInput = document.getElementById('input-perimeter_mean');
    const areaInput = document.getElementById('input-area_mean');
    
    if (perimeterInput) {
        // P = 2 * pi * r
        const calculatedPerimeter = Math.round(2 * Math.PI * radiusValue * 10) / 10;
        perimeterInput.value = calculatedPerimeter;
        const display = document.getElementById('val-perimeter_mean');
        if (display) display.innerText = `${calculatedPerimeter} mm`;
    }
    
    if (areaInput) {
        // A = pi * r^2
        const calculatedArea = Math.round(Math.PI * (radiusValue ** 2) * 10) / 10;
        areaInput.value = calculatedArea;
        const display = document.getElementById('val-area_mean');
        if (display) display.innerText = `${calculatedArea} mm²`;
    }
}

// Help format numerical readouts with their respective labels or units
function formatFieldValue(field, value) {
    if (field.labels && field.labels[value] !== undefined) {
        return field.labels[value];
    }
    
    if (field.unit) {
        return `${value.toLocaleString()} ${field.unit}`;
    }
    
    // Decimal points formatting
    if (field.step < 1) {
        // Calculate decimal length from step size
        const decPlaces = (field.step.toString().split('.')[1] || '').length;
        return value.toFixed(decPlaces);
    }
    
    return value.toLocaleString();
}

// Fetch ML performance validation metrics from fastapi server
async function fetchPerformance() {
    try {
        const response = await fetch(`/api/performance/${activeDisease}`);
        if (!response.ok) throw new Error('Failed loading metrics');
        
        performanceMetrics = await response.json();
        renderCharts();
    } catch (err) {
        console.error(err);
        showToast('Error downloading machine learning metrics');
    }
}

// Draw performance curves and bar charts
function renderCharts() {
    const models = Object.keys(performanceMetrics);
    const labelMapping = {
        'svm': 'SVM Classifier',
        'logistic_regression': 'Logistic Reg.',
        'random_forest': 'Random Forest',
        'gradient_boosting': 'Grad. Boosting'
    };
    
    const displayLabels = models.map(m => labelMapping[m] || m);
    
    // --- 1. Bar Chart: Performance Metrics Comparison ---
    const ctxMetrics = document.getElementById('chart-metrics').getContext('2d');
    if (charts.metrics) charts.metrics.destroy();
    
    charts.metrics = new Chart(ctxMetrics, {
        type: 'bar',
        data: {
            labels: displayLabels,
            datasets: [
                { label: 'Accuracy', data: models.map(m => performanceMetrics[m].accuracy), backgroundColor: '#06b6d4' },
                { label: 'Precision', data: models.map(m => performanceMetrics[m].precision), backgroundColor: '#10b981' },
                { label: 'Recall', data: models.map(m => performanceMetrics[m].recall), backgroundColor: '#f59e0b' },
                { label: 'F1-Score', data: models.map(m => performanceMetrics[m].f1_score), backgroundColor: '#ef4444' },
                { label: 'ROC-AUC', data: models.map(m => performanceMetrics[m].roc_auc), backgroundColor: '#a855f7' }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#94a3b8', boxWidth: 10, font: { size: 9 } } }
            },
            scales: {
                x: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: 'rgba(6,182,212,0.03)' } },
                y: { min: 0.5, max: 1.0, ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: 'rgba(6,182,212,0.03)' } }
            }
        }
    });

    // --- 2. Scatter Plot: ROC Curves ---
    const ctxRoc = document.getElementById('chart-roc').getContext('2d');
    if (charts.roc) charts.roc.destroy();
    
    const colors = {
        'svm': '#06b6d4',
        'logistic_regression': '#10b981',
        'random_forest': '#f59e0b',
        'gradient_boosting': '#a855f7'
    };
    
    const datasets = models.map(m => {
        const points = performanceMetrics[m].roc_curve.map(p => ({ x: p.fpr, y: p.tpr }));
        return {
            label: labelMapping[m] || m,
            data: points,
            borderColor: colors[m] || '#fff',
            borderWidth: 2.5,
            showLine: true,
            fill: false,
            pointRadius: 0
        };
    });
    
    // Baseline diagonal curve
    datasets.push({
        label: 'Baseline (0.50)',
        data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
        borderColor: 'rgba(255, 255, 255, 0.12)',
        borderWidth: 1,
        borderDash: [5, 5],
        showLine: true,
        fill: false,
        pointRadius: 0
    });

    charts.roc = new Chart(ctxRoc, {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#94a3b8', boxWidth: 10, font: { size: 9 } } }
            },
            scales: {
                x: { type: 'linear', position: 'bottom', ticks: { color: '#94a3b8', font: { size: 9 } }, grid: { color: 'rgba(6,182,212,0.03)' }, min: 0, max: 1 },
                y: { ticks: { color: '#94a3b8', font: { size: 9 } }, grid: { color: 'rgba(6,182,212,0.03)' }, min: 0, max: 1 }
            }
        }
    });
}

// Request real-time risk diagnostic prediction from server
async function triggerPrediction() {
    try {
        const payload = { model_name: activeModel };
        
        // Assemble payload depending on current disease inputs
        const schemaList = FEATURE_SCHEMAS[activeDisease];
        schemaList.forEach(field => {
            const input = document.getElementById(`input-${field.id}`);
            if (input) {
                const val = parseFloat(input.value);
                payload[field.id] = val;
            }
        });
        
        // Endpoint mappings
        const endpoint = `/api/predict/${activeDisease}`;
        
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error('Prediction request failed');
        const res = await response.json();
        
        updateUI(res);
        
    } catch (err) {
        console.error(err);
    }
}

// Update SVG gauge and result panels
function updateUI(res) {
    const isPositive = res.status === 'Positive';
    const probPercentage = res.probability * 100;
    
    // Set probability rating
    document.getElementById('result-probability').innerText = `${probPercentage.toFixed(1)}%`;
    setProgress(probPercentage);
    
    // Dynamically adjust stroke coloring based on diagnostic severity
    if (res.risk_level === 'Low') {
        circle.style.stroke = 'var(--success-color)';
    } else if (res.risk_level === 'Moderate') {
        circle.style.stroke = 'var(--warning-color)';
    } else if (res.risk_level === 'High') {
        circle.style.stroke = '#f97316'; // Orange
    } else {
        circle.style.stroke = 'var(--danger-color)'; // Critical red
    }
    
    // Risk severity badge
    const badge = document.getElementById('result-severity-badge');
    badge.innerText = `${res.risk_level} RISK`;
    badge.className = `severity-badge ${res.risk_level}`;
    
    // Status metrics
    const diagnosticEl = document.getElementById('result-diagnostic');
    
    // Format status label depending on selected disease type
    if (activeDisease === 'breast_cancer') {
        diagnosticEl.innerText = isPositive ? 'MALIGNANT' : 'BENIGN';
    } else if (activeDisease === 'diabetes') {
        diagnosticEl.innerText = isPositive ? 'DIABETIC' : 'NON-DIABETIC';
    } else {
        diagnosticEl.innerText = isPositive ? 'CARDIAC RISK' : 'NORMAL';
    }
    
    diagnosticEl.className = isPositive ? 'text-danger' : 'text-success';
    
    // Severity text
    const severityText = document.getElementById('result-severity-text');
    severityText.innerText = `${res.risk_level} Assessment`;
    if (res.risk_level === 'Low') {
        severityText.className = 'text-success';
    } else if (res.risk_level === 'Moderate') {
        severityText.className = 'text-warning';
    } else {
        severityText.className = 'text-danger';
    }
}

// Toast alerts helper
function showToast(message) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.innerHTML = `<span>${message}</span>`;
    
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => { container.removeChild(toast); }, 300);
    }, 3000);
}
