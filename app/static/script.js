// DOM Elements
const form = document.getElementById('prediction-form');
const resultContent = document.getElementById('result-content');
const resultPlaceholder = document.getElementById('result-placeholder');
const predictionCount = document.getElementById('prediction-count');
const predictionTimestamp = document.getElementById('prediction-timestamp');
const confidenceBadge = document.getElementById('confidence-badge');
const responseTimeElement = document.getElementById('response-time');
const btnLoader = document.getElementById('btn-loader');
const predictBtn = document.getElementById('predict-btn');
const modelSelector = document.getElementById('model-selector');
const compareBtn = document.getElementById('compare-btn');

// Global variables
let selectedModel = 'original'; // Changed from 'random_forest' to 'original'
let availableModels = {};
let currentPredictionData = null;

// Show loading state
function showLoading() {
    if (btnLoader) {
        btnLoader.style.display = 'flex';
    }
    if (predictBtn) {
        predictBtn.disabled = true;
        predictBtn.style.opacity = '0.7';
    }
}

// Hide loading state
function hideLoading() {
    if (btnLoader) {
        btnLoader.style.display = 'none';
    }
    if (predictBtn) {
        predictBtn.disabled = false;
        predictBtn.style.opacity = '1';
    }
}

// Show notification
function showNotification(message, type = 'info') {
    // Remove existing notifications
    document.querySelectorAll('.notification').forEach(n => n.remove());
    
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-${type === 'success' ? 'check-circle' : 
                              type === 'error' ? 'exclamation-circle' : 
                              type === 'warning' ? 'exclamation-triangle' : 
                              'info-circle'}"></i>
            <span>${message}</span>
        </div>
        <button class="notification-close" onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => {
                if (notification.parentElement) {
                    notification.remove();
                }
            }, 300);
        }
    }, 5000);
}

// Format model name for display
function formatModelName(modelName) {
    const names = {
        'original': 'Random Forest (Original)',
        'improved': 'Enhanced Random Forest',
        'random_forest': 'Random Forest',
        'xgboost': 'XGBoost',
        'lightgbm': 'LightGBM',
        'gradient_boosting': 'Gradient Boosting'
    };
    return names[modelName] || modelName.replace('_', ' ').toUpperCase();
}

// Get season encoding for form submission
function getSeasonEncoding(season) {
    const encoding = {};
    const seasons = ['Spring', 'Summer', 'Autumn', 'Winter'];
    
    seasons.forEach(s => {
        encoding[`Seasons_${s}`] = s === season ? 1 : 0;
    });
    
    return encoding;
}

// Validate form inputs
function validateForm(formData) {
    const errors = [];
    
    // Validate hour
    const hour = parseInt(formData['Hour']);
    if (isNaN(hour) || hour < 0 || hour > 23) {
        errors.push('Hour must be between 0 and 23');
    }
    
    // Validate temperature - FIXED: Use Korean character field name
    const temp = parseFloat(formData['Temperature(캜)']);
    if (isNaN(temp) || temp < -20 || temp > 40) {
        errors.push('Temperature must be between -20°C and 40°C');
    }
    
    // Validate humidity
    const humidity = parseFloat(formData['Humidity(%)']);
    if (isNaN(humidity) || humidity < 0 || humidity > 100) {
        errors.push('Humidity must be between 0% and 100%');
    }
    
    return errors;
}

// Update prediction results display
function updatePredictionResults(data, responseTime) {
    // Hide placeholder, show results
    if (resultPlaceholder) {
        resultPlaceholder.style.display = 'none';
    }
    if (resultContent) {
        resultContent.style.display = 'block';
    }
    
    // Update prediction count
    const prediction = data.predicted_rented_bikes || data.rounded || Math.round(data.prediction);
    if (predictionCount) {
        predictionCount.textContent = prediction;
        
        // Animate the count
        animateCounter(predictionCount, 0, prediction, 1000);
    }
    
    // Update details
    if (document.getElementById('detail-model')) {
        document.getElementById('detail-model').textContent = formatModelName(data.model_type);
    }
    if (document.getElementById('detail-features')) {
        document.getElementById('detail-features').textContent = data.model_info?.features_used || '14';
    }
    if (responseTimeElement) {
        responseTimeElement.textContent = `${responseTime}s`;
    }
    if (document.getElementById('prediction-id')) {
        document.getElementById('prediction-id').textContent = data.prediction_id || '--';
    }
    
    // Update timestamp
    const timestamp = new Date(data.timestamp || new Date());
    if (predictionTimestamp) {
        predictionTimestamp.textContent = timestamp.toLocaleTimeString([], { 
            hour: '2-digit', 
            minute: '2-digit' 
        });
    }
    
    // Update confidence
    if (confidenceBadge) {
        confidenceBadge.textContent = `${data.confidence || 80}% confidence`;
    }
    
    // Update model badge
    const modelTypeBadge = document.getElementById('model-type-badge');
    if (modelTypeBadge) {
        modelTypeBadge.textContent = formatModelName(data.model_type);
        modelTypeBadge.className = `model-badge ${data.model_type.replace('_', '-')}`;
    }
    
    // Update model metrics if available
    if (data.model_metrics) {
        updateModelMetrics(data.model_metrics);
    }
    
    // Store current prediction data
    currentPredictionData = data;
}

// Animate counter
function animateCounter(element, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const value = Math.floor(progress * (end - start) + start);
        element.textContent = value;
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

// Update model metrics display
function updateModelMetrics(metrics) {
    const metricElements = {
        'metric-r2': document.getElementById('metric-r2'),
        'metric-rmse': document.getElementById('metric-rmse'),
        'metric-mae': document.getElementById('metric-mae'),
        'metric-gap': document.getElementById('metric-gap')
    };
    
    for (const [id, element] of Object.entries(metricElements)) {
        if (element && metrics[id.replace('metric-', '')]) {
            const value = metrics[id.replace('metric-', '')];
            element.textContent = typeof value === 'number' ? value.toFixed(2) : value;
        }
    }
}

// Select model function
function selectModel(modelName) {
    selectedModel = modelName;
    
    // Update model selector dropdown
    if (modelSelector) {
        modelSelector.value = modelName;
    }
    
    // Update model option indicators
    document.querySelectorAll('.model-option').forEach(option => {
        const indicator = option.querySelector('.selection-indicator');
        if (option.getAttribute('data-model') === modelName) {
            indicator.classList.add('active');
        } else {
            indicator.classList.remove('active');
        }
    });
    
    // Update UI
    updateModelUI();
    
    // Show notification
    showNotification(`Selected model: ${formatModelName(modelName)}`, 'success');
}

// Update UI based on selected model
function updateModelUI() {
    const modelName = formatModelName(selectedModel);
    
    // Update predict button color
    const colors = {
        'original': 'linear-gradient(135deg, #4361ee, #3a56d4)',
        'improved': 'linear-gradient(135deg, #7209b7, #3a0ca3)',
        'random_forest': 'linear-gradient(135deg, #4361ee, #3a56d4)',
        'xgboost': 'linear-gradient(135deg, #f72585, #b5179e)',
        'lightgbm': 'linear-gradient(135deg, #4cc9f0, #4895ef)',
        'gradient_boosting': 'linear-gradient(135deg, #2a9d8f, #1d7873)'
    };
    
    if (predictBtn) {
        predictBtn.style.background = colors[selectedModel] || colors.original;
    }
    
    // Update model indicator
    const modelIndicator = document.getElementById('model-indicator');
    if (modelIndicator) {
        modelIndicator.textContent = modelName;
        modelIndicator.className = `model-indicator ${selectedModel.replace('_', '-')}`;
    }
    
    // Update insight
    const insightModel = document.getElementById('insight-model');
    if (insightModel) {
        insightModel.textContent = modelName;
    }
    
    // Update model badge in results
    const modelTypeBadge = document.getElementById('model-type-badge');
    if (modelTypeBadge) {
        modelTypeBadge.textContent = modelName;
        modelTypeBadge.className = `model-badge ${selectedModel.replace('_', '-')}`;
    }
    
    // Load model metrics if available
    if (availableModels[selectedModel]) {
        updateModelMetrics(availableModels[selectedModel]);
    }
}

// Load available models from API
async function loadAvailableModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        
        if (data.success && data.models) {
            availableModels = data.models;
            
            // Update models available count
            const availableCount = Object.values(availableModels).filter(m => m.available).length;
            const modelsAvailableElement = document.getElementById('models-available');
            if (modelsAvailableElement) {
                modelsAvailableElement.textContent = availableCount;
            }
            
            // Update current model metrics
            if (availableModels[selectedModel]) {
                updateModelMetrics(availableModels[selectedModel]);
                
                // Update sidebar
                const currentModelName = document.getElementById('current-model-name');
                const currentModelR2 = document.getElementById('current-model-r2');
                
                if (currentModelName) {
                    currentModelName.textContent = formatModelName(selectedModel);
                }
                if (currentModelR2) {
                    currentModelR2.textContent = availableModels[selectedModel].r2?.toFixed(3) || 'N/A';
                }
            }
            
            // Update model status
            updateModelStatus();
            
            return true;
        }
    } catch (error) {
        console.error('Error loading models:', error);
    }
    return false;
}

// Update model status
function updateModelStatus() {
    const statusText = document.getElementById('model-status-text');
    const statusIndicator = document.querySelector('.model-status-badge .status-indicator');
    
    if (availableModels[selectedModel]?.available) {
        if (statusText) {
            statusText.textContent = 'Model Loaded';
        }
        if (statusIndicator) {
            statusIndicator.className = 'status-indicator active';
        }
    } else {
        if (statusText) {
            statusText.textContent = 'Model Not Available';
        }
        if (statusIndicator) {
            statusIndicator.className = 'status-indicator';
        }
    }
}

// Form submission handler - FIXED: Added debug logging
if (form) {
    console.log('✅ Form found, adding event listener');
    
    form.addEventListener('submit', async function(event) {
        event.preventDefault();
        console.log('🎯 Form submitted!');
        
        const startTime = performance.now();
        
        // Show loading state
        showLoading();
        
        try {
            // Collect form data
            const formData = {};
            const formElements = this.elements;
            
            // Collect all form data
            for (let element of formElements) {
                if (element.name && element.value !== undefined && element.value !== '') {
                    // Handle number inputs
                    if (element.type === 'number') {
                        formData[element.name] = parseFloat(element.value);
                    } 
                    // Handle checkbox toggles
                    else if (element.type === 'checkbox') {
                        // For checkboxes, use 1 for checked, 0 for unchecked
                        formData[element.name] = element.checked ? 1 : 0;
                    }
                    // Handle all other inputs
                    else {
                        formData[element.name] = element.value;
                    }
                }
            }
            
            // Add season encoding
            const season = document.getElementById('seasons')?.value || 'Summer';
            const seasonEncoding = getSeasonEncoding(season);
            Object.assign(formData, seasonEncoding);
            
            console.log('📤 Form data to send:', formData);
            
            // Validate inputs
            const validationErrors = validateForm(formData);
            if (validationErrors.length > 0) {
                showNotification(validationErrors.join(', '), 'error');
                hideLoading();
                return;
            }
            
            // Make prediction request with selected model
            const url = `/predict?model_type=${selectedModel}`;
            console.log(`📡 Making request to: ${url}`);
            
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });
            
            const responseTime = ((performance.now() - startTime) / 1000).toFixed(2);
            const data = await response.json();
            
            console.log('📥 Response received:', data);
            
            hideLoading();
            
            if (data.success) {
                // Update UI with results
                updatePredictionResults(data, responseTime);
                
                // Update predictions today count
                updatePredictionsCount();
                
                showNotification('Prediction generated successfully!', 'success');
            } else {
                showNotification(`Prediction failed: ${data.error || 'Unknown error'}`, 'error');
                console.error('Prediction error details:', data);
            }
            
        } catch (error) {
            hideLoading();
            showNotification(`Network error: ${error.message}`, 'error');
            console.error('Fetch error details:', error);
        }
    });
} else {
    console.error('❌ Form not found! Check if element with id="prediction-form" exists');
}

// Update predictions count
async function updatePredictionsCount() {
    try {
        const response = await fetch('/metrics');
        const data = await response.json();
        
        if (data.success) {
            const predictionsTodayElement = document.getElementById('predictions-today');
            if (predictionsTodayElement) {
                predictionsTodayElement.textContent = data.metrics.predictions_today || 0;
            }
        }
    } catch (error) {
        console.error('Error updating predictions count:', error);
    }
}

// Compare all models - FIXED: Use correct endpoint
async function compareAllModels() {
    try {
        const response = await fetch('/predict/compare', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                Hour: 14,
                Temperature: 22.5,
                Humidity: 60
            })
        });
        const data = await response.json();
        
        if (data.success && data.results) {
            // Update comparison table
            if (data.results.comparison) {
                showNotification(`Models compared: Original vs Improved`, 'info');
            }
        } else {
            showNotification('Could not compare models', 'error');
        }
    } catch (error) {
        console.error('Error comparing models:', error);
        showNotification('Could not compare models', 'error');
    }
}

// Quick compare function
async function compareModels() {
    if (!form || !form.checkValidity()) {
        showNotification('Please fill all required fields first', 'warning');
        return;
    }
    
    showLoading();
    
    try {
        // Collect form data
        const formData = {};
        const formElements = form.elements;
        
        for (let element of formElements) {
            if (element.name && element.value !== undefined && element.value !== '') {
                if (element.type === 'number') {
                    formData[element.name] = parseFloat(element.value);
                } else {
                    formData[element.name] = element.value;
                }
            }
        }
        
        // Add season encoding
        const season = document.getElementById('seasons')?.value || 'Summer';
        const seasonEncoding = getSeasonEncoding(season);
        Object.assign(formData, seasonEncoding);
        
        const response = await fetch('/predict/compare', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        
        hideLoading();
        
        if (data.success && data.results) {
            showComparisonResults(data.results);
        } else {
            showNotification('Comparison failed', 'error');
        }
        
    } catch (error) {
        hideLoading();
        showNotification(`Network error: ${error.message}`, 'error');
    }
}

// Show comparison results modal
function showComparisonResults(results) {
    const modal = document.createElement('div');
    modal.className = 'comparison-modal';
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.8);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 1000;
    `;
    
    let html = `
        <div class="comparison-content" style="
            background: white;
            padding: 30px;
            border-radius: 15px;
            max-width: 800px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
        ">
            <div class="comparison-header" style="
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
            ">
                <h2 style="margin: 0; color: #2c3e50;">
                    <i class="fas fa-balance-scale"></i> Model Comparison Results
                </h2>
                <button onclick="this.parentElement.parentElement.parentElement.remove()" style="
                    background: none;
                    border: none;
                    font-size: 24px;
                    cursor: pointer;
                    color: #666;
                ">&times;</button>
            </div>
            <div class="comparison-table">
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="background: #f8f9fa;">
                            <th style="padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6;">Model</th>
                            <th style="padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6;">Prediction</th>
                            <th style="padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6;">Confidence</th>
                            <th style="padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6;">Status</th>
                        </tr>
                    </thead>
                    <tbody>
    `;
    
    // Show original model result
    if (results.original) {
        const isCurrent = 'original' === selectedModel;
        html += `
            <tr style="${isCurrent ? 'background: #e8f4fd;' : ''}">
                <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">
                    <strong>${formatModelName('original')}</strong>
                    ${isCurrent ? '<span style="color: #4361ee; font-size: 12px; margin-left: 8px;">(Current)</span>' : ''}
                </td>
                <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">
                    ${results.original.prediction ? Math.round(results.original.prediction) : '--'} bikes
                </td>
                <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">
                    ${results.original.confidence ? results.original.confidence + '%' : '--'}
                </td>
                <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">
                    <span style="
                        padding: 4px 8px;
                        border-radius: 4px;
                        font-size: 12px;
                        background: #d4edda;
                        color: #155724;
                    ">
                        Available
                    </span>
                </td>
            </tr>
        `;
    }
    
    // Show improved model result
    if (results.improved) {
        const isCurrent = 'improved' === selectedModel;
        const status = results.improved.error ? 'Error' : 'Available';
        html += `
            <tr style="${isCurrent ? 'background: #e8f4fd;' : ''}">
                <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">
                    <strong>${formatModelName('improved')}</strong>
                    ${isCurrent ? '<span style="color: #4361ee; font-size: 12px; margin-left: 8px;">(Current)</span>' : ''}
                </td>
                <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">
                    ${results.improved.prediction ? Math.round(results.improved.prediction) : '--'} bikes
                </td>
                <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">
                    ${results.improved.confidence ? results.improved.confidence + '%' : '--'}
                </td>
                <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">
                    <span style="
                        padding: 4px 8px;
                        border-radius: 4px;
                        font-size: 12px;
                        background: ${status === 'Available' ? '#d4edda' : '#f8d7da'};
                        color: ${status === 'Available' ? '#155724' : '#721c24'};
                    ">
                        ${status}
                    </span>
                </td>
            </tr>
        `;
    }
    
    if (results.comparison) {
        html += `
            <tr style="background: #f8f9fa;">
                <td colspan="4" style="padding: 15px; border-top: 2px solid #dee2e6;">
                    <strong>Comparison:</strong> 
                    Difference: ${results.comparison.difference.toFixed(0)} bikes 
                    (${results.comparison.difference_percent.toFixed(1)}%)
                </td>
            </tr>
        `;
    }
    
    html += `
                    </tbody>
                </table>
            </div>
            <div style="margin-top: 20px; text-align: center;">
                <button onclick="selectModel('improved')" style="
                    background: #7209b7;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 8px;
                    cursor: pointer;
                    font-weight: 600;
                    margin-right: 10px;
                ">
                    <i class="fas fa-rocket"></i> Switch to Improved Model
                </button>
                <button onclick="this.parentElement.parentElement.parentElement.remove()" style="
                    background: #6c757d;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 8px;
                    cursor: pointer;
                    font-weight: 600;
                ">
                    Close
                </button>
            </div>
        </div>
    `;
    
    modal.innerHTML = html;
    document.body.appendChild(modal);
    
    // Close modal when clicking outside
    modal.addEventListener('click', function(e) {
        if (e.target === modal) {
            modal.remove();
        }
    });
}

// Model Manager functions - SIMPLIFIED for your setup
function openModelManager() {
    // Create a simple model manager modal
    const modal = document.createElement('div');
    modal.id = 'simple-model-manager';
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.8);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 1000;
    `;
    
    modal.innerHTML = `
        <div style="
            background: white;
            padding: 30px;
            border-radius: 15px;
            max-width: 600px;
            width: 90%;
        ">
            <h2 style="margin-top: 0; color: #2c3e50;">
                <i class="fas fa-cogs"></i> Available Models
            </h2>
            <div style="margin: 20px 0;">
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 10px;">
                    <h3 style="margin-top: 0;">Original Model</h3>
                    <p><strong>Features:</strong> 14</p>
                    <p><strong>R² Score:</strong> 0.820</p>
                    <button onclick="selectModel('original'); document.getElementById('simple-model-manager').remove()" 
                            style="background: #4361ee; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;">
                        ${selectedModel === 'original' ? '✓ Current' : 'Use This Model'}
                    </button>
                </div>
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px;">
                    <h3 style="margin-top: 0;">Improved Model</h3>
                    <p><strong>Features:</strong> 35+</p>
                    <p><strong>Status:</strong> ${availableModels.improved?.available ? 'Available' : 'Not trained'}
                    </p>
                    <button onclick="selectModel('improved'); document.getElementById('simple-model-manager').remove()" 
                            ${!availableModels.improved?.available ? 'disabled' : ''}
                            style="background: #7209b7; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; ${!availableModels.improved?.available ? 'opacity: 0.5;' : ''}">
                        ${selectedModel === 'improved' ? '✓ Current' : 'Use This Model'}
                    </button>
                </div>
            </div>
            <div style="text-align: center; margin-top: 20px;">
                <button onclick="document.getElementById('simple-model-manager').remove()" 
                        style="background: #6c757d; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer;">
                    Close
                </button>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // Close modal when clicking outside
    modal.addEventListener('click', function(e) {
        if (e.target === modal) {
            modal.remove();
        }
    });
}

function closeModelManager() {
    const modal = document.getElementById('simple-model-manager');
    if (modal) {
        modal.remove();
    }
}

// Switch model function - SIMPLIFIED
async function switchModel(modelName) {
    if (!availableModels[modelName]?.available) {
        showNotification(`Model ${formatModelName(modelName)} is not available`, 'error');
        return;
    }
    
    selectedModel = modelName;
    selectModel(modelName);
    showNotification(`Switched to ${formatModelName(modelName)} model`, 'success');
    
    // Update current model info
    await loadAvailableModels();
    
    // Close modal if open
    closeModelManager();
}

// Show current model info
function showCurrentModelInfo() {
    const modelInfo = availableModels[selectedModel];
    if (modelInfo) {
        let infoText = `Model: ${formatModelName(selectedModel)}\n`;
        infoText += `Status: ${modelInfo.available ? 'Available' : 'Not Available'}\n`;
        if (modelInfo.r2) infoText += `R² Score: ${modelInfo.r2.toFixed(3)}\n`;
        if (modelInfo.rmse) infoText += `RMSE: ${modelInfo.rmse}\n`;
        if (modelInfo.features) infoText += `Features: ${modelInfo.features}`;
        
        alert(infoText);
    } else {
        showNotification('Model information not available', 'error');
    }
}

// Show training info
function showTrainingInfo() {
    alert('To train the improved model, run:\n\npython train_model_fixed.py\n\nThis will train the enhanced Random Forest model with additional features.');
}

// Refresh models
async function refreshModels() {
    const success = await loadAvailableModels();
    if (success) {
        showNotification('Models refreshed successfully', 'success');
    } else {
        showNotification('Failed to refresh models', 'error');
    }
}

// Initialize page - FIXED: Added debug logging
async function initPage() {
    console.log('🚀 Initializing page...');
    
    // Add debug button click listener as backup
    if (predictBtn) {
        console.log('✅ Predict button found, adding click listener');
        predictBtn.addEventListener('click', function(e) {
            console.log('🎯 Predict button clicked (direct listener)');
            if (form) {
                form.dispatchEvent(new Event('submit'));
            }
        });
    } else {
        console.error('❌ Predict button not found! Check HTML for id="predict-btn"');
    }
    
    // Initialize tabs if they exist
    document.querySelectorAll('.tab-btn').forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.getAttribute('data-tab');
            
            // Update active tab button
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            // Show corresponding tab content
            document.querySelectorAll('.form-tab').forEach(tab => {
                tab.classList.remove('active');
            });
            const tabElement = document.getElementById(`${tabName}-tab`);
            if (tabElement) {
                tabElement.classList.add('active');
            }
        });
    });
    
    // Initialize season selector
    document.querySelectorAll('.season-option').forEach(option => {
        option.addEventListener('click', () => {
            document.querySelectorAll('.season-option').forEach(opt => opt.classList.remove('active'));
            option.classList.add('active');
            const seasonsInput = document.getElementById('seasons');
            if (seasonsInput) {
                seasonsInput.value = option.getAttribute('data-value');
            }
        });
    });
    
    // Initialize hour slider sync
    const hourInput = document.getElementById('hour');
    const hourSlider = document.getElementById('hour-slider');
    
    if (hourInput && hourSlider) {
        hourInput.addEventListener('input', () => {
            hourSlider.value = hourInput.value;
        });
        
        hourSlider.addEventListener('input', () => {
            hourInput.value = hourSlider.value;
        });
    }
    
    // Initialize temperature slider sync
    const tempInput = document.getElementById('temperature');
    const tempSlider = document.getElementById('temp-slider');
    
    if (tempInput && tempSlider) {
        tempInput.addEventListener('input', () => {
            tempSlider.value = tempInput.value;
        });
        
        tempSlider.addEventListener('input', () => {
            tempInput.value = tempSlider.value;
        });
    }
    
    // Initialize humidity indicator
    const humidityInput = document.getElementById('humidity');
    if (humidityInput) {
        humidityInput.addEventListener('input', () => {
            const humidityBar = document.querySelector('.humidity-fill');
            if (humidityBar) {
                humidityBar.style.width = `${humidityInput.value}%`;
            }
        });
    }
    
    // Initialize model selector
    if (modelSelector) {
        modelSelector.addEventListener('change', function() {
            selectModel(this.value);
        });
    }
    
    // Set default model selection
    selectModel('original');
    
    // Load available models
    await loadAvailableModels();
    
    // Update predictions count
    await updatePredictionsCount();
    
    // Test API connection
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        console.log('API Status:', data);
    } catch (error) {
        console.warn('Could not connect to API:', error);
    }
    
    console.log('✅ Page initialization complete');
}

// Load sample data - FIXED: Use correct field names
function loadSampleData() {
    const samples = [
        { 
            hour: 8, 
            'Temperature(캜)': 18.5, // FIXED: Korean characters
            'Humidity(%)': 70, 
            'Wind speed (m/s)': 1.5, 
            'Visibility (10m)': 1800, 
            'Rainfall(mm)': 0.0, 
            'Snowfall (cm)': 0.0,
            'Solar Radiation (MJ/m2)': 0.8, 
            'Dew point temperature(캜)': 12.0,
            season: 'Spring'
        },
        { 
            hour: 14, 
            'Temperature(캜)': 22.5, 
            'Humidity(%)': 60, 
            'Wind speed (m/s)': 2.5, 
            'Visibility (10m)': 2000, 
            'Rainfall(mm)': 0.0, 
            'Snowfall (cm)': 0.0,
            'Solar Radiation (MJ/m2)': 1.5, 
            'Dew point temperature(캜)': 15.0,
            season: 'Summer'
        }
    ];
    
    const sample = samples[Math.floor(Math.random() * samples.length)];
    
    // Update form fields with correct IDs
    document.getElementById('hour').value = sample.hour;
    if (document.getElementById('hour-slider')) {
        document.getElementById('hour-slider').value = sample.hour;
    }
    
    document.getElementById('temperature').value = sample['Temperature(캜)'];
    if (document.getElementById('temp-slider')) {
        document.getElementById('temp-slider').value = sample['Temperature(캜)'];
    }
    
    document.getElementById('humidity').value = sample['Humidity(%)'];
    document.getElementById('windspeed').value = sample['Wind speed (m/s)'];
    document.getElementById('visibility').value = sample['Visibility (10m)'];
    document.getElementById('rainfall').value = sample['Rainfall(mm)'];
    document.getElementById('snowfall').value = sample['Snowfall (cm)'];
    document.getElementById('solar').value = sample['Solar Radiation (MJ/m2)'];
    document.getElementById('dewpoint').value = sample['Dew point temperature(캜)'];
    
    // Update season
    document.querySelectorAll('.season-option').forEach(opt => {
        opt.classList.remove('active');
        if (opt.getAttribute('data-value') === sample.season) {
            opt.classList.add('active');
            const seasonsInput = document.getElementById('seasons');
            if (seasonsInput) {
                seasonsInput.value = sample.season;
            }
        }
    });
    
    // Update humidity bar
    const humidityBar = document.querySelector('.humidity-fill');
    if (humidityBar) {
        humidityBar.style.width = `${sample['Humidity(%)']}%`;
    }
    
    showNotification('Sample data loaded successfully', 'success');
}

// Clear form
function clearForm() {
    if (form) {
        form.reset();
        document.getElementById('hour').value = 14;
        if (document.getElementById('hour-slider')) {
            document.getElementById('hour-slider').value = 14;
        }
        document.getElementById('temperature').value = 22.5;
        if (document.getElementById('temp-slider')) {
            document.getElementById('temp-slider').value = 22.5;
        }
        document.getElementById('humidity').value = 60;
        document.getElementById('windspeed').value = 2.5;
        document.getElementById('visibility').value = 2000;
        document.getElementById('rainfall').value = 0.0;
        document.getElementById('snowfall').value = 0.0;
        document.getElementById('solar').value = 1.5;
        document.getElementById('dewpoint').value = 10.0;
        
        const humidityBar = document.querySelector('.humidity-fill');
        if (humidityBar) {
            humidityBar.style.width = '60%';
        }
        
        document.querySelectorAll('.season-option').forEach(opt => {
            opt.classList.remove('active');
            if (opt.getAttribute('data-value') === 'Summer') {
                opt.classList.add('active');
                const seasonsInput = document.getElementById('seasons');
                if (seasonsInput) {
                    seasonsInput.value = 'Summer';
                }
            }
        });
        
        showNotification('Form cleared', 'info');
    }
}

// Copy prediction to clipboard
function copyPrediction() {
    if (currentPredictionData) {
        const prediction = currentPredictionData.predicted_rented_bikes || 
                         currentPredictionData.rounded || 
                         Math.round(currentPredictionData.prediction);
        const model = formatModelName(currentPredictionData.model_type);
        const text = `Predicted bike demand: ${prediction} bikes (using ${model} model)`;
        
        navigator.clipboard.writeText(text).then(() => {
            showNotification('Prediction copied to clipboard', 'success');
        }).catch(err => {
            showNotification('Failed to copy to clipboard', 'error');
        });
    } else {
        showNotification('No prediction to copy', 'error');
    }
}

// Export prediction
function exportPrediction() {
    showNotification('Export available in Analytics dashboard', 'info');
    setTimeout(() => {
        window.location.href = '/analytics';
    }, 1500);
}

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('simple-model-manager');
    if (modal && event.target === modal) {
        closeModelManager();
    }
}

// Initialize when page loads
window.addEventListener('DOMContentLoaded', function() {
    console.log('📄 DOM Content Loaded, initializing...');
    initPage().catch(error => {
        console.error('Error during initialization:', error);
    });
});

// Export functions for global access
window.loadSampleData = loadSampleData;
window.clearForm = clearForm;
window.copyPrediction = copyPrediction;
window.exportPrediction = exportPrediction;
window.compareModels = compareModels;
window.compareAllModels = compareAllModels;
window.openModelManager = openModelManager;
window.closeModelManager = closeModelManager;
window.switchModel = switchModel;
window.refreshModels = refreshModels;
window.showCurrentModelInfo = showCurrentModelInfo;
window.showTrainingInfo = showTrainingInfo;
window.selectModel = selectModel;
window.showNotification = showNotification;

console.log('✅ script.js loaded successfully');