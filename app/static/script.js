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

// Global variables
let selectedModel = 'original';
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
        'original': 'Standard Model',
        'improved': 'Feature-Engineered',
        'advanced': 'Advanced RF',
        'xgboost': 'XGBoost'
    };
    return names[modelName] || modelName;
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
    
    // Validate temperature
    const temp = parseFloat(formData['Temperature(캜)'] || formData['Temperature']);
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
    const prediction = data.predicted_rented_bikes || data.rounded || Math.round(data.prediction || 0);
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
        document.getElementById('detail-features').textContent = data.input_summary?.features_used || '14';
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
        modelTypeBadge.className = `model-badge ${data.model_type || 'original'}`;
    }
    
    // Store current prediction data
    currentPredictionData = data;
    
    // Update model performance metrics if available in response
    if (data.model_metrics) {
        updateModelPerformanceMetrics(data.model_metrics);
    } else {
        // Otherwise fetch fresh metrics
        fetchModelPerformanceMetrics();
    }
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

// Select model function
function selectModel(modelName) {
    selectedModel = modelName;
    
    // Update model option indicators
    document.querySelectorAll('.model-option').forEach(option => {
        const indicator = option.querySelector('.selection-indicator');
        if (option.getAttribute('data-model') === modelName) {
            indicator.classList.add('active');
            option.classList.add('active');
        } else {
            indicator.classList.remove('active');
            option.classList.remove('active');
        }
    });
    
    // Update UI
    updateModelUI();
    
    // Refresh metrics to show the selected model's performance
    fetchModelPerformanceMetrics();
    
    showNotification(`${formatModelName(modelName)} selected`, 'info');
}

// Update UI based on selected model
function updateModelUI() {
    const modelName = formatModelName(selectedModel);
    
    // Update predict button color
    const colors = {
        'original': 'linear-gradient(135deg, #4361ee, #3a56d4)',
        'improved': 'linear-gradient(135deg, #7209b7, #3a0ca3)',
        'advanced': 'linear-gradient(135deg, #06d6a0, #1b9e5c)',
        'xgboost': 'linear-gradient(135deg, #f72585, #b5179e)'
    };
    
    if (predictBtn) {
        predictBtn.style.background = colors[selectedModel] || colors.original;
    }
    
    // Update model indicator
    const modelIndicator = document.getElementById('model-indicator');
    if (modelIndicator) {
        modelIndicator.textContent = modelName;
        modelIndicator.className = `model-indicator ${selectedModel}`;
    }
    
    // Update model badge in results
    const modelTypeBadge = document.getElementById('model-type-badge');
    if (modelTypeBadge) {
        modelTypeBadge.textContent = modelName;
        modelTypeBadge.className = `model-badge ${selectedModel}`;
    }
}

// Update model performance metrics
function updateModelPerformanceMetrics(metrics) {
    console.log('📊 ========== METRICS UPDATE START ==========');
    console.log('📊 Raw metrics received:', metrics);
    console.log('📊 Selected model:', selectedModel);
    
    // Check if metrics exists
    if (!metrics) {
        console.log('❌ No metrics data available');
        return;
    }
    
    // Log all available keys in metrics
    console.log('📊 Available keys in metrics:', Object.keys(metrics));
    
    // Determine which model's metrics to show based on selected model
    let r2Value = null;
    let rmseValue = null;
    let maeValue = null;
    
    if (selectedModel === 'advanced' && metrics.advanced_model_metrics && Object.keys(metrics.advanced_model_metrics).length > 0) {
        // Show advanced model metrics
        r2Value = metrics.advanced_model_metrics.test_r2;
        rmseValue = metrics.advanced_model_metrics.test_rmse;
        maeValue = metrics.advanced_model_metrics.test_mae;
        console.log('📊 Using advanced model metrics:', {r2Value, rmseValue, maeValue});
    } else if (selectedModel === 'xgboost' && metrics.xgboost_model_metrics && Object.keys(metrics.xgboost_model_metrics).length > 0) {
        // Show XGBoost metrics if available
        r2Value = metrics.xgboost_model_metrics.test_r2;
        rmseValue = metrics.xgboost_model_metrics.test_rmse;
        maeValue = metrics.xgboost_model_metrics.test_mae;
        console.log('📊 Using XGBoost model metrics:', {r2Value, rmseValue, maeValue});
    } else if (selectedModel === 'improved' && metrics.improved_model_metrics) {
        // Show improved model metrics
        r2Value = metrics.improved_model_metrics.r2 || 0.899;
        rmseValue = metrics.improved_model_metrics.rmse || 194.59;
        maeValue = metrics.improved_model_metrics.mae || 150.2;
        console.log('📊 Using improved model metrics:', {r2Value, rmseValue, maeValue});
    } else {
        // Show base/original model metrics
        r2Value = metrics.r2_score;
        rmseValue = metrics.root_mean_squared_error;
        maeValue = metrics.mean_absolute_error;
        
        console.log('📊 Using base model metrics from API:', {r2Value, rmseValue, maeValue});
    }
    
    // Get DOM elements
    const r2Element = document.getElementById('metric-r2');
    const rmseElement = document.getElementById('metric-rmse');
    const maeElement = document.getElementById('metric-mae');
    const gapElement = document.getElementById('metric-gap');
    
    console.log('📊 DOM Elements found:', {
        r2: !!r2Element,
        rmse: !!rmseElement,
        mae: !!maeElement,
        gap: !!gapElement
    });
    
    // Update R² Score
    if (r2Element) {
        const oldValue = r2Element.textContent;
        if (r2Value !== null && r2Value !== undefined) {
            r2Element.textContent = typeof r2Value === 'number' ? r2Value.toFixed(3) : r2Value;
            console.log(`✅ R² updated from "${oldValue}" to "${r2Element.textContent}"`);
        } else {
            console.warn('⚠️ R² value is null/undefined, keeping existing value:', oldValue);
        }
    } else {
        console.error('❌ R² element not found! ID: metric-r2');
    }
    
    // Update RMSE
    if (rmseElement) {
        const oldValue = rmseElement.textContent;
        if (rmseValue !== null && rmseValue !== undefined) {
            rmseElement.textContent = typeof rmseValue === 'number' ? rmseValue.toFixed(1) : rmseValue;
            console.log(`✅ RMSE updated from "${oldValue}" to "${rmseElement.textContent}"`);
        } else {
            console.warn('⚠️ RMSE value is null/undefined, keeping existing value:', oldValue);
        }
    } else {
        console.error('❌ RMSE element not found! ID: metric-rmse');
    }
    
    // Update MAE
    if (maeElement) {
        const oldValue = maeElement.textContent;
        if (maeValue !== null && maeValue !== undefined) {
            maeElement.textContent = typeof maeValue === 'number' ? maeValue.toFixed(1) : maeValue;
            console.log(`✅ MAE updated from "${oldValue}" to "${maeElement.textContent}"`);
        } else {
            console.warn('⚠️ MAE value is null/undefined, keeping existing value:', oldValue);
        }
    } else {
        console.error('❌ MAE element not found! ID: metric-mae');
    }
    
    // Update Accuracy Gap (1 - R²)
    if (gapElement) {
        const oldValue = gapElement.textContent;
        if (r2Value !== null && r2Value !== undefined) {
            const gap = ((1 - r2Value) * 100).toFixed(1) + '%';
            gapElement.textContent = gap;
            console.log(`✅ Accuracy Gap updated from "${oldValue}" to "${gap}"`);
        } else {
            console.warn('⚠️ Cannot calculate gap - R² is null/undefined, keeping existing value:', oldValue);
        }
    } else {
        console.error('❌ Gap element not found! ID: metric-gap');
    }
    
    console.log('📊 ========== METRICS UPDATE END ==========');
}

// Fetch model performance metrics from API
async function fetchModelPerformanceMetrics() {
    console.log('📡 ========== FETCHING METRICS ==========');
    
    try {
        console.log('📡 Making request to /metrics...');
        const response = await fetch('/metrics');
        console.log('📡 Response status:', response.status);
        
        if (response.ok) {
            const data = await response.json();
            console.log('📡 RAW API RESPONSE:', data);
            
            if (data.metrics) {
                console.log('📡 Metrics object found');
                updateModelPerformanceMetrics(data.metrics);
                return data.metrics;
            } else if (data.success && data.metrics) {
                console.log('📡 Metrics in success wrapper');
                updateModelPerformanceMetrics(data.metrics);
                return data.metrics;
            } else {
                console.warn('📡 No metrics found in response');
            }
        }
    } catch (error) {
        console.error('❌ Error fetching model performance metrics:', error);
    }
    
    console.log('📡 ========== FETCH COMPLETE ==========');
    return null;
}

// Form submission handler
if (form) {
    console.log('✅ Form found, adding event listener');
    
    form.addEventListener('submit', async function(event) {
        event.preventDefault();
        console.log('🎯 Form submitted!');
        
        const startTime = performance.now();
        
        // Show loading state
        showLoading();
        
        try {
            // Collect form data with CORRECT field names matching the backend
            const formData = {};
            
            // Get values from the form with the correct field names from HTML
            formData['Hour'] = parseInt(document.getElementById('Hour')?.value || 14);
            formData['Temperature(캜)'] = parseFloat(document.getElementById('Temperature(캜)')?.value || 22.5);
            formData['Humidity(%)'] = parseFloat(document.getElementById('Humidity(%)')?.value || 60);
            formData['Wind speed (m/s)'] = parseFloat(document.getElementById('Wind speed (m/s)')?.value || 2.5);
            formData['Visibility (10m)'] = parseInt(document.getElementById('Visibility (10m)')?.value || 2000);
            formData['Rainfall(mm)'] = parseFloat(document.getElementById('Rainfall(mm)')?.value || 0);
            formData['Snowfall (cm)'] = parseFloat(document.getElementById('Snowfall (cm)')?.value || 0);
            formData['Solar Radiation (MJ/m2)'] = parseFloat(document.getElementById('Solar Radiation (MJ/m2)')?.value || 1.5);
            formData['Dew point temperature(캜)'] = parseFloat(document.getElementById('Dew point temperature(캜)')?.value || 10);
            
            // Get season from active season option
            const activeSeason = document.querySelector('.season-option.active')?.getAttribute('data-value') || 'Summer';
            const seasonEncoding = getSeasonEncoding(activeSeason);
            Object.assign(formData, seasonEncoding);
            
            // Get holiday and functioning day values from hidden inputs
            formData['Holiday_No Holiday'] = parseInt(document.getElementById('Holiday_No Holiday')?.value || 1);
            formData['Holiday_Holiday'] = parseInt(document.getElementById('Holiday_Holiday')?.value || 0);
            formData['Functioning Day_Yes'] = parseInt(document.getElementById('Functioning Day_Yes')?.value || 1);
            formData['Functioning Day_No'] = parseInt(document.getElementById('Functioning Day_No')?.value || 0);
            
            // Add day of week if available
            if (document.getElementById('day_of_week')) {
                formData['day_of_week'] = parseInt(document.getElementById('day_of_week')?.value || 0);
            }
            
            console.log('📤 Form data to send:', formData);
            
            // Validate inputs
            const validationErrors = validateForm(formData);
            if (validationErrors.length > 0) {
                showNotification(validationErrors.join(', '), 'error');
                hideLoading();
                return;
            }
            
            // Make prediction request with selected model
            let url = `/predict?model_type=${selectedModel}`;
            if (selectedModel === 'advanced' || selectedModel === 'xgboost') {
                url += '&calculate_features=true';
                console.log(`🧮 Auto-calculation enabled for ${selectedModel} model`);
            }else if (selectedModel === 'improved') {
                console.log(`📊 Improved model will calculate all 38 features on backend`);
            }
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
                updateDashboardMetrics();
                
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

// Update dashboard metrics
async function updateDashboardMetrics() {
    console.log('📊 updateDashboardMetrics called');
    
    try {
        // Get elements by their specific IDs
        const modelAccuracyEl = document.getElementById('model-accuracy');
        const predictionsTodayEl = document.getElementById('predictions-today');
        const avgErrorEl = document.getElementById('avg-error');
        const modelsAvailableEl = document.getElementById('models-available');
        
        console.log('Found elements:', {
            modelAccuracy: !!modelAccuracyEl,
            predictionsToday: !!predictionsTodayEl,
            avgError: !!avgErrorEl,
            modelsAvailable: !!modelsAvailableEl
        });
        
        // Show loading states
        document.querySelectorAll('.metric-loading').forEach(el => {
            el.style.display = 'inline';
        });
        document.querySelectorAll('.metric-value').forEach(el => {
            el.style.display = 'none';
        });
        
        console.log('📡 Fetching from /api/dashboard-metrics...');
        const response = await fetch('/api/dashboard-metrics');
        console.log('Response status:', response.status);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Dashboard metrics data received:', data);
        
        if (data.success && data.metrics) {
            console.log('✅ Updating metrics with:', data.metrics);
            
            // Update Model Accuracy
            if (modelAccuracyEl) {
                const loadingEl = modelAccuracyEl.querySelector('.metric-loading');
                const valueEl = modelAccuracyEl.querySelector('.metric-value');
                if (loadingEl && valueEl) {
                    loadingEl.style.display = 'none';
                    valueEl.style.display = 'inline';
                    valueEl.textContent = data.metrics.model_accuracy + '%';
                    console.log('✅ Updated model accuracy to:', data.metrics.model_accuracy + '%');
                }
            }
            
            // Update Predictions Today
            if (predictionsTodayEl) {
                const loadingEl = predictionsTodayEl.querySelector('.metric-loading');
                const valueEl = predictionsTodayEl.querySelector('.metric-value');
                if (loadingEl && valueEl) {
                    loadingEl.style.display = 'none';
                    valueEl.style.display = 'inline';
                    valueEl.textContent = data.metrics.predictions_today;
                    console.log('✅ Updated predictions today to:', data.metrics.predictions_today);
                }
            }
            
            // Update Avg Error
            if (avgErrorEl) {
                const loadingEl = avgErrorEl.querySelector('.metric-loading');
                const valueEl = avgErrorEl.querySelector('.metric-value');
                if (loadingEl && valueEl) {
                    loadingEl.style.display = 'none';
                    valueEl.style.display = 'inline';
                    valueEl.textContent = '±' + data.metrics.avg_error + ' bikes';
                    console.log('✅ Updated avg error to:', '±' + data.metrics.avg_error + ' bikes');
                }
            }
            
            // Update Models Available
            if (modelsAvailableEl) {
                const loadingEl = modelsAvailableEl.querySelector('.metric-loading');
                const valueEl = modelsAvailableEl.querySelector('.metric-value');
                if (loadingEl && valueEl) {
                    loadingEl.style.display = 'none';
                    valueEl.style.display = 'inline';
                    valueEl.textContent = data.metrics.models_available;
                    console.log('✅ Updated models available to:', data.metrics.models_available);
                }
            }
        } else {
            console.warn('⚠️ Dashboard metrics API returned unsuccessful response');
        }
    } catch (error) {
        console.error('❌ Error updating dashboard metrics:', error);
        // Show fallback values
        document.querySelectorAll('.metric-loading').forEach(el => {
            el.style.display = 'none';
        });
        document.querySelectorAll('.metric-value').forEach(el => {
            el.style.display = 'inline';
        });
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
                const valueEl = modelsAvailableElement.querySelector('.metric-value');
                if (valueEl) {
                    valueEl.textContent = availableCount;
                }
            }
            
            return true;
        }
    } catch (error) {
        console.error('Error loading models:', error);
    }
    return false;
}

// Initialize page
async function initPage() {
    console.log('🚀 Initializing page...');
    
    // Set default model selection
    selectModel('original');
    
    // Load available models
    await loadAvailableModels();
    
    // Fetch model performance metrics
    await fetchModelPerformanceMetrics();
    
    // Update dashboard metrics
    console.log('📊 Calling updateDashboardMetrics...');
    await updateDashboardMetrics();
    
    // Set up periodic refresh (every 60 seconds)
    setInterval(updateDashboardMetrics, 60000);
    setInterval(fetchModelPerformanceMetrics, 60000);
    
    console.log('✅ Page initialization complete');
}

// Initialize when DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('📄 DOM Content Loaded, initializing...');
    initPage().catch(error => {
        console.error('Error during initialization:', error);
    });
});

// Model Manager function
function openModelManager() {
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
            max-height: 80vh;
            overflow-y: auto;
        ">
            <h2 style="margin-top: 0; color: #2c3e50;">
                <i class="fas fa-cogs"></i> Available Models
            </h2>
            <div style="margin: 20px 0;">
                <!-- Standard Model -->
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #4361ee;">
                    <h3 style="margin-top: 0; color: #4361ee;">Standard Model</h3>
                    <p><strong>Features:</strong> 14 basic features</p>
                    <p><strong>R² Score:</strong> 0.820</p>
                    <p><strong>RMSE:</strong> 112.5</p>
                    <p><strong>Use Case:</strong> Quick predictions with basic weather data</p>
                    <button onclick="selectModel('original'); document.getElementById('simple-model-manager').remove()" 
                            style="background: #4361ee; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;">
                        ${selectedModel === 'original' ? '✓ Current' : 'Use This Model'}
                    </button>
                </div>
                
                <!-- Improved Model -->
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #7209b7;">
                    <h3 style="margin-top: 0; color: #7209b7;">Improved Model</h3>
                    <p><strong>Features:</strong> 38 engineered features</p>
                    <p><strong>R² Score:</strong> 0.899</p>
                    <p><strong>RMSE:</strong> 194.59</p>
                    <p><strong>Key Features:</strong> hour², peak hours, weather severity, comfort index</p>
                    <button onclick="selectModel('improved'); document.getElementById('simple-model-manager').remove()" 
                            style="background: #7209b7; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;">
                        ${selectedModel === 'improved' ? '✓ Current' : 'Use This Model'}
                    </button>
                </div>
                
                <!-- Advanced Model -->
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #06d6a0;">
                    <h3 style="margin-top: 0; color: #06d6a0;">Advanced Model</h3>
                    <p><strong>Features:</strong> 100+ advanced features</p>
                    <p><strong>R² Score:</strong> 0.92</p>
                    <p><strong>RMSE:</strong> ≤ 80</p>
                    <p><strong>Key Features:</strong> hour_sin/cos, weather categories, interactions</p>
                    <button onclick="selectModel('advanced'); document.getElementById('simple-model-manager').remove()" 
                            style="background: #06d6a0; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;">
                        ${selectedModel === 'advanced' ? '✓ Current' : 'Use This Model'}
                    </button>
                </div>
                
                <!-- XGBoost Model -->
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-top: 10px; border-left: 4px solid #f72585;">
                    <h3 style="margin-top: 0; color: #f72585;">XGBoost Model</h3>
                    <p><strong>Features:</strong> 100+ advanced features</p>
                    <p><strong>R² Score:</strong> 0.94</p>
                    <p><strong>RMSE:</strong> 153</p>
                    <p><strong>MAE:</strong> 91</p>
                    <p><strong>Algorithm:</strong> Gradient Boosting</p>
                    <button onclick="selectModel('xgboost'); document.getElementById('simple-model-manager').remove()" 
                            style="background: #f72585; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;">
                        ${selectedModel === 'xgboost' ? '✓ Current' : 'Use This Model'}
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
    
    modal.addEventListener('click', function(e) {
        if (e.target === modal) {
            modal.remove();
        }
    });
}

// Close modal
function closeModelManager() {
    const modal = document.getElementById('simple-model-manager');
    if (modal) {
        modal.remove();
    }
}

// Load sample data - ENHANCED VERSION
function loadSampleData() {
    const samples = [
        { 
            Hour: 8, 
            'Temperature(캜)': 18.5,
            'Humidity(%)': 70, 
            'Wind speed (m/s)': 1.5, 
            'Visibility (10m)': 1800, 
            'Rainfall(mm)': 0.0, 
            'Snowfall (cm)': 0.0,
            'Solar Radiation (MJ/m2)': 0.8, 
            'Dew point temperature(캜)': 12.0,
            season: 'Spring',
            holiday: false,
            functioning: true,
            day_of_week: 1,
            description: 'Morning commute - Spring conditions'
        },
        { 
            Hour: 14, 
            'Temperature(캜)': 22.5, 
            'Humidity(%)': 60, 
            'Wind speed (m/s)': 2.5, 
            'Visibility (10m)': 2000, 
            'Rainfall(mm)': 0.0, 
            'Snowfall (cm)': 0.0,
            'Solar Radiation (MJ/m2)': 1.5, 
            'Dew point temperature(캜)': 15.0,
            season: 'Summer',
            holiday: false,
            functioning: true,
            day_of_week: 3,
            description: 'Afternoon - Optimal summer conditions'
        },
        { 
            Hour: 17, 
            'Temperature(캜)': 20.0, 
            'Humidity(%)': 65, 
            'Wind speed (m/s)': 3.0, 
            'Visibility (10m)': 1900, 
            'Rainfall(mm)': 0.0, 
            'Snowfall (cm)': 0.0,
            'Solar Radiation (MJ/m2)': 0.5, 
            'Dew point temperature(캜)': 14.0,
            season: 'Autumn',
            holiday: false,
            functioning: true,
            day_of_week: 4,
            description: 'Evening commute - Autumn conditions'
        },
        { 
            Hour: 12, 
            'Temperature(캜)': 5.0, 
            'Humidity(%)': 45, 
            'Wind speed (m/s)': 5.5, 
            'Visibility (10m)': 1500, 
            'Rainfall(mm)': 0.0, 
            'Snowfall (cm)': 2.5,
            'Solar Radiation (MJ/m2)': 0.3, 
            'Dew point temperature(캜)': -1.0,
            season: 'Winter',
            holiday: true,
            functioning: true,
            day_of_week: 6,
            description: 'Snowy winter day - Holiday'
        },
        { 
            Hour: 9, 
            'Temperature(캜)': 15.0, 
            'Humidity(%)': 85, 
            'Wind speed (m/s)': 2.0, 
            'Visibility (10m)': 800, 
            'Rainfall(mm)': 5.5, 
            'Snowfall (cm)': 0.0,
            'Solar Radiation (MJ/m2)': 0.2, 
            'Dew point temperature(캜)': 12.5,
            season: 'Spring',
            holiday: false,
            functioning: true,
            day_of_week: 0,
            description: 'Rainy morning - Reduced visibility'
        }
    ];
    
    const sample = samples[Math.floor(Math.random() * samples.length)];
    console.log('📋 Loading sample data:', sample);
    
    try {
        // Update numeric fields with validation
        updateFieldWithValidation('Hour', sample.Hour, 0, 23);
        updateFieldWithValidation('Temperature(캜)', sample['Temperature(캜)'], -20, 40);
        updateFieldWithValidation('Humidity(%)', sample['Humidity(%)'], 0, 100);
        updateFieldWithValidation('Wind speed (m/s)', sample['Wind speed (m/s)'], 0, 20);
        updateFieldWithValidation('Visibility (10m)', sample['Visibility (10m)'], 0, 5000);
        updateFieldWithValidation('Rainfall(mm)', sample['Rainfall(mm)'], 0, 100);
        updateFieldWithValidation('Snowfall (cm)', sample['Snowfall (cm)'], 0, 50);
        updateFieldWithValidation('Solar Radiation (MJ/m2)', sample['Solar Radiation (MJ/m2)'], 0, 5);
        updateFieldWithValidation('Dew point temperature(캜)', sample['Dew point temperature(캜)'], -30, 30);
        
        // Sync sliders
        syncSlider('hour-slider', 'Hour', sample.Hour);
        syncSlider('temp-slider', 'Temperature(캜)', sample['Temperature(캜)']);
        
        // Update season selector
        updateSeasonSelector(sample.season);
        
        // Update holiday and functioning day toggles
        updateToggleSwitches(sample.holiday, sample.functioning);
        
        // Update humidity bar
        updateHumidityBar(sample['Humidity(%)']);
        
        // Update hour description
        if (typeof updateHourDescription === 'function') {
            updateHourDescription(sample.Hour);
        } else {
            window.updateHourDescription(sample.Hour);
        }
        
        // Update weather insight
        if (typeof updateWeatherInsight === 'function') {
            updateWeatherInsight(sample['Temperature(캜)']);
        } else {
            window.updateWeatherInsight(sample['Temperature(캜)']);
        }
        
        // Update day of week dropdown
        const dayOfWeekSelect = document.getElementById('day_of_week');
        if (dayOfWeekSelect) {
            dayOfWeekSelect.value = sample.day_of_week;
        }
        
        // Show success notification with sample description
        showNotification(`Sample data loaded: ${sample.description}`, 'success');
        
        // Auto-predict after loading sample (optional - uncomment to enable)
        // setTimeout(() => document.getElementById('predict-btn').click(), 500);
        
    } catch (error) {
        console.error('❌ Error loading sample data:', error);
        showNotification('Failed to load sample data', 'error');
    }
}

// Helper function to update field with validation
function updateFieldWithValidation(fieldId, value, min, max) {
    const field = document.getElementById(fieldId);
    if (field) {
        // Ensure value is within bounds
        let validatedValue = value;
        if (min !== undefined && value < min) validatedValue = min;
        if (max !== undefined && value > max) validatedValue = max;
        
        field.value = validatedValue;
        
        // Trigger input event for any listeners
        field.dispatchEvent(new Event('input', { bubbles: true }));
    } else {
        console.warn(`⚠️ Field not found: ${fieldId}`);
    }
}

// Helper function to sync slider with input
function syncSlider(sliderId, fieldId, value) {
    const slider = document.getElementById(sliderId);
    if (slider) {
        slider.value = value;
        // Trigger input event
        slider.dispatchEvent(new Event('input', { bubbles: true }));
    }
}

// Helper function to update season selector
function updateSeasonSelector(season) {
    const seasonOptions = document.querySelectorAll('.season-option');
    let found = false;
    
    seasonOptions.forEach(opt => {
        if (opt.getAttribute('data-value') === season) {
            opt.classList.add('active');
            found = true;
        } else {
            opt.classList.remove('active');
        }
    });
    
    if (!found) {
        console.warn(`⚠️ Season "${season}" not found, defaulting to Summer`);
        // Default to Summer if season not found
        seasonOptions.forEach(opt => {
            if (opt.getAttribute('data-value') === 'Summer') {
                opt.classList.add('active');
            }
        });
    }
}

// Helper function to update toggle switches
function updateToggleSwitches(holiday, functioning) {
    // Holiday toggle
    const holidayToggle = document.getElementById('holiday-toggle');
    const holidayNo = document.getElementById('Holiday_No Holiday');
    const holidayYes = document.getElementById('Holiday_Holiday');
    
    if (holidayToggle && holidayNo && holidayYes) {
        holidayToggle.checked = holiday;
        holidayNo.value = holiday ? 0 : 1;
        holidayYes.value = holiday ? 1 : 0;
        
        // Trigger change event
        holidayToggle.dispatchEvent(new Event('change', { bubbles: true }));
    }
    
    // Functioning day toggle
    const functioningToggle = document.getElementById('functioning-toggle');
    const functioningYes = document.getElementById('Functioning Day_Yes');
    const functioningNo = document.getElementById('Functioning Day_No');
    
    if (functioningToggle && functioningYes && functioningNo) {
        functioningToggle.checked = functioning;
        functioningYes.value = functioning ? 1 : 0;
        functioningNo.value = functioning ? 0 : 1;
        
        // Trigger change event
        functioningToggle.dispatchEvent(new Event('change', { bubbles: true }));
    }
}

// Helper function to update humidity bar
function updateHumidityBar(humidity) {
    const humidityBar = document.querySelector('.humidity-fill');
    if (humidityBar) {
        humidityBar.style.width = `${humidity}%`;
    }
}

// Quick sample function - loads a specific scenario
function loadSpecificSample(sampleType) {
    const samples = {
        'morning': {
            Hour: 8, 'Temperature(캜)': 18.5, 'Humidity(%)': 70, 
            'Wind speed (m/s)': 1.5, 'Visibility (10m)': 1800, 
            'Rainfall(mm)': 0.0, 'Snowfall (cm)': 0.0,
            'Solar Radiation (MJ/m2)': 0.8, 'Dew point temperature(캜)': 12.0,
            season: 'Spring', holiday: false, functioning: true,
            day_of_week: 1
        },
        'afternoon': {
            Hour: 14, 'Temperature(캜)': 22.5, 'Humidity(%)': 60, 
            'Wind speed (m/s)': 2.5, 'Visibility (10m)': 2000, 
            'Rainfall(mm)': 0.0, 'Snowfall (cm)': 0.0,
            'Solar Radiation (MJ/m2)': 1.5, 'Dew point temperature(캜)': 15.0,
            season: 'Summer', holiday: false, functioning: true,
            day_of_week: 3
        },
        'evening': {
            Hour: 17, 'Temperature(캜)': 20.0, 'Humidity(%)': 65, 
            'Wind speed (m/s)': 3.0, 'Visibility (10m)': 1900, 
            'Rainfall(mm)': 0.0, 'Snowfall (cm)': 0.0,
            'Solar Radiation (MJ/m2)': 0.5, 'Dew point temperature(캜)': 14.0,
            season: 'Autumn', holiday: false, functioning: true,
            day_of_week: 4
        },
        'winter': {
            Hour: 12, 'Temperature(캜)': 5.0, 'Humidity(%)': 45, 
            'Wind speed (m/s)': 5.5, 'Visibility (10m)': 1500, 
            'Rainfall(mm)': 0.0, 'Snowfall (cm)': 2.5,
            'Solar Radiation (MJ/m2)': 0.3, 'Dew point temperature(캜)': -1.0,
            season: 'Winter', holiday: true, functioning: true,
            day_of_week: 6
        },
        'rainy': {
            Hour: 9, 'Temperature(캜)': 15.0, 'Humidity(%)': 85, 
            'Wind speed (m/s)': 2.0, 'Visibility (10m)': 800, 
            'Rainfall(mm)': 5.5, 'Snowfall (cm)': 0.0,
            'Solar Radiation (MJ/m2)': 0.2, 'Dew point temperature(캜)': 12.5,
            season: 'Spring', holiday: false, functioning: true,
            day_of_week: 0
        }
    };
    
    const sample = samples[sampleType];
    if (sample) {
        // Update all fields with the specific sample
        updateFieldWithValidation('Hour', sample.Hour, 0, 23);
        updateFieldWithValidation('Temperature(캜)', sample['Temperature(캜)'], -20, 40);
        updateFieldWithValidation('Humidity(%)', sample['Humidity(%)'], 0, 100);
        updateFieldWithValidation('Wind speed (m/s)', sample['Wind speed (m/s)'], 0, 20);
        updateFieldWithValidation('Visibility (10m)', sample['Visibility (10m)'], 0, 5000);
        updateFieldWithValidation('Rainfall(mm)', sample['Rainfall(mm)'], 0, 100);
        updateFieldWithValidation('Snowfall (cm)', sample['Snowfall (cm)'], 0, 50);
        updateFieldWithValidation('Solar Radiation (MJ/m2)', sample['Solar Radiation (MJ/m2)'], 0, 5);
        updateFieldWithValidation('Dew point temperature(캜)', sample['Dew point temperature(캜)'], -30, 30);
        
        updateSeasonSelector(sample.season);
        updateToggleSwitches(sample.holiday, sample.functioning);
        updateHumidityBar(sample['Humidity(%)']);
        
        // Update day of week
        const dayOfWeekSelect = document.getElementById('day_of_week');
        if (dayOfWeekSelect) {
            dayOfWeekSelect.value = sample.day_of_week;
        }
        
        // Update descriptions
        updateHourDescription(sample.Hour);
        updateWeatherInsight(sample['Temperature(캜)']);
        
        showNotification(`Loaded ${sampleType} scenario`, 'success');
    }
}

// Clear form 
function clearForm() {
    // Reset to default values
    document.getElementById('Hour').value = 14;
    if (document.getElementById('hour-slider')) {
        document.getElementById('hour-slider').value = 14;
    }
    
    document.getElementById('Temperature(캜)').value = 22.5;
    if (document.getElementById('temp-slider')) {
        document.getElementById('temp-slider').value = 22.5;
    }
    
    document.getElementById('Humidity(%)').value = 60;
    document.getElementById('Wind speed (m/s)').value = 2.5;
    document.getElementById('Visibility (10m)').value = 2000;
    document.getElementById('Rainfall(mm)').value = 0.0;
    document.getElementById('Snowfall (cm)').value = 0.0;
    document.getElementById('Solar Radiation (MJ/m2)').value = 1.5;
    document.getElementById('Dew point temperature(캜)').value = 10.0;
    
    // Reset toggle switches
    document.getElementById('holiday-toggle').checked = false;
    document.getElementById('Holiday_No Holiday').value = 1;
    document.getElementById('Holiday_Holiday').value = 0;
    
    document.getElementById('functioning-toggle').checked = true;
    document.getElementById('Functioning Day_Yes').value = 1;
    document.getElementById('Functioning Day_No').value = 0;
    
    // Reset day of week
    const dayOfWeekSelect = document.getElementById('day_of_week');
    if (dayOfWeekSelect) {
        dayOfWeekSelect.value = 3; // Thursday
    }
    
    // Reset season to Summer
    document.querySelectorAll('.season-option').forEach(opt => {
        opt.classList.remove('active');
        if (opt.getAttribute('data-value') === 'Summer') {
            opt.classList.add('active');
        }
    });
    
    // Reset humidity bar
    const humidityBar = document.querySelector('.humidity-fill');
    if (humidityBar) {
        humidityBar.style.width = '60%';
    }
    
    // Reset hour description
    updateHourDescription(14);
    
    // Reset weather insight
    updateWeatherInsight(22.5);
    
    showNotification('Form cleared', 'info');
}

// Helper function to update hour description
function updateHourDescription(hour) {
    const hourNum = parseInt(hour);
    let description = "Moderate demand expected";
    
    if (hourNum >= 0 && hourNum <= 5) {
        description = "Late Night - Very low demand expected";
    } else if (hourNum >= 6 && hourNum <= 9) {
        description = "Morning Peak - High demand expected";
    } else if (hourNum >= 10 && hourNum <= 15) {
        description = "Daytime - Moderate demand expected";
    } else if (hourNum >= 16 && hourNum <= 19) {
        description = "Evening Peak - Very high demand expected";
    } else if (hourNum >= 20 && hourNum <= 23) {
        description = "Evening - Low to moderate demand expected";
    }
    
    const hourDesc = document.getElementById('hour-description');
    if (hourDesc) {
        hourDesc.textContent = description;
    }
}

// Helper function to update weather insight
function updateWeatherInsight(temp) {
    const tempNum = parseFloat(temp);
    let insight = "Current conditions suggest ";
    
    if (tempNum < 5) {
        insight += "<strong>low biking suitability</strong> due to cold temperatures.";
    } else if (tempNum < 15) {
        insight += "<strong>moderate biking suitability</strong> with cooler conditions.";
    } else if (tempNum < 25) {
        insight += "<strong>high biking suitability</strong> with optimal temperature.";
    } else {
        insight += "<strong>reduced biking suitability</strong> due to heat.";
    }
    
    const weatherInsight = document.getElementById('weather-insight');
    if (weatherInsight) {
        const pElement = weatherInsight.querySelector('p');
        if (pElement) {
            pElement.innerHTML = insight;
        }
    }
}

// Copy prediction to clipboard
function copyPrediction() {
    if (currentPredictionData) {
        const prediction = currentPredictionData.predicted_rented_bikes || 
                         currentPredictionData.rounded || 
                         Math.round(currentPredictionData.prediction || 0);
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
function exportPrediction(format = 'json') {
    console.log('📤 Export function called');
    
    // Check if there's a prediction to export
    if (!currentPredictionData) {
        // Try to get data from the UI if currentPredictionData is not set
        const resultContent = document.getElementById('result-content');
        const predictionCount = document.getElementById('prediction-count');
        
        if (!resultContent || resultContent.style.display === 'none' || !predictionCount || predictionCount.textContent === '0') {
            showNotification('No prediction to export. Generate a prediction first.', 'warning');
            return;
        }
        
        // Create a minimal data object from UI
        currentPredictionData = {
            predicted_rented_bikes: parseInt(predictionCount.textContent) || 0,
            model_type: document.getElementById('detail-model')?.textContent.toLowerCase().replace(' ', '-') || 'original',
            input_summary: {
                features_used: document.getElementById('detail-features')?.textContent || '14'
            },
            prediction_id: document.getElementById('prediction-id')?.textContent || 'N/A',
            timestamp: new Date().toISOString(),
            confidence: parseInt(confidenceBadge?.textContent) || 80
        };
    }
    
    try {
        // Gather complete prediction data
        const exportData = {
            exportDate: new Date().toISOString(),
            prediction: {
                bikesRequired: currentPredictionData.predicted_rented_bikes || 
                              currentPredictionData.rounded || 
                              Math.round(currentPredictionData.prediction || 0),
                modelUsed: formatModelName(currentPredictionData.model_type),
                modelType: currentPredictionData.model_type,
                features: currentPredictionData.input_summary?.features_used || '14',
                responseTime: document.getElementById('response-time')?.textContent || '0.00s',
                predictionId: currentPredictionData.prediction_id || document.getElementById('prediction-id')?.textContent || 'N/A',
                timestamp: currentPredictionData.timestamp || new Date().toISOString(),
                confidence: currentPredictionData.confidence || 80
            },
            modelMetrics: {
                r2: document.getElementById('metric-r2')?.textContent || 'N/A',
                rmse: document.getElementById('metric-rmse')?.textContent || 'N/A',
                mae: document.getElementById('metric-mae')?.textContent || 'N/A',
                gap: document.getElementById('metric-gap')?.textContent || 'N/A'
            }
        };
        
        // Get input parameters from the form
        const form = document.getElementById('prediction-form');
        if (form) {
            exportData.inputParameters = {};
            const formInputs = form.querySelectorAll('input, select');
            formInputs.forEach(input => {
                if (input.name && input.type !== 'hidden') {
                    exportData.inputParameters[input.name] = input.value;
                }
            });
            
            // Add season
            const activeSeason = document.querySelector('.season-option.active')?.getAttribute('data-value') || 'Summer';
            exportData.inputParameters.season = activeSeason;
        }
        
        // Format based on export type
        let content, filename, mimeType;
        
        if (format === 'json' || format === 'detailed') {
            // JSON format with pretty printing
            content = JSON.stringify(exportData, null, 2);
            filename = `bike_prediction_${new Date().toISOString().slice(0,19).replace(/:/g, '-')}.json`;
            mimeType = 'application/json';
        } else if (format === 'csv') {
            // Simple CSV format
            const rows = [];
            
            // Add prediction details
            rows.push(['Field', 'Value']);
            rows.push(['Export Date', exportData.exportDate]);
            rows.push(['Prediction ID', exportData.prediction.predictionId]);
            rows.push(['Bikes Required', exportData.prediction.bikesRequired]);
            rows.push(['Model Used', exportData.prediction.modelUsed]);
            rows.push(['Features Count', exportData.prediction.features]);
            rows.push(['Response Time', exportData.prediction.responseTime]);
            rows.push(['Confidence', exportData.prediction.confidence + '%']);
            rows.push(['R² Score', exportData.modelMetrics.r2]);
            rows.push(['RMSE', exportData.modelMetrics.rmse]);
            rows.push(['MAE', exportData.modelMetrics.mae]);
            rows.push(['Accuracy Gap', exportData.modelMetrics.gap]);
            rows.push('', ''); // Empty row for separation
            rows.push(['Input Parameters', 'Value']);
            
            // Add input parameters
            if (exportData.inputParameters) {
                Object.entries(exportData.inputParameters).forEach(([key, value]) => {
                    rows.push([key, value]);
                });
            }
            
            const csvContent = rows.map(row => {
                // Escape quotes and wrap in quotes if contains comma
                return row.map(cell => {
                    if (typeof cell === 'string' && (cell.includes(',') || cell.includes('"'))) {
                        return `"${cell.replace(/"/g, '""')}"`;
                    }
                    return cell;
                }).join(',');
            }).join('\n');
            
            content = csvContent;
            filename = `bike_prediction_${new Date().toISOString().slice(0,19).replace(/:/g, '-')}.csv`;
            mimeType = 'text/csv';
        } else {
            // Simple text format
            content = `BIKE DEMAND PREDICTION
===============================
Date: ${new Date().toLocaleString()}
Prediction ID: ${exportData.prediction.predictionId}

RESULT:
Bikes Required: ${exportData.prediction.bikesRequired}
Model Used: ${exportData.prediction.modelUsed}
Confidence: ${exportData.prediction.confidence}%

MODEL METRICS:
R² Score: ${exportData.modelMetrics.r2}
RMSE: ${exportData.modelMetrics.rmse}
MAE: ${exportData.modelMetrics.mae}

INPUT PARAMETERS:
${Object.entries(exportData.inputParameters || {}).map(([k, v]) => `${k}: ${v}`).join('\n')}

===============================
Generated by UrbanFlow Intelligence`;
            
            filename = `bike_prediction_${new Date().toISOString().slice(0,10)}.txt`;
            mimeType = 'text/plain';
        }
        
        // Create download link
        const blob = new Blob([content], { type: `${mimeType};charset=utf-8;` });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        
        link.href = url;
        link.download = filename;
        link.style.display = 'none';
        
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        
        showNotification(`Prediction exported as ${filename.split('.').pop().toUpperCase()} file`, 'success');
        
        // Log export for debugging
        console.log(`✅ Export successful: ${filename}`);
        console.log('📊 Exported data:', exportData);
        
    } catch (error) {
        console.error('❌ Export failed:', error);
        showNotification('Failed to export prediction. Please try again.', 'error');
    }
}

// Enhanced export function with format selection
function exportPredictionWithFormat() {
    // Create format selector modal
    const modal = document.createElement('div');
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.5);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 1000;
    `;
    
    modal.innerHTML = `
        <div style="background: white; padding: 25px; border-radius: 12px; width: 300px; text-align: center;">
            <h3 style="margin-top: 0; color: #2c3e50;">Export Format</h3>
            <p style="color: #6c757d; margin-bottom: 20px;">Choose export format:</p>
            
            <button onclick="exportPrediction('json'); this.closest('div').closest('div').remove()" 
                    style="display: block; width: 100%; padding: 12px; margin: 8px 0; 
                           background: #4361ee; color: white; border: none; 
                           border-radius: 8px; cursor: pointer; font-size: 16px;">
                <i class="fas fa-code"></i> JSON (Detailed)
            </button>
            
            <button onclick="exportPrediction('csv'); this.closest('div').closest('div').remove()" 
                    style="display: block; width: 100%; padding: 12px; margin: 8px 0; 
                           background: #2ecc71; color: white; border: none; 
                           border-radius: 8px; cursor: pointer; font-size: 16px;">
                <i class="fas fa-table"></i> CSV (Spreadsheet)
            </button>
            
            <button onclick="exportPrediction('text'); this.closest('div').closest('div').remove()" 
                    style="display: block; width: 100%; padding: 12px; margin: 8px 0; 
                           background: #f39c12; color: white; border: none; 
                           border-radius: 8px; cursor: pointer; font-size: 16px;">
                <i class="fas fa-file-alt"></i> Text (Simple)
            </button>
            
            <button onclick="this.closest('div').closest('div').remove()" 
                    style="display: block; width: 100%; padding: 10px; margin-top: 15px; 
                           background: #6c757d; color: white; border: none; 
                           border-radius: 8px; cursor: pointer;">
                Cancel
            </button>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // Close when clicking outside
    modal.addEventListener('click', function(e) {
        if (e.target === modal) {
            modal.remove();
        }
    });
}

// Export functions for global access
window.loadSampleData = loadSampleData;
window.loadSpecificSample = loadSpecificSample;
window.clearForm = clearForm;
window.copyPrediction = copyPrediction;
window.exportPrediction = exportPrediction;
window.exportPredictionWithFormat = exportPredictionWithFormat;
window.openModelManager = openModelManager;
window.closeModelManager = closeModelManager;
window.selectModel = selectModel;
window.showNotification = showNotification;
window.updateHourDescription = updateHourDescription;
window.updateWeatherInsight = updateWeatherInsight;

console.log('✅ script.js loaded successfully');