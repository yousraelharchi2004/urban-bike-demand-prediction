// Professional Analytics Dashboard JavaScript
const ANALYTICS_DEBUG = false;
function analyticsDbg(...args) { if (ANALYTICS_DEBUG) console.log(...args); }

class AnalyticsDashboard {
    constructor() {
        this.baseUrl = window.location.origin;
        this.predictionLimit = 10; // Reduced for better UX
        this.charts = {};
        this.data = {};
        this.currentTimeFilter = 'today';
        this.currentHistoryFilter = 'all';
        this.customDateRange = null;
        
        // Professional color scheme
        this.colors = {
            primary: '#2563eb',
            primaryDark: '#1d4ed8',
            secondary: '#6366f1',
            success: '#16a34a',
            warning: '#ca8a04',
            danger: '#dc2626',
            info: '#0284c7',
            dark: '#0f172a',
            light: '#f1f5f9',
            gray100: '#f1f5f9',
            gray200: '#e2e8f0',
            gray300: '#cbd5e1',
            gray400: '#94a3b8',
            gray500: '#adb5bd',
            gray600: '#6c757d',
            gray700: '#495057',
            gray800: '#1e293b',
            gray900: '#0f172a'
        };
        
        // Initialize theme
        this.initTheme();
        
        // Initialize date pickers
        this.initDatePickers();
    }

    initTheme() {
        // Set CSS custom properties
        document.documentElement.style.setProperty('--primary', this.colors.primary);
        document.documentElement.style.setProperty('--primary-dark', this.colors.primaryDark);
        document.documentElement.style.setProperty('--secondary', this.colors.secondary);
        document.documentElement.style.setProperty('--success', this.colors.success);
        document.documentElement.style.setProperty('--warning', this.colors.warning);
        document.documentElement.style.setProperty('--danger', this.colors.danger);
        document.documentElement.style.setProperty('--dark', this.colors.dark);
    }

    initDatePickers() {
        // Initialize flatpickr for date range picker if available
        if (typeof flatpickr !== 'undefined') {
            const dateRangeInput = document.getElementById('dateRangePicker');
            if (dateRangeInput) {
                flatpickr(dateRangeInput, {
                    mode: "range",
                    dateFormat: "Y-m-d",
                    maxDate: "today",
                    onChange: (selectedDates, dateStr, instance) => {
                        if (selectedDates.length === 2) {
                            const [startDate, endDate] = selectedDates;
                            this.applyCustomDateRange(startDate, endDate);
                        }
                    }
                });
            }
        }
    }

    async init() {
                try {
            // Show loading state (but preserve canvas elements)
            this.showLoadingState();
            
            // Load all data with error handling
            await this.loadAllData();
            
            // Create all charts
            this.createCharts();
            
            // Update UI components
            this.updateUI();
            
            // Initialize event listeners
            this.initEventListeners();
            
                        this.showNotification('Dashboard loaded successfully!', 'success');
            
        } catch (error) {
            console.error('Dashboard initialization error:', error);
            this.showError('Failed to initialize dashboard. Please refresh the page.');
            
            // Try to at least show the UI with fallback data
            try {
                this.updateUI();
            } catch (uiError) {
                console.error('Could not update UI:', uiError);
            }
        } finally {
            // Always hide loading state
            this.hideLoadingState();
        }
    }

    initEventListeners() {
        // Time filter buttons
        document.querySelectorAll('.time-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const timeRange = btn.getAttribute('data-time');
                this.applyTimeFilter(timeRange);
            });
        });
        
        // History filter buttons
        document.querySelectorAll('.history-filter').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.history-filter').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                const filter = btn.getAttribute('data-filter');
                this.applyHistoryFilter(filter);
            });
        });
        
        // Custom date range apply button
        const applyCustomRangeBtn = document.getElementById('applyCustomRange');
        if (applyCustomRangeBtn) {
            applyCustomRangeBtn.addEventListener('click', () => this.applyCustomRangeFromInputs());
        }
        
        // Date range picker clear button
        const clearDateRangeBtn = document.getElementById('clearDateRange');
        if (clearDateRangeBtn) {
            clearDateRangeBtn.addEventListener('click', () => this.clearCustomDateRange());
        }
    }

    async loadAllData() {
        try {
                        // Load KPIs with current time filter
            await this.loadKPIs();
            
            // Load feature importance
            await this.loadFeatureImportance();

            // Load SHAP importance
            await this.loadShapImportance();
            
            // Load model performance
            await this.loadModelPerformance();
            
            // Load hourly patterns with current time filter
            await this.loadHourlyPatterns();
            
            // Load insights with current time filter
            await this.loadInsights();
            
            // Load prediction history
            await this.loadPredictionHistory();
            
                    } catch (error) {
            console.error('Error loading analytics data:', error);
            this.data = {
                kpis: null,
                featureImportance: null,
                shapImportance: null,
                modelPerformance: null,
                hourlyPatterns: null,
                insights: null,
                history: {
                    success: true,
                    total_predictions: 0,
                    filtered_count: 0,
                    history: [],
                    filter: this.currentHistoryFilter || 'all'
                }
            };
        }
    }

    async loadKPIs() {
        try {
            const queryParams = this.getTimeFilterQueryParams();
            const kpisResponse = await fetch(`${this.baseUrl}/api/analytics/kpis?${queryParams}`);
            if (kpisResponse.ok) {
                this.data.kpis = await kpisResponse.json();
                            } else {
                throw new Error(`KPIs API returned ${kpisResponse.status}`);
            }
        } catch (error) {
                        this.data.kpis = null;
        }
    }

    async loadFeatureImportance() {
        try {
            const featureResponse = await fetch(`${this.baseUrl}/api/analytics/feature-importance`);
            if (featureResponse.ok) {
                this.data.featureImportance = await featureResponse.json();
                            } else {
                throw new Error(`Feature API returned ${featureResponse.status}`);
            }
        } catch (error) {
                        this.data.featureImportance = null;
        }
    }

    async loadShapImportance() {
        try {
            const response = await fetch(`${this.baseUrl}/api/analytics/shap-importance`);
            if (response.ok) {
                this.data.shapImportance = await response.json();
            } else {
                throw new Error(`SHAP API returned ${response.status}`);
            }
        } catch (error) {
            this.data.shapImportance = null;
        }
    }

    async loadModelPerformance() {
        try {
            const performanceResponse = await fetch(`${this.baseUrl}/api/analytics/model-performance`);
            if (performanceResponse.ok) {
                this.data.modelPerformance = await performanceResponse.json();
                            } else {
                throw new Error(`Performance API returned ${performanceResponse.status}`);
            }
        } catch (error) {
                        this.data.modelPerformance = null;
        }
    }

    async loadHourlyPatterns() {
        try {
            const queryParams = this.getTimeFilterQueryParams();
            const patternsResponse = await fetch(`${this.baseUrl}/api/analytics/hourly-patterns?${queryParams}`);
            if (patternsResponse.ok) {
                this.data.hourlyPatterns = await patternsResponse.json();
                            } else {
                throw new Error(`Patterns API returned ${patternsResponse.status}`);
            }
        } catch (error) {
                        this.data.hourlyPatterns = null;
        }
    }

    async loadInsights() {
        try {
            const queryParams = this.getTimeFilterQueryParams();
            const insightsResponse = await fetch(`${this.baseUrl}/api/analytics/insights?${queryParams}`);
            if (insightsResponse.ok) {
                this.data.insights = await insightsResponse.json();
                            } else {
                throw new Error(`Insights API returned ${insightsResponse.status}`);
            }
        } catch (error) {
                        this.data.insights = null;
        }
    }

    async loadPredictionHistory() {
        try {
            const filter = this.currentHistoryFilter || 'all';
            const historyResponse = await fetch(`${this.baseUrl}/prediction-history?filter=${filter}&limit=${this.predictionLimit}`);
            if (historyResponse.ok) {
                const historyData = await historyResponse.json();
                this.data.history = historyData;
                            } else {
                throw new Error(`History API returned ${historyResponse.status}`);
            }
        } catch (error) {
                        this.data.history = {
                success: true,
                total_predictions: 0,
                filtered_count: 0,
                history: [],
                filter: this.currentHistoryFilter || 'all'
            };
        }
    }

    getTimeFilterQueryParams() {
        let queryParams = `filter=${this.currentTimeFilter}`;
        
        if (this.currentTimeFilter === 'custom' && this.customDateRange) {
            const { startDate, endDate } = this.customDateRange;
            queryParams += `&start_date=${startDate}&end_date=${endDate}`;
        }
        
        return queryParams;
    }

    async applyTimeFilter(timeRange) {
        try {
            this.currentTimeFilter = timeRange;
            this.customDateRange = null; // Clear custom range when switching to preset filters
            
            // Update UI button states
            document.querySelectorAll('.time-btn').forEach(btn => {
                btn.classList.remove('active');
                if (btn.getAttribute('data-time') === timeRange) {
                    btn.classList.add('active');
                }
            });
            
            // Hide custom date range info if switching away from custom
            if (timeRange !== 'custom') {
                this.hideCustomDateRangeInfo();
            }
            
            this.showNotification(`Loading ${timeRange} analytics...`, 'info');
            this.showLoadingState();
            
            // Reload time-sensitive data with new filter
            await this.loadKPIs();
            await this.loadHourlyPatterns();
            await this.loadInsights();
            
            // Update charts with filtered data
            this.updateChartsWithFilteredData();
            
            // Update UI
            this.updateUI();
            
            this.showNotification(`Showing analytics for ${timeRange} period`, 'success');
            
        } catch (error) {
            console.error('Time filter error:', error);
            this.showNotification(`Could not load ${timeRange} data`, 'error');
            this.hideLoadingState();
        }
    }

    async applyCustomDateRange(startDate, endDate) {
        try {
            this.currentTimeFilter = 'custom';
            this.customDateRange = { startDate, endDate };
            
            // Update UI button states
            document.querySelectorAll('.time-btn').forEach(btn => {
                btn.classList.remove('active');
                if (btn.getAttribute('data-time') === 'custom') {
                    btn.classList.add('active');
                }
            });
            
            // Format dates for display
            const formattedStart = startDate.toISOString().split('T')[0];
            const formattedEnd = endDate.toISOString().split('T')[0];
            
            this.showNotification(`Loading custom range: ${formattedStart} to ${formattedEnd}`, 'info');
            this.showLoadingState();
            
            // Reload time-sensitive data with custom filter
            await this.loadKPIs();
            await this.loadHourlyPatterns();
            await this.loadInsights();
            
            // Update charts with filtered data
            this.updateChartsWithFilteredData();
            
            // Update UI with custom range info
            this.updateCustomDateRangeInfo(formattedStart, formattedEnd);
            this.updateUI();
            
            this.showNotification(`Showing analytics for custom range: ${formattedStart} to ${formattedEnd}`, 'success');
            
        } catch (error) {
            console.error('Custom date range error:', error);
            this.showNotification('Could not load custom date range data', 'error');
            this.hideLoadingState();
        }
    }

    async applyCustomRangeFromInputs() {
        const startDateInput = document.getElementById('customStartDate');
        const endDateInput = document.getElementById('customEndDate');
        
        if (!startDateInput || !endDateInput) {
            this.showNotification('Date range inputs not found', 'error');
            return;
        }
        
        const startDate = new Date(startDateInput.value);
        const endDate = new Date(endDateInput.value);
        
        if (!startDateInput.value || !endDateInput.value) {
            this.showNotification('Please select both start and end dates', 'error');
            return;
        }
        
        if (startDate > endDate) {
            this.showNotification('Start date must be before end date', 'error');
            return;
        }
        
        await this.applyCustomDateRange(startDate, endDate);
    }

    updateCustomDateRangeInfo(startDate, endDate) {
        let customRangeInfo = document.getElementById('customRangeInfo');
        
        if (!customRangeInfo) {
            customRangeInfo = document.createElement('div');
            customRangeInfo.id = 'customRangeInfo';
            customRangeInfo.className = 'custom-range-info';
            
            const timeFilterContainer = document.querySelector('.time-filter-container');
            if (timeFilterContainer) {
                timeFilterContainer.appendChild(customRangeInfo);
            }
        }
        
        customRangeInfo.innerHTML = `
            <div class="custom-range-display">
                <i class="fas fa-calendar-alt"></i>
                <span>${startDate} to ${endDate}</span>
                <button class="btn-clear-range" onclick="window.dashboard.clearCustomDateRange()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
    }

    hideCustomDateRangeInfo() {
        const customRangeInfo = document.getElementById('customRangeInfo');
        if (customRangeInfo) {
            customRangeInfo.style.display = 'none';
        }
    }

    clearCustomDateRange() {
        this.customDateRange = null;
        this.hideCustomDateRangeInfo();
        
        // Reset to today filter
        this.applyTimeFilter('today');
    }

    async loadFilteredHistory(filter = 'all', limit = null) {
        try {
            const limitParam = limit ? `&limit=${limit}` : '';
            const url = `${this.baseUrl}/prediction-history?filter=${filter}${limitParam}`;
            const response = await fetch(url);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.success) {
                this.data.history = data;
                this.currentHistoryFilter = filter;
                                return data;
            } else {
                throw new Error(data.error || 'Failed to load history');
            }
            
        } catch (error) {
                        throw error;
        }
    }

    createCharts() {
        // Clear existing charts
        Object.values(this.charts).forEach(chart => {
            if (chart) chart.destroy();
        });
        this.charts = {};

        // Create Feature Importance Chart
        this.createFeatureImportanceChart();

        // Create SHAP Importance Chart
        this.createShapImportanceChart();

        // Create Performance Radar Chart
        this.createPerformanceRadarChart();
        
        // Create Hourly Pattern Chart
        this.createHourlyPatternChart();
    }

    createFeatureImportanceChart() {
        const ctx = document.getElementById('featureImportanceChart');
        if (!ctx) return;

        // Use real data if available, otherwise use fallback
        let features = [];
        let importanceData = [];
        let chartTitle = "Feature Importance";
        
        if (this.data.featureImportance?.success) {
            const featureData = this.data.featureImportance.features;
            features = featureData.map(f => f.feature);
            importanceData = featureData.map(f => f.importance * 100); // Convert to percentage
            chartTitle = `Feature Importance (${this.data.featureImportance.source_model})`;
        } else {
            // Fallback data
            features = [
                'Temperature(°C)', 'Hour', 'Solar Radiation (MJ/m²)', 
                'Functioning Day_Yes', 'Humidity(%)', 'Seasons_Summer',
                'Wind speed (m/s)', 'Dew point temperature(°C)'
            ];
            importanceData = [25, 20, 15, 12, 10, 8, 5, 5];
            chartTitle = "Feature Importance (Sample)";
        }

        // Create gradient for bars
        const gradient = ctx.getContext('2d').createLinearGradient(0, 0, 0, 400);
        gradient.addColorStop(0, this.colors.primary);
        gradient.addColorStop(1, this.colors.secondary);

        this.charts.featureImportance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: features,
                datasets: [{
                    label: 'Importance (%)',
                    data: importanceData,
                    backgroundColor: gradient,
                    borderColor: this.colors.primaryDark,
                    borderWidth: 2,
                    borderRadius: 8,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    title: {
                        display: true,
                        text: chartTitle,
                        font: { size: 16 }
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => `${context.dataset.label}: ${context.raw.toFixed(1)}%`
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Importance (%)'
                        },
                        ticks: { callback: value => value + '%' }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Features'
                        }
                    }
                }
            }
        });
        
        // Update insight text
        this.updateFeatureInsight();
    }

    createShapImportanceChart() {
        const ctx = document.getElementById('shapImportanceChart');
        if (!ctx) return;

        let features = [];
        let shapValues = [];
        let barColors = [];
        let chartTitle = 'SHAP Feature Importance';

        if (this.data.shapImportance?.success && this.data.shapImportance.features.length > 0) {
            const shapData = this.data.shapImportance.features;
            features = shapData.map(f => f.feature);
            shapValues = shapData.map(f => f.mean_shap);
            barColors = shapValues.map(v => v >= 0 ? '#06b6d4' : '#ef4444');
            chartTitle = `SHAP Analysis (${this.data.shapImportance.source})`;

            // Update insight
            const topPositive = shapData.find(f => f.direction === 'positive');
            const topNegative = shapData.find(f => f.direction === 'negative');
            const insightEl = document.getElementById('shapInsight');
            if (insightEl) {
                let text = '';
                if (topPositive) text += `"${topPositive.feature}" increases demand. `;
                if (topNegative) text += `"${topNegative.feature}" decreases demand.`;
                insightEl.innerHTML = `<p><i class="fas fa-lightbulb"></i> <strong>Insight:</strong> ${text}</p>`;
            }
        } else {
            features = ['No SHAP data'];
            shapValues = [0];
            barColors = ['#94a3b8'];
            const insightEl = document.getElementById('shapInsight');
            if (insightEl) {
                insightEl.innerHTML = '<p><i class="fas fa-lightbulb"></i> <strong>Insight:</strong> SHAP analysis not available. Run SHAP on a trained model to see directional feature impact.</p>';
            }
        }

        this.charts.shapImportance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: features,
                datasets: [{
                    label: 'Mean SHAP Value',
                    data: shapValues,
                    backgroundColor: barColors,
                    borderColor: barColors,
                    borderWidth: 1,
                    borderRadius: 4,
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    title: {
                        display: true,
                        text: chartTitle,
                        font: { size: 16 }
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const val = context.raw;
                                const dir = val >= 0 ? 'increases' : 'decreases';
                                return `${dir} demand by ${Math.abs(val).toFixed(1)} bikes`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Mean SHAP Value (impact on prediction)' },
                        grid: { color: 'rgba(0,0,0,0.05)' }
                    },
                    y: {
                        ticks: { font: { size: 11 } }
                    }
                }
            }
        });
    }

    createPerformanceRadarChart() {
        const ctx = document.getElementById('performanceRadarChart');
        if (!ctx) return;

        let radarData = {
            labels: ['Accuracy (R²)', 'Precision', 'Recall', 'F1-Score', 'Stability', 'Speed'],
            datasets: [{
                label: 'Model Performance',
                data: [87, 88, 86, 85, 92, 96],
                backgroundColor: 'rgba(79, 70, 229, 0.1)',
                borderColor: this.colors.primary,
                pointBackgroundColor: this.colors.primary,
                borderWidth: 2,
                fill: true
            }]
        };
        
        let chartTitle = "Model Performance";

        if (this.data.modelPerformance?.success) {
            radarData = this.data.modelPerformance.radar_data;
            chartTitle = `Model Performance (${this.data.modelPerformance.total_models} models)`;
        }

        this.charts.performanceRadar = new Chart(ctx, {
            type: 'radar',
            data: radarData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: chartTitle,
                        font: { size: 16 }
                    }
                },
                scales: {
                    r: {
                        ticks: { display: false },
                        suggestedMin: 0,
                        suggestedMax: 100,
                        pointLabels: {
                            font: { size: 12 }
                        }
                    }
                }
            }
        });
        
        // Update insight text
        this.updatePerformanceInsight();
    }

    createHourlyPatternChart() {
        const ctx = document.getElementById('hourlyPatternChart');
        if (!ctx) return;

        let hours = Array.from({length: 24}, (_, i) => i);
        let hourlyDemand = [];
        let chartTitle = "Hourly Demand Pattern";
        let isSampleData = true;
        let filterInfo = '';

        if (this.data.hourlyPatterns?.success) {
            const patterns = this.data.hourlyPatterns.patterns;
            hours = patterns.map(p => p.hour);
            hourlyDemand = patterns.map(p => p.average_demand);
            chartTitle = `Hourly Demand Pattern`;
            isSampleData = this.data.hourlyPatterns.is_sample;
            filterInfo = ` (${this.data.hourlyPatterns.filter}, ${this.data.hourlyPatterns.filtered_predictions} predictions)`;
        } else {
            // Generate sample data
            hourlyDemand = hours.map(hour => {
                let demand = 500;
                if (hour >= 7 && hour <= 9) demand = 1200 + Math.random() * 300;
                else if (hour >= 17 && hour <= 19) demand = 1400 + Math.random() * 400;
                else if (hour >= 12 && hour <= 14) demand = 900 + Math.random() * 200;
                else if (hour >= 22 || hour <= 5) demand = 100 + Math.random() * 50;
                else demand = 600 + Math.random() * 200;
                return Math.round(demand);
            });
        }

        this.charts.hourlyPattern = new Chart(ctx, {
            type: 'line',
            data: {
                labels: hours.map(h => h.toString().padStart(2, '0') + ':00'),
                datasets: [{
                    label: 'Bike Demand Pattern',
                    data: hourlyDemand,
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderColor: this.colors.success,
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: chartTitle + (isSampleData ? " (Sample)" : "") + filterInfo,
                        font: { size: 16 }
                    }
                },
                scales: {
                    y: { 
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Bike Demand'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Hour of Day'
                        }
                    }
                }
            }
        });
        
        // Update insight text
        this.updateHourlyInsight();
    }

    updateChartsWithFilteredData() {
        try {
            // Update hourly pattern chart if it exists
            if (this.charts.hourlyPattern && this.data.hourlyPatterns?.success) {
                const patterns = this.data.hourlyPatterns.patterns;
                const hours = patterns.map(p => p.hour);
                const hourlyDemand = patterns.map(p => p.average_demand);
                
                this.charts.hourlyPattern.data.labels = hours.map(h => h.toString().padStart(2, '0') + ':00');
                this.charts.hourlyPattern.data.datasets[0].data = hourlyDemand;
                
                // Update chart title with filter info
                const filter = this.data.hourlyPatterns.filter || 'all';
                const count = this.data.hourlyPatterns.filtered_predictions || 0;
                const isSample = this.data.hourlyPatterns.is_sample;
                this.charts.hourlyPattern.options.plugins.title.text = 
                    `Hourly Demand Pattern (${filter}, ${count} predictions)`;
                
                this.charts.hourlyPattern.update();
                
                // Update insight text
                this.updateHourlyInsight();
            }
            
            // Update feature insight based on filter
            if (this.data.insights?.success) {
                const filteredInsights = this.data.insights.insights.filter(
                    insight => insight.type === 'features'
                );
                if (filteredInsights.length > 0) {
                    const featureInsightElement = document.getElementById('featureInsight');
                    if (featureInsightElement) {
                        featureInsightElement.innerHTML = `
                            <p><i class="fas fa-lightbulb"></i> <strong>Insight:</strong> 
                            ${filteredInsights[0].description}</p>
                        `;
                    }
                }
            }
            
                    } catch (error) {
            console.error('Error updating charts with filtered data:', error);
        }
    }

    updateUI() {
        this.updateKPIs();
        this.updatePredictionHistory();
        this.updateInsights();
        this.hideLoadingState();
    }

    updateKPIs() {
        const kpis = this.data.kpis?.kpis;
        
        if (kpis) {
            // Update accuracy KPI
            const accuracyElement = document.getElementById('kpi-accuracy');
            if (accuracyElement && kpis.accuracy) {
                accuracyElement.textContent = `${kpis.accuracy.value}%`;
                this.updateKPITrend('kpi-accuracy-trend', kpis.accuracy.trend);
            }

            // Update error KPI
            const errorElement = document.getElementById('kpi-error');
            if (errorElement && kpis.error) {
                errorElement.textContent = kpis.error.value;
                this.updateKPITrend('kpi-error-trend', kpis.error.trend);
            }

            // Update predictions KPI
            const predictionsElement = document.getElementById('kpi-predictions');
            if (predictionsElement && kpis.predictions) {
                predictionsElement.textContent = kpis.predictions.value.toLocaleString();
                this.updateKPITrend('kpi-predictions-trend', kpis.predictions.trend);
            }

            // Update features KPI
            const featuresElement = document.getElementById('kpi-features');
            if (featuresElement && kpis.features) {
                featuresElement.textContent = kpis.features.value;
            }
        }

        // Update filter count display
        const filterCountElement = document.getElementById('filterCount');
        if (filterCountElement && this.data.history) {
            const filter = this.data.history.filter || 'all';
            const count = this.data.history.filtered_count || 0;
            const total = this.data.history.total_predictions || 0;
            filterCountElement.textContent = `${count} of ${total} predictions`;
        }
    }

    updateKPITrend(elementId, trendText) {
        const trendElement = document.getElementById(elementId);
        if (!trendElement) return;
        
        trendElement.innerHTML = '';
        
        if (trendText.includes('+')) {
            trendElement.className = 'kpi-trend positive';
            trendElement.innerHTML = `<i class="fas fa-arrow-up"></i><span>${trendText}</span>`;
        } else if (trendText.includes('-')) {
            trendElement.className = 'kpi-trend negative';
            trendElement.innerHTML = `<i class="fas fa-arrow-down"></i><span>${trendText}</span>`;
        } else {
            trendElement.className = 'kpi-trend';
            trendElement.innerHTML = `<span>${trendText}</span>`;
        }
    }

    updateInsights() {
        const insightsContainer = document.querySelector('.insights-list');
        if (!insightsContainer || !this.data.insights?.success) return;

        const insights = this.data.insights.insights;
        
        insightsContainer.innerHTML = insights.map(insight => `
            <div class="insight-item ${insight.type}">
                <div class="insight-title">
                    <i class="fas fa-${insight.icon}"></i>
                    ${insight.title}
                </div>
                <div class="insight-desc">
                    ${insight.description}
                </div>
            </div>
        `).join('');
    }

    updateFeatureInsight() {
        const insightElement = document.getElementById('featureInsight');
        if (!insightElement || !this.data.featureImportance?.success) return;
        
        const topFeature = this.data.featureImportance.features[0];
        if (topFeature) {
            insightElement.innerHTML = `
                <p><i class="fas fa-lightbulb"></i> <strong>Insight:</strong> 
                "${topFeature.feature}" is the most important feature, contributing ${topFeature.importance_percent} to predictions.</p>
            `;
        }
    }

    updatePerformanceInsight() {
        const insightElement = document.getElementById('performanceInsight');
        if (!insightElement || !this.data.modelPerformance?.success) return;
        
        const modelCount = this.data.modelPerformance.total_models;
        insightElement.innerHTML = `
            <p><i class="fas fa-lightbulb"></i> <strong>Insight:</strong> 
            Comparing ${modelCount} model${modelCount > 1 ? 's' : ''}. Advanced models show improved accuracy with feature engineering.</p>
        `;
    }

    updateHourlyInsight() {
        const insightElement = document.getElementById('hourlyInsight');
        if (!insightElement) return;
        
        if (this.data.hourlyPatterns?.success && !this.data.hourlyPatterns.is_sample) {
            const patterns = this.data.hourlyPatterns.patterns;
            const peakHours = patterns.filter(p => p.is_peak_hour && p.average_demand > 1000);
            
            if (peakHours.length > 0) {
                const peakHour = peakHours.reduce((max, p) => p.average_demand > max.average_demand ? p : max);
                insightElement.innerHTML = `
                    <p><i class="fas fa-lightbulb"></i> <strong>Insight:</strong> 
                    Peak demand at ${peakHour.hour_label} with ~${Math.round(peakHour.average_demand)} bikes. 
                    Clear commuting patterns visible.</p>
                `;
                return;
            }
        }
        
        // Default insight
        insightElement.innerHTML = `
            <p><i class="fas fa-lightbulb"></i> <strong>Insight:</strong> 
            Peak demand occurs during morning (8-9 AM) and evening (5-7 PM) rush hours.</p>
        `;
    }

    updatePredictionHistory() {
        const historyContainer = document.getElementById('predictionHistory');
        if (!historyContainer) return;

        const history = this.data.history?.history;
        const filter = this.data.history?.filter || 'all';
        const filteredCount = this.data.history?.filtered_count || 0;
        const totalCount = this.data.history?.total_predictions || 0;

        if (!history || history.length === 0) {
            historyContainer.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-inbox"></i>
                    <p>No predictions found for "${filter}".</p>
                    <p style="font-size: 0.9rem; margin-top: 10px; color: var(--gray-600)">
                        Total predictions in database: ${totalCount}
                    </p>
                </div>
            `;
            return;
        }

        historyContainer.innerHTML = history.map(pred => {
            // Parse the timestamp to format it nicely
            const timestamp = pred.timestamp || pred.timestamp_readable || 'Unknown time';
            let formattedTime = timestamp;
            
            try {
                if (timestamp.includes('T')) {
                    // ISO format: 2024-01-06T11:01:37
                    const date = new Date(timestamp);
                    formattedTime = date.toLocaleString('en-US', {
                        month: 'short',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit'
                    });
                } else if (timestamp.includes('-')) {
                    // Readable format: 2024-01-06 11:01:37
                    const date = new Date(timestamp.replace(' ', 'T'));
                    formattedTime = date.toLocaleString('en-US', {
                        month: 'short',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit'
                    });
                }
            } catch (e) {
                console.warn('Could not parse timestamp:', timestamp);
            }

            // Get temperature value
            const temp = pred.input?.['Temperature(°C)'] || pred.input?.['Temperature(캜)'] || 'N/A';
            
            // Get season
            let season = "Unknown";
            if (pred.input?.Seasons) {
                season = pred.input.Seasons;
            } else if (pred.input?.Seasons_Spring === 1) {
                season = "Spring";
            } else if (pred.input?.Seasons_Summer === 1) {
                season = "Summer";
            } else if (pred.input?.Seasons_Winter === 1) {
                season = "Winter";
            } else if (pred.input?.Seasons_Autumn === 1) {
                season = "Autumn";
            }

            return `
                <div class="history-item">
                    <div class="history-meta">
                        <span class="history-id">Prediction #${pred.id || 'N/A'}</span>
                        <span class="history-time">${formattedTime}</span>
                    </div>
                    <div class="history-prediction">${(pred.rounded || pred.prediction || 0).toLocaleString()} bikes</div>
                    <div class="history-features">
                        <span class="feature-tag">Hour: ${pred.input?.Hour || 'N/A'}</span>
                        <span class="feature-tag">Temp: ${temp}°C</span>
                        <span class="feature-tag">Season: ${season}</span>
                        <span class="feature-tag">Model: ${pred.model_type || 'original'}</span>
                    </div>
                </div>
            `;
        }).join('');

        // Update load more button
        this.updateLoadMoreButton(filteredCount);
    }

    applyHistoryFilter(filter) {
        this.currentHistoryFilter = filter;
        this.showNotification(`Loading ${filter} predictions...`, 'info');
        
        // Reload history with filter
        this.loadFilteredHistory(filter, this.predictionLimit)
            .then(() => {
                this.updatePredictionHistory();
                this.showNotification(`Showing ${this.data.history.filtered_count} predictions from ${filter}`, 'success');
                
                // Update the filter count display
                this.updateKPIs();
            })
            .catch(error => {
                console.error('Filter error:', error);
                this.showNotification(`Could not load ${filter} predictions`, 'error');
            });
    }

    updateLoadMoreButton(filteredCount) {
        const loadMoreBtn = document.getElementById('loadMoreBtn');
        if (!loadMoreBtn) return;

        if (filteredCount <= this.predictionLimit) {
            loadMoreBtn.style.display = 'none';
        } else {
            loadMoreBtn.style.display = 'block';
            const remaining = filteredCount - this.predictionLimit;
            loadMoreBtn.innerHTML = `
                <i class="fas fa-plus"></i>
                Load More (${remaining} more ${remaining === 1 ? 'prediction' : 'predictions'})
            `;
        }
    }

    async loadMorePredictions() {
        const newLimit = this.predictionLimit + 10;
        try {
            await this.loadFilteredHistory(this.currentHistoryFilter, newLimit);
            this.predictionLimit = newLimit;
            this.updatePredictionHistory();
            this.showNotification(`Showing ${this.predictionLimit} predictions`, 'success');
        } catch (error) {
            console.error('Load more error:', error);
            this.showNotification('Could not load more predictions', 'error');
        }
    }

    showLoadingState() {
        const historyContainer = document.getElementById('predictionHistory');
        if (historyContainer) {
            // Store original content before overwriting
            if (!historyContainer.dataset.originalContent) {
                historyContainer.dataset.originalContent = historyContainer.innerHTML;
            }
            historyContainer.innerHTML = `
                <div class="loading-state">
                    <i class="fas fa-spinner fa-spin"></i>
                    <p>Loading predictions...</p>
                </div>
            `;
        }
        
        // For charts, DON'T overwrite the canvas elements - just show overlay
        const chartIds = ['featureImportanceChart', 'shapImportanceChart', 'performanceRadarChart', 'hourlyPatternChart'];
        chartIds.forEach(chartId => {
            const canvas = document.getElementById(chartId);
            if (canvas) {
                const parent = canvas.parentElement;
                if (parent) {
                    // Create loading overlay instead of replacing canvas
                    let overlay = parent.querySelector('.chart-loading-overlay');
                    if (!overlay) {
                        overlay = document.createElement('div');
                        overlay.className = 'chart-loading-overlay';
                        overlay.innerHTML = `
                            <div class="loading-state">
                                <i class="fas fa-spinner fa-spin fa-2x"></i>
                                <p>Loading chart...</p>
                            </div>
                        `;
                        parent.style.position = 'relative';
                        parent.appendChild(overlay);
                    }
                    overlay.style.display = 'flex';
                }
            }
        });
    }

    hideLoadingState() {
        // Remove loading overlays from charts
        const chartContainers = document.querySelectorAll('.chart-container');
        chartContainers.forEach(container => {
            const overlay = container.querySelector('.chart-loading-overlay');
            if (overlay) {
                overlay.style.display = 'none';
            }
        });
        
        // Restore original content if needed
        const historyContainer = document.getElementById('predictionHistory');
        if (historyContainer && historyContainer.dataset.originalContent) {
            // Don't restore here - let updatePredictionHistory handle it
            delete historyContainer.dataset.originalContent;
        }
    }

    async refreshDashboard() {
                // Disable refresh button and show loading
        const refreshBtn = document.getElementById('refreshBtn');
        if (refreshBtn) {
            refreshBtn.disabled = true;
            const originalHTML = refreshBtn.innerHTML;
            refreshBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Refreshing...';
            
            // Store original for restoration
            refreshBtn.dataset.originalHTML = originalHTML;
        }
        
        this.showLoadingState();
        this.showNotification('Refreshing dashboard...', 'info');
        
        try {
            // Reload all data with current filters
            await this.loadAllData();
            
            // Update charts with new data
            this.updateChartsWithFilteredData();
            
            // Update UI
            this.updateUI();
            
                        this.showNotification('Dashboard refreshed successfully!', 'success');
            
        } catch (error) {
            console.error('Refresh failed:', error);
            this.showNotification('Refresh failed. Showing previous data.', 'error');
            
            // Restore UI with existing data
            try {
                this.updateUI();
            } catch (uiError) {
                console.error('Could not update UI:', uiError);
            }
        } finally {
            // Re-enable refresh button
            if (refreshBtn) {
                refreshBtn.disabled = false;
                refreshBtn.innerHTML = refreshBtn.dataset.originalHTML || '<i class="fas fa-sync-alt"></i> Refresh';
                delete refreshBtn.dataset.originalHTML;
            }
            
            // Hide loading state
            this.hideLoadingState();
        }
    }

    // ========== REAL EMAIL FUNCTIONALITY ==========

    async sendAnalyticsEmail(emailData) {
        try {
            this.showNotification('Preparing email report...', 'info');
            
            // Validate email data
            const validation = this.validateEmailData(emailData);
            if (!validation.valid) {
                this.showNotification(validation.message, 'error');
                return { success: false, error: validation.message };
            }
            
            // Show sending state
            this.showNotification('Sending analytics report...', 'info');
            
            // Send to real API endpoint
            const response = await fetch(`${this.baseUrl}/api/send-analytics-email`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(emailData)
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showNotification(result.message || 'Email sent successfully!', 'success');
                return result;
            } else {
                this.showNotification(result.error || 'Failed to send email', 'error');
                return { success: false, error: result.error };
            }
            
        } catch (error) {
            console.error('Email sending error:', error);
            this.showNotification('Failed to send email. Please try again.', 'error');
            throw error;
        }
    }

    validateEmailData(emailData) {
        // Basic validation
        if (!emailData.recipients || !Array.isArray(emailData.recipients) || emailData.recipients.length === 0) {
            return { valid: false, message: 'Please specify at least one recipient' };
        }
        
        // Email format validation
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        for (const email of emailData.recipients) {
            if (!emailRegex.test(email)) {
                return { valid: false, message: `Invalid email address: ${email}` };
            }
        }
        
        if (!emailData.subject || emailData.subject.trim() === '') {
            return { valid: false, message: 'Email subject is required' };
        }
        
        if (emailData.subject.length > 200) {
            return { valid: false, message: 'Email subject is too long (max 200 characters)' };
        }
        
        return { valid: true, message: 'Email data is valid' };
    }

    // ========== REAL CHART EXPORT FUNCTIONALITY ==========

    async exportChartReal(chartId, format = 'png') {
        try {
            // Map chart IDs to backend chart types
            const chartTypeMap = {
                'featureImportance': 'feature_importance',
                'hourlyPattern': 'hourly_patterns',
                'performanceRadar': 'performance_radar'
            };
            
            const chartType = chartTypeMap[chartId];
            if (!chartType) {
                this.showNotification('Invalid chart type for export', 'error');
                return;
            }
            
            this.showNotification(`Exporting ${chartId} as ${format.toUpperCase()}...`, 'info');
            
            // Get current time filter
            const timeFilter = this.currentTimeFilter || 'all';
            
            // Call backend export endpoint
            const url = `${this.baseUrl}/api/export/chart/${chartType}?format=${format}&filter=${timeFilter}`;
            const response = await fetch(url);
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Export failed');
            }
            
            // Create download link
            const blob = await response.blob();
            const urlObject = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = urlObject;
            
            // Get filename from content-disposition header or generate one
            const contentDisposition = response.headers.get('content-disposition');
            let filename = `urbanflow_${chartType}_${new Date().toISOString().slice(0, 10)}.${format}`;
            
            if (contentDisposition) {
                const filenameMatch = contentDisposition.match(/filename="(.+)"/);
                if (filenameMatch) {
                    filename = filenameMatch[1];
                }
            }
            
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            window.URL.revokeObjectURL(urlObject);
            
            this.showNotification(`Chart exported as ${format.toUpperCase()}!`, 'success');
            
        } catch (error) {
            console.error('Chart export error:', error);
            
            // Fallback to client-side export if backend fails
            if (this.charts[chartId]) {
                this.exportChartClientSide(chartId, format);
            } else {
                this.showNotification(`Failed to export chart: ${error.message}`, 'error');
            }
        }
    }

    exportChartClientSide(chartId, format = 'png') {
        // Client-side export using Chart.js
        if (this.charts[chartId]) {
            const chart = this.charts[chartId];
            const link = document.createElement('a');
            
            if (format === 'png') {
                link.href = chart.toBase64Image();
                link.download = `${chartId}_${new Date().toISOString().slice(0, 10)}.png`;
                link.click();
                this.showNotification('Chart exported as PNG (client-side)', 'success');
            } else {
                this.showNotification('PDF/SVG export requires backend support', 'warning');
            }
        }
    }

    async exportAllCharts(format = 'zip') {
        try {
            this.showNotification('Preparing all charts for export...', 'info');
            
            const timeFilter = this.currentTimeFilter || 'all';
            const url = `${this.baseUrl}/api/export/all-charts?format=png&filter=${timeFilter}`;
            const response = await fetch(url);
            
            if (!response.ok) {
                throw new Error('Failed to export charts');
            }
            
            const blob = await response.blob();
            const urlObject = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = urlObject;
            link.download = `urbanflow_all_charts_${new Date().toISOString().slice(0, 10)}.zip`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            window.URL.revokeObjectURL(urlObject);
            
            this.showNotification('All charts exported as ZIP!', 'success');
            
        } catch (error) {
            console.error('All charts export error:', error);
            this.showNotification('Failed to export charts. Please try individual exports.', 'error');
        }
    }

    // ========== REAL DATA EXPORT FUNCTIONALITY ==========

    async exportPredictions(format) {
        try {
            this.showNotification(`Exporting predictions as ${format.toUpperCase()}...`, 'info');
            
            const validation = this.validateExportRequest('predictions', format);
            if (!validation.valid) {
                this.showNotification(validation.message, 'error');
                return;
            }
            
            // Get current filter
            const filter = this.currentHistoryFilter || 'all';
            
            let url;
            if (format === 'csv') {
                url = `${this.baseUrl}/export/predictions/csv?filter=${filter}`;
            } else if (format === 'json') {
                url = `${this.baseUrl}/export/predictions/json?filter=${filter}`;
            } else {
                this.showNotification(`Format ${format} not supported for predictions`, 'error');
                return;
            }
            
            const response = await fetch(url);
            
            if (!response.ok) {
                throw new Error('Export failed');
            }
            
            const blob = await response.blob();
            this.downloadBlob(blob, `predictions_${new Date().toISOString().slice(0, 10)}.${format}`);
            
            this.showNotification(`Predictions exported as ${format.toUpperCase()}!`, 'success');
            
        } catch (error) {
            console.error('Export error:', error);
            this.showNotification(`Failed to export predictions: ${error.message}`, 'error');
        }
    }

    async exportModelReport(format) {
        try {
            this.showNotification(`Exporting model report as ${format.toUpperCase()}...`, 'info');

            const validation = this.validateExportRequest('report', format);
            if (!validation.valid) {
                this.showNotification(validation.message, 'error');
                return;
            }

            if (format === 'html') {
                const url = `${this.baseUrl}/export/model-report`;
                const response = await fetch(url);

                if (!response.ok) {
                    throw new Error('Export failed');
                }

                const blob = await response.blob();
                this.downloadBlob(blob, `model_report_${new Date().toISOString().slice(0, 10)}.html`);
                this.showNotification('Model report exported as HTML!', 'success');

            } else if (format === 'pdf') {
                const response = await fetch(`${this.baseUrl}/export/metrics/pdf`);

                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    throw new Error(errorData.error || `Export failed with status ${response.status}`);
                }

                const contentType = response.headers.get('content-type') || '';
                if (contentType.includes('application/pdf')) {
                    const blob = await response.blob();
                    this.downloadBlob(blob, `urbanflow_metrics_report_${new Date().toISOString().slice(0, 10)}.pdf`);
                    this.showNotification('PDF report downloaded successfully!', 'success');
                } else {
                    // Server returned JSON instead of PDF (fallback)
                    const data = await response.json();
                    throw new Error(data.message || data.error || 'Server did not return a PDF file');
                }

            } else {
                this.showNotification(`Format ${format} not supported for model reports`, 'error');
            }

        } catch (error) {
            console.error('Report export error:', error);
            this.showNotification(`Failed to export report: ${error.message}`, 'error');
        }
    }

    async exportAllData() {
        try {
            this.showNotification('Preparing comprehensive data export...', 'info');
            
            // Use the new API endpoint for complete export
            const response = await fetch(`${this.baseUrl}/api/export/all?format=zip`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    filters: {
                        time_filter: this.currentTimeFilter
                    }
                })
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Export failed');
            }
            
            const blob = await response.blob();
            this.downloadBlob(blob, `urbanflow_complete_export_${new Date().toISOString().slice(0, 10)}.zip`);
            
            this.showNotification('Complete data export successful!', 'success');
            
        } catch (error) {
            console.error('Batch export error:', error);
            this.showNotification(`Batch export failed: ${error.message}`, 'error');
        }
    }

    validateExportRequest(exportType, format) {
        const validFormats = {
            'predictions': ['csv', 'json', 'excel'],
            'report': ['html', 'pdf'],
            'charts': ['png', 'pdf', 'svg', 'jpg', 'zip'],
            'all': ['zip']
        };
        
        if (!validFormats[exportType]) {
            return { valid: false, message: `Invalid export type: ${exportType}` };
        }
        
        if (!validFormats[exportType].includes(format)) {
            return { 
                valid: false, 
                message: `Invalid format for ${exportType}. Must be one of: ${validFormats[exportType].join(', ')}`
            };
        }
        
        // Check if there's data to export
        if (exportType === 'predictions' || exportType === 'all') {
            const historyCount = this.data.history?.filtered_count || 0;
            if (historyCount === 0) {
                return { valid: false, message: 'No prediction data available to export' };
            }
        }
        
        return { valid: true, message: 'Export request is valid' };
    }

    downloadBlob(blob, filename, contentType = null) {
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        
        if (contentType) {
            link.type = contentType;
        }
        
        link.style.display = 'none';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
    }

    showError(message, details = '') {
        console.error('Dashboard Error:', message, details);
        const fullMessage = details ? `${message} - ${details}` : message;
        showNotification(fullMessage, 'error');
    }

    showNotification(message, type = 'info') {
        showNotification(message, type);
    }
}

// ========== GLOBAL FUNCTIONS ==========

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    try {
        const dashboard = new AnalyticsDashboard();
        window.dashboard = dashboard;
        dashboard.init();
        
    } catch (error) {
        console.error('Failed to initialize dashboard:', error);
        showNotification('Failed to load analytics dashboard. Please check console.', 'error');
    }
});

// Global refresh function
function refreshDashboard() {
    analyticsDbg('Refresh button clicked');
    if (window.dashboard) {
        analyticsDbg('Dashboard instance found');
        window.dashboard.refreshDashboard();
    } else {
        analyticsDbg('No dashboard instance, reloading page');
        location.reload();
    }
}

// Global load more function
function loadMorePredictions() {
    if (window.dashboard) {
        window.dashboard.loadMorePredictions();
    }
}

// Global export functions - UPDATED TO USE REAL EXPORTS
async function exportPredictions(format) {
    if (window.dashboard) {
        await window.dashboard.exportPredictions(format);
    }
}

async function exportModelReport(format) {
    if (window.dashboard) {
        await window.dashboard.exportModelReport(format);
    }
}

async function exportAllData() {
    if (window.dashboard) {
        await window.dashboard.exportAllData();
    }
}

function exportChart(chartId) {
    if (window.dashboard) {
        window.dashboard.exportChartReal(chartId, 'png');
    }
}

function exportCharts() {
    if (window.dashboard) {
        window.dashboard.exportAllCharts('zip');
    }
}

// Apply time filter globally
function applyTimeFilter(timeRange) {
    if (window.dashboard) {
        window.dashboard.applyTimeFilter(timeRange);
    }
}

// Apply custom date range globally
function applyCustomDateRange() {
    if (window.dashboard) {
        window.dashboard.applyCustomRangeFromInputs();
    }
}

// Clear custom date range globally
function clearCustomDateRange() {
    if (window.dashboard) {
        window.dashboard.clearCustomDateRange();
    }
}

// Toggle chart type globally
function toggleChartType(chartId) {
    if (window.dashboard && window.dashboard.charts[chartId]) {
        const chart = window.dashboard.charts[chartId];
        const newType = chart.config.type === 'line' ? 'bar' : 'line';
        chart.config.type = newType;
        chart.update();
        showNotification(`Switched to ${newType} view`, 'success');
    }
}

// Send analytics email globally
async function sendAnalyticsEmail(emailData) {
    if (window.dashboard) {
        return await window.dashboard.sendAnalyticsEmail(emailData);
    }
}

// Export chart with specific format globally
async function exportChartWithFormat(chartId, format = 'png') {
    if (window.dashboard) {
        await window.dashboard.exportChartReal(chartId, format);
    }
}

// Notification system
function showNotification(message, type = 'info') {
    document.querySelectorAll('.notification').forEach(n => n.remove());

    const notification = document.createElement('div');
    notification.className = `notification ${type}`;

    const iconMap = {
        success: 'check-circle',
        error: 'exclamation-circle',
        warning: 'exclamation-triangle',
        info: 'info-circle',
    };

    notification.innerHTML = `
        <i class="fas fa-${iconMap[type] || 'info-circle'}"></i>
        <span>${message}</span>
    `;

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
            if (notification.parentNode) notification.remove();
        }, 300);
    }, 3000);
}

// ========== EMAIL MODAL FUNCTIONS ==========

function openEmailModal(context = 'full') {
    const modal = document.getElementById('emailModal');
    const subject = document.getElementById('emailSubject');
    const now = new Date();
    const dateStr = now.toISOString().split('T')[0];

    switch (context) {
        case 'charts':
            subject.value = `UrbanFlow Charts Report - ${dateStr}`;
            document.getElementById('includePDF').checked = false;
            document.getElementById('includeCSV').checked = false;
            document.getElementById('includeCharts').checked = true;
            break;
        case 'scheduled':
            subject.value = 'UrbanFlow Weekly Analytics Report';
            document.getElementById('includePDF').checked = true;
            document.getElementById('includeCSV').checked = true;
            document.getElementById('includeCharts').checked = true;
            document.getElementById('scheduleReport').checked = true;
            break;
        default:
            subject.value = `UrbanFlow Analytics Report - ${dateStr}`;
            document.getElementById('includePDF').checked = true;
            document.getElementById('includeCSV').checked = true;
            document.getElementById('includeCharts').checked = false;
            document.getElementById('scheduleReport').checked = false;
    }

    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeEmailModal() {
    const modal = document.getElementById('emailModal');
    modal.classList.remove('active');
    document.body.style.overflow = '';
    document.getElementById('recipientsError').style.display = 'none';
}

function validateEmail(email) {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

function validateEmails(emailString) {
    const emails = emailString.split(',').map(e => e.trim()).filter(e => e);
    return emails.length > 0 && emails.every(validateEmail);
}

async function sendAnalyticsEmailFromForm() {
    const recipients = document.getElementById('emailRecipients').value;
    const subject = document.getElementById('emailSubject').value;
    const message = document.getElementById('emailMessage').value;
    const includePDF = document.getElementById('includePDF').checked;
    const includeCSV = document.getElementById('includeCSV').checked;
    const includeCharts = document.getElementById('includeCharts').checked;
    const scheduleReport = document.getElementById('scheduleReport').checked;

    if (!validateEmails(recipients)) {
        document.getElementById('recipientsError').style.display = 'block';
        return;
    }
    document.getElementById('recipientsError').style.display = 'none';

    const sendBtn = document.querySelector('.email-btn-send');
    const originalText = sendBtn.innerHTML;
    sendBtn.disabled = true;
    sendBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...';

    try {
        if (window.dashboard && window.dashboard.sendAnalyticsEmail) {
            await window.dashboard.sendAnalyticsEmail({
                recipients: recipients.split(',').map(e => e.trim()),
                subject, message, includePDF, includeCSV, includeCharts,
                schedule: scheduleReport ? 'weekly' : 'once',
            });
            showNotification('Analytics report email sent successfully!', 'success');
            closeEmailModal();
        } else {
            setTimeout(() => {
                showNotification('Email sent successfully! (Demo mode)', 'success');
                closeEmailModal();
            }, 1500);
        }
    } catch (error) {
        showNotification('Failed to send email: ' + error.message, 'error');
    } finally {
        sendBtn.disabled = false;
        sendBtn.innerHTML = originalText;
    }
}

// ========== DATE RANGE MODAL FUNCTIONS ==========

function openDateRangeModal() {
    const modal = document.getElementById('dateRangeModal');
    const today = new Date().toISOString().split('T')[0];
    const weekAgo = new Date();
    weekAgo.setDate(weekAgo.getDate() - 7);

    document.getElementById('startDate').value = weekAgo.toISOString().split('T')[0];
    document.getElementById('endDate').value = today;
    document.getElementById('endDate').max = today;
    document.getElementById('startDate').max = today;

    updateDateRangeSummary();
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeDateRangeModal() {
    const modal = document.getElementById('dateRangeModal');
    modal.classList.remove('active');
    document.body.style.overflow = '';
}

function updateDateRangeSummary() {
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;
    const summary = document.getElementById('dateRangeSummary');
    if (!startDate || !endDate || !summary) return;

    const start = new Date(startDate);
    const end = new Date(endDate);
    const daysDiff = Math.ceil((end - start) / (1000 * 60 * 60 * 24)) + 1;

    summary.innerHTML = `
        <h4>Selected Range</h4>
        <p><strong>From:</strong> ${formatDateReadable(start)}</p>
        <p><strong>To:</strong> ${formatDateReadable(end)}</p>
        <p><strong>Duration:</strong> ${daysDiff} day${daysDiff !== 1 ? 's' : ''}</p>
        <p><i class="fas fa-info-circle"></i> Will analyze all predictions within this date range.</p>
    `;
}

function formatDateReadable(date) {
    return date.toLocaleDateString('en-US', {
        weekday: 'short', year: 'numeric', month: 'short', day: 'numeric',
    });
}

function applyPresetRange(days, months) {
    const endDate = new Date();
    const startDate = new Date();
    if (days) startDate.setDate(startDate.getDate() - days);
    else if (months) startDate.setMonth(startDate.getMonth() - months);

    document.getElementById('startDate').value = startDate.toISOString().split('T')[0];
    document.getElementById('endDate').value = endDate.toISOString().split('T')[0];
    updateDateRangeSummary();
}

async function applyCustomDateRangeFromModal() {
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;

    if (!startDate || !endDate) {
        showNotification('Please select both start and end dates', 'error');
        return;
    }

    const start = new Date(startDate);
    const end = new Date(endDate);
    const today = new Date();

    if (start > end) {
        showNotification('Start date must be before end date', 'error');
        return;
    }
    if (start > today || end > today) {
        showNotification('Dates cannot be in the future', 'error');
        return;
    }

    const daysDiff = Math.ceil((end - start) / (1000 * 60 * 60 * 24)) + 1;
    if (daysDiff > 365) {
        showNotification('Date range cannot exceed 365 days', 'error');
        return;
    }

    closeDateRangeModal();

    if (window.dashboard) {
        document.querySelectorAll('.time-btn').forEach(btn => {
            btn.classList.remove('active');
            if (btn.classList.contains('custom-range-btn')) btn.classList.add('active');
        });

        const customRangeInfo = document.getElementById('customRangeInfo');
        if (customRangeInfo) {
            customRangeInfo.innerHTML = `
                <div class="custom-range-info">
                    <i class="fas fa-calendar-check"></i>
                    ${startDate} to ${endDate} (${daysDiff} days)
                    <button class="btn-clear-range" onclick="clearCustomDateRange()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            `;
        }

        await window.dashboard.applyCustomDateRange(start, end);
        showNotification(`Applied custom date range: ${startDate} to ${endDate}`, 'success');
    }
}

async function exportLastWeek() {
    if (window.dashboard) {
        showNotification("Exporting last week's data...", 'info');
        await exportAllData();
    }
}

// ========== ANALYTICS PAGE INITIALIZATION ==========

document.addEventListener('DOMContentLoaded', () => {
    // Date range input listeners
    document.getElementById('startDate')?.addEventListener('change', updateDateRangeSummary);
    document.getElementById('endDate')?.addEventListener('change', updateDateRangeSummary);

    // Preset button listeners
    document.querySelectorAll('.preset-btn').forEach(btn => {
        btn.addEventListener('click', function () {
            const days = this.getAttribute('data-days');
            const months = this.getAttribute('data-months');
            applyPresetRange(days ? parseInt(days) : null, months ? parseInt(months) : null);
        });
    });

    // Close modals with Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            const emailModal = document.getElementById('emailModal');
            const dateModal = document.getElementById('dateRangeModal');
            if (emailModal && emailModal.classList.contains('active')) closeEmailModal();
            if (dateModal && dateModal.classList.contains('active')) closeDateRangeModal();
        }
    });

    // Close modals on backdrop click
    document.getElementById('emailModal')?.addEventListener('click', function (e) {
        if (e.target === this) closeEmailModal();
    });
    document.getElementById('dateRangeModal')?.addEventListener('click', function (e) {
        if (e.target === this) closeDateRangeModal();
    });
});
