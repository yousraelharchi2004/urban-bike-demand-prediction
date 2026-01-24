// Feature Engineering Dashboard JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Load model comparison data
    fetch('/api/model-comparison')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Error loading comparison:', data.error);
                return;
            }
            
            // Update comparison metrics
            document.getElementById('original-r2').textContent = data.original.r2.toFixed(3);
            document.getElementById('original-rmse').textContent = data.original.rmse.toFixed(1);
            document.getElementById('original-features').textContent = data.original.features;
            
            document.getElementById('advanced-r2').textContent = data.advanced.r2.toFixed(3);
            document.getElementById('advanced-rmse').textContent = data.advanced.rmse.toFixed(1);
            document.getElementById('advanced-features').textContent = data.advanced.features;
            
            // Calculate and display improvement
            const r2Improvement = ((data.advanced.r2 - data.original.r2) / data.original.r2 * 100).toFixed(1);
            document.getElementById('improvement-text').textContent = `+${r2Improvement}% improvement`;
        })
        .catch(error => console.error('Error fetching comparison:', error));
    
    // Load feature importance data
    fetch('/api/advanced-model-info')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Error loading model info:', data.error);
                return;
            }
            
            // Create feature importance chart
            createFeatureImportanceChart(data.top_features);
            
            // Display engineered features
            displayEngineeredFeatures(data.engineered_features);
        })
        .catch(error => console.error('Error fetching model info:', error));
});

function createFeatureImportanceChart(features) {
    // Sort features by importance
    features.sort((a, b) => b.importance - a.importance);
    
    // Take top 15 features
    const topFeatures = features.slice(0, 15);
    
    // Prepare data for chart
    const labels = topFeatures.map(feat => {
        // Shorten long feature names
        const name = feat.feature;
        if (name.length > 20) {
            return name.substring(0, 17) + '...';
        }
        return name;
    });
    
    const importanceValues = topFeatures.map(feat => feat.importance * 100);
    
    // Determine colors (engineered vs original)
    const colors = topFeatures.map(feat => {
        const name = feat.feature.toLowerCase();
        if (name.includes('_squared') || name.includes('_sin') || name.includes('_cos') || 
            name.includes('interaction') || name.includes('is_') || name.includes('temp_') ||
            name.includes('comfort_') || name.includes('rush_') || name.includes('good_') ||
            name.includes('time_of_day_') || name.includes('season_') || name.includes('temp_category_')) {
            return '#4CAF50'; // Green for engineered
        }
        return '#2196F3'; // Blue for original
    });
    
    // Create chart
    const ctx = document.getElementById('featureImportanceChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Feature Importance (%)',
                data: importanceValues,
                backgroundColor: colors,
                borderColor: colors.map(color => color.replace('0.8', '1')),
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const featureName = topFeatures[context.dataIndex].feature;
                            const importance = context.parsed.x.toFixed(1);
                            return `${featureName}: ${importance}%`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Importance (%)'
                    }
                }
            }
        }
    });
}

function displayEngineeredFeatures(features) {
    const grid = document.getElementById('engineered-features-grid');
    
    features.forEach(feat => {
        const featureCard = document.createElement('div');
        featureCard.className = 'feature-card';
        
        // Determine feature category
        let category = 'Other';
        let icon = '🔧';
        
        if (feat.name.includes('temp_') && feat.name.includes('interaction')) {
            category = 'Weather Interaction';
            icon = '🌡️🕒';
        } else if (feat.name.includes('_squared')) {
            category = 'Non-linear Effect';
            icon = '📈';
        } else if (feat.name.includes('_sin') || feat.name.includes('_cos')) {
            category = 'Cyclical Time';
            icon = '🔄';
        } else if (feat.name.includes('is_')) {
            category = 'Binary Indicator';
            icon = '✅';
        } else if (feat.name.includes('season_')) {
            category = 'Seasonal';
            icon = '🍂❄️🌸☀️';
        } else if (feat.name.includes('time_of_day_')) {
            category = 'Time of Day';
            icon = '🕐';
        } else if (feat.name.includes('comfort_') || feat.name.includes('good_')) {
            category = 'Comfort Index';
            icon = '😊';
        } else if (feat.name.includes('rush_')) {
            category = 'Traffic Intensity';
            icon = '🚗';
        }
        
        featureCard.innerHTML = `
            <div class="feature-icon">${icon}</div>
            <div class="feature-name">${feat.name}</div>
            <div class="feature-importance">${feat.importance.toFixed(1)}%</div>
            <div class="feature-category">${category}</div>
        `;
        
        grid.appendChild(featureCard);
    });
}

// Add some CSS styles dynamically
const style = document.createElement('style');
style.textContent = `
.features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 15px;
    margin-top: 20px;
}

.feature-card {
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 15px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}

.feature-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.feature-icon {
    font-size: 24px;
    margin-bottom: 10px;
}

.feature-name {
    font-weight: bold;
    margin-bottom: 5px;
    word-break: break-word;
}

.feature-importance {
    font-size: 18px;
    font-weight: bold;
    color: #4CAF50;
    margin-bottom: 5px;
}

.feature-category {
    font-size: 12px;
    color: #666;
    background: #f5f5f5;
    padding: 3px 8px;
    border-radius: 12px;
    display: inline-block;
}

.comparison-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 20px 0;
}

.model-card {
    flex: 1;
    padding: 20px;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    text-align: center;
}

.model-card.improved {
    border-color: #4CAF50;
    background: #f0f9f0;
}

.improvement-arrow {
    padding: 0 20px;
    text-align: center;
}

.improvement-text {
    font-weight: bold;
    color: #4CAF50;
    margin-top: 10px;
}

.metric {
    margin: 10px 0;
}

.metric-label {
    font-size: 14px;
    color: #666;
}

.metric-value {
    font-size: 24px;
    font-weight: bold;
}

.metric-value.improved-value {
    color: #4CAF50;
}

.categories-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    margin-top: 20px;
}

.category-card {
    background: #f9f9f9;
    border-left: 4px solid #4CAF50;
    padding: 15px;
    border-radius: 4px;
}

.category-card ul {
    padding-left: 20px;
    margin-top: 10px;
}

.category-card li {
    margin-bottom: 8px;
}

.code-section {
    background: #f5f5f5;
    padding: 15px;
    border-radius: 4px;
    margin-top: 20px;
    overflow-x: auto;
}

.code-section pre {
    margin: 0;
}

.legend {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-top: 20px;
}

.legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
}

.legend-item.original:before {
    content: '';
    display: inline-block;
    width: 15px;
    height: 15px;
    background: #2196F3;
}

.legend-item.engineered:before {
    content: '';
    display: inline-block;
    width: 15px;
    height: 15px;
    background: #4CAF50;
}

.chart-container {
    position: relative;
    height: 500px;
    width: 100%;
}
`;
document.head.appendChild(style);