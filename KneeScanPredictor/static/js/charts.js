// Functions for handling charts on the results page

document.addEventListener('DOMContentLoaded', function() {
    // If there's a knee health comparison chart canvas, initialize it
    const kneeHealthChartCanvas = document.getElementById('kneeHealthChart');
    
    if (kneeHealthChartCanvas) {
        // Get data from data attributes
        const userScore = parseFloat(kneeHealthChartCanvas.dataset.userScore || 0);
        
        // Calculate values for patient (derived from knee health score)
        // This is a simplified example - real values would come from model
        const userJointSpace = Math.max(0, userScore - 10 + getRandomInt(-5, 5));
        const userBoneDensity = Math.max(0, userScore - 5 + getRandomInt(-10, 10));
        const userCartilage = Math.max(0, userScore + 5 + getRandomInt(-15, 5));
        
        // Create chart
        new Chart(kneeHealthChartCanvas, {
            type: 'bar',
            data: {
                labels: ['Joint Space', 'Bone Density', 'Cartilage', 'Overall'],
                datasets: [
                    {
                        label: 'Your Knee',
                        data: [userJointSpace, userBoneDensity, userCartilage, userScore],
                        backgroundColor: '#3498db',
                        borderColor: '#2980b9',
                        borderWidth: 1
                    },
                    {
                        label: 'Normal Healthy Knee',
                        data: [85, 90, 88, 90],
                        backgroundColor: '#2ecc71',
                        borderColor: '#27ae60',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Health Score (%)'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Knee Health Comparison',
                        font: {
                            size: 16
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                label += Math.round(context.raw * 10) / 10 + '%';
                                return label;
                            }
                        }
                    }
                }
            }
        });
    }
    
    // If there's a progress history chart canvas, initialize it
    const progressHistoryCanvas = document.getElementById('progressHistoryChart');
    
    if (progressHistoryCanvas && progressHistoryCanvas.dataset.history) {
        try {
            // Parse the JSON data from data attribute
            const historyData = JSON.parse(progressHistoryCanvas.dataset.history);
            
            // Extract dates and scores
            const dates = historyData.map(item => new Date(item.date).toLocaleDateString());
            const scores = historyData.map(item => item.score);
            
            // Create chart
            new Chart(progressHistoryCanvas, {
                type: 'line',
                data: {
                    labels: dates,
                    datasets: [{
                        label: 'Knee Health Score',
                        data: scores,
                        backgroundColor: 'rgba(52, 152, 219, 0.2)',
                        borderColor: 'rgba(52, 152, 219, 1)',
                        borderWidth: 2,
                        tension: 0.2,
                        pointBackgroundColor: 'rgba(52, 152, 219, 1)',
                        pointRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: false,
                            min: Math.max(0, Math.min(...scores) - 10),
                            max: 100,
                            title: {
                                display: true,
                                text: 'Health Score (%)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Scan Date'
                            }
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'Knee Health Progress Over Time',
                            font: {
                                size: 16
                            }
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Error creating progress chart:', error);
        }
    }
    
    // If there's a confidence gauge, initialize it
    const confidenceGaugeCanvas = document.getElementById('confidenceGauge');
    
    if (confidenceGaugeCanvas) {
        const confidenceValue = parseFloat(confidenceGaugeCanvas.dataset.confidence || 0);
        
        new Chart(confidenceGaugeCanvas, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [confidenceValue * 100, (1 - confidenceValue) * 100],
                    backgroundColor: [
                        'rgba(46, 204, 113, 0.8)',
                        'rgba(236, 240, 241, 0.5)'
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                circumference: 180,
                rotation: 270,
                cutout: '75%',
                plugins: {
                    tooltip: {
                        enabled: false
                    },
                    legend: {
                        display: false
                    }
                }
            }
        });
        
        // Add the confidence text in the center
        const confidenceTextElement = document.getElementById('confidenceText');
        if (confidenceTextElement) {
            confidenceTextElement.innerText = `${Math.round(confidenceValue * 1000) / 10}%`;
        }
    }
});

// Helper function to get random integer between min and max (inclusive)
function getRandomInt(min, max) {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min + 1)) + min;
}
