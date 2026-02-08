document.getElementById('predictionForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    // Show loading state
    const submitButton = document.querySelector('button[type="submit"]');
    const originalText = submitButton.textContent;
    submitButton.textContent = 'Predicting...';
    submitButton.disabled = true;

    // Hide previous results and errors
    document.getElementById('results').style.display = 'none';
    document.getElementById('error').style.display = 'none';

    // Collect form data
    const formData = new FormData(event.target);
    const data = {};
    for (let [key, value] of formData.entries()) {
        data[key] = parseFloat(value);
    }

    try {
        // Send POST request to /predict
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        const result = await response.json();

        if (response.ok) {
            // Display results
            displayResults(result);
        } else {
            // Display error
            displayError(result.error || 'An error occurred');
        }
    } catch (error) {
        displayError('Network error: ' + error.message);
    } finally {
        // Reset button
        submitButton.textContent = originalText;
        submitButton.disabled = false;
    }
});

function displayResults(result) {
    const resultsDiv = document.getElementById('results');
    const failureProbDiv = document.getElementById('failureProbability');
    const riskFactorsDiv = document.getElementById('riskFactors');

    // Failure Probability
    const probPercent = (result.failure_probability * 100).toFixed(2);
    failureProbDiv.innerHTML = `
        <h3>Failure Probability: ${probPercent}%</h3>
        <div class="probability-bar">
            <div class="probability-fill" style="width: ${probPercent}%"></div>
        </div>
        <p>Risk Level: ${getRiskLevel(result.failure_probability)}</p>
    `;

    // Create Failure Probability Chart
    createFailureChart(result.failure_probability);

    // Top Risk Factors
    let riskHtml = '<h3>Top Risk Factors:</h3><ul>';
    for (const [feature, shapValue] of Object.entries(result.top_risk_factors)) {
        const direction = shapValue > 0 ? 'increases' : 'decreases';
        const absValue = Math.abs(shapValue).toFixed(4);
        riskHtml += `<li><strong>${feature}</strong>: ${direction} risk by ${absValue}</li>`;
    }
    riskHtml += '</ul>';
    riskFactorsDiv.innerHTML = riskHtml;

    // Create Risk Factors Chart
    createRiskChart(result.top_risk_factors);

    resultsDiv.style.display = 'block';
}

function createFailureChart(probability) {
    const ctx = document.getElementById('failureChart').getContext('2d');
    const probPercent = probability * 100;

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Failure Risk', 'Safe'],
            datasets: [{
                data: [probPercent, 100 - probPercent],
                backgroundColor: ['#dc3545', '#28a745'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom',
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.label + ': ' + context.parsed.toFixed(2) + '%';
                        }
                    }
                }
            }
        }
    });
}

function createRiskChart(riskFactors) {
    const ctx = document.getElementById('riskChart').getContext('2d');
    const features = Object.keys(riskFactors);
    const values = Object.values(riskFactors).map(Math.abs);

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: features,
            datasets: [{
                label: 'SHAP Value (Absolute)',
                data: values,
                backgroundColor: '#007bff',
                borderColor: '#0056b3',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Impact on Failure Risk'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Features'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

function displayError(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.innerHTML = `<p><strong>Error:</strong> ${message}</p>`;
    errorDiv.style.display = 'block';
}

function getRiskLevel(probability) {
    if (probability < 0.2) return 'Low';
    if (probability < 0.5) return 'Medium';
    if (probability < 0.8) return 'High';
    return 'Critical';
}
