// Lens Benchmark v2.2 Site Scripts
document.addEventListener('DOMContentLoaded', function() {
    loadLeaderboardData();
    initializeInteractions();
});

async function loadLeaderboardData() {
    try {
        const response = await fetch('data/leaderboard.json');
        const data = await response.json();
        renderLeaderboard(data.systems);
    } catch (error) {
        console.error('Failed to load leaderboard data:', error);
        showError('Failed to load leaderboard data');
    }
}

function renderLeaderboard(systems) {
    const tbody = document.querySelector('#results-table tbody');
    if (!tbody) return;
    
    tbody.innerHTML = systems.map((system, index) => {
        const statusIcon = system.sla_compliant && system.calibrated ? '✅' : '⚠️';
        return `
            <tr>
                <td>${system.rank}</td>
                <td>${statusIcon} ${system.display_name}</td>
                <td><strong>${system.ndcg_at_10.toFixed(4)}</strong></td>
                <td>±${system.ci_width.toFixed(4)}</td>
                <td>${system.ece.toFixed(4)}</td>
                <td>${system.tail_ratio.toFixed(2)}</td>
            </tr>
        `;
    }).join('');
}

function initializeInteractions() {
    // Add interactive features
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', function() {
            // Remove active class from siblings
            this.parentNode.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            // Add active class to clicked tab
            this.classList.add('active');
        });
    });
}

function showError(message) {
    const container = document.querySelector('#results-table tbody') || document.body;
    container.innerHTML = `<div class="error-message">❌ ${message}</div>`;
}