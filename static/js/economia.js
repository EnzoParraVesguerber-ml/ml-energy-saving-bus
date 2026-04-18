Chart.defaults.color = '#a0aabf';
Chart.defaults.font.family = 'Poppins';

async function carregarDashboardEconomia() {
    try {
        const response = await fetch('/api/kpis_economia');
        const data = await response.json();

        // 1. Preencher os KPIs Superiores
        document.getElementById('kpi-kwh').innerText = data.kpi_kwh.toFixed(1) + ' kWh';
        document.getElementById('kpi-money').innerText = 'R$ ' + data.kpi_money.toFixed(2).replace('.', ',');
        document.getElementById('kpi-co2').innerText = data.kpi_co2.toFixed(2) + ' kg';

        // 2. Renderizar Gráfico Principal (Linha)
        const ctxLine = document.getElementById('economiaChart').getContext('2d');
        new Chart(ctxLine, {
            type: 'line',
            data: {
                labels: data.chart_diario.labels,
                datasets: [
                    {
                        label: 'Consumo Padrão (Sem IA)',
                        data: data.chart_diario.sem_ia,
                        borderColor: '#ff0844', 
                        backgroundColor: 'rgba(255, 8, 68, 0.1)',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        fill: true,
                        tension: 0.4
                    },
                    {
                        label: 'Consumo Inteligente (Com IA)',
                        data: data.chart_diario.com_ia,
                        borderColor: '#00f260',
                        backgroundColor: 'rgba(0, 242, 96, 0.2)',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { position: 'top' } },
                scales: {
                    x: { grid: { color: 'rgba(255, 255, 255, 0.05)' } },
                    y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, beginAtZero: true }
                }
            }
        });

        // 3. Renderizar Gráfico de Barras (Semanal) Dinâmico
        const ctxBar = document.getElementById('barChart').getContext('2d');
        new Chart(ctxBar, {
            type: 'bar',
            data: {
                labels: data.chart_semanal.labels, // Lendo os dias dinâmicos da API
                datasets: [{
                    label: 'kWh Poupados',
                    data: data.chart_semanal.data, // Lendo os valores dinâmicos da API
                    backgroundColor: '#4facfe',
                    borderRadius: 5
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { display: false } },
                    y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, beginAtZero: true }
                }
            }
        });

        // 4. Renderizar Gráfico de Rosca (Distribuição)
        const ctxDoughnut = document.getElementById('doughnutChart').getContext('2d');
        new Chart(ctxDoughnut, {
            type: 'doughnut',
            data: {
                labels: ['Desligado/Ventilação', 'Frio Leve (< 50%)', 'Potência Alta (> 80%)'],
                datasets: [{
                    data: data.chart_distribuicao,
                    backgroundColor: ['#00f260', '#00f2fe', '#ff0844'],
                    borderWidth: 0,
                    hoverOffset: 4
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom', labels: { boxWidth: 12 } }
                },
                cutout: '70%'
            }
        });

    } catch (error) {
        console.error("Erro ao carregar dados financeiros:", error);
    }
}

// Iniciar a construção do dashboard assim que o script for carregado
carregarDashboardEconomia();