// Configurações globais do Chart.js
Chart.defaults.color = '#a0aabf';
Chart.defaults.font.family = 'Poppins';

// Inicialização do Gráfico de Resíduo
let currentData = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]; 
let labels = ['-9m', '-8m', '-7m', '-6m', '-5m', '-4m', '-3m', '-2m', '-1m', 'Agora'];

const ctx = document.getElementById('residuoChart').getContext('2d');
const residuoChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: labels,
        datasets: [{
            label: 'Resíduo (kW)',
            data: currentData,
            borderColor: '#00f2fe',
            backgroundColor: 'rgba(0, 242, 254, 0.1)',
            borderWidth: 3,
            fill: true,
            tension: 0.4,
            pointBackgroundColor: '#ffffff',
            pointRadius: 4
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
            x: { grid: { color: 'rgba(255, 255, 255, 0.05)' } },
            y: {
                grid: { color: 'rgba(255, 255, 255, 0.05)' },
                suggestedMin: 0,
                suggestedMax: 3,
                title: { display: true, text: 'Diferença em kW', color: '#a0aabf' }
            }
        }
    }
});

// Função para procurar dados reais no Backend
async function atualizarStatus() {
    try {
        const response = await fetch('/api/status_atual');
        const data = await response.json();

        // Se o banco estiver vazio, não tenta atualizar para não dar erro
        if (data.error) {
            console.log("Aguardando dados do sensor...");
            return;
        }

        // Atualizar KPIs Físicos usando IDs de forma segura
        document.getElementById('valor-lotacao').innerText = data.lotacao + ' pass.';
        document.getElementById('valor-temp').innerText = data.temp_externa + '°C';
        document.getElementById('consumo-real').innerText = data.consumo_real.toFixed(1) + ' kW';

        // Atualizar o Gráfico
        currentData.shift(); // Remove o dado mais antigo
        currentData.push(data.residuo); // Insere o novo resíduo
        residuoChart.update();

        // Lógica de Alerta Visual de Manutenção
        const cardManutencao = document.getElementById('card-manutencao');
        const badgeFalha = document.getElementById('badge-falha');

        if (data.alerta_manutencao) {
            cardManutencao.classList.add('alert');
            badgeFalha.className = 'status-badge danger';
            badgeFalha.innerText = '⚠️ ALERTA MECÂNICO (>25%)';
        } else {
            cardManutencao.classList.remove('alert');
            badgeFalha.className = 'status-badge success';
            badgeFalha.innerText = '✔️ Sistema Saudável';
        }

    } catch (error) {
        console.error("Erro ao buscar dados de status:", error);
    }
}

// Executar a função a cada 2.5 segundos para simular tempo real
setInterval(atualizarStatus, 2500);
// Executar logo no arranque da página
atualizarStatus();