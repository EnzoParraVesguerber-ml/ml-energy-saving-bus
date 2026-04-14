// Variável para manter o estado no Frontend (Stateless Backend)
let historicoResiduos = [];

async function executarInferenciaIA() {
    // 1. Recolher os valores digitados pelo utilizador e incluir o histórico
    const payload = {
        temp_externa: parseFloat(document.getElementById('hvac-temp-ext').value),
        lotacao: parseInt(document.getElementById('hvac-lotacao').value),
        incidencia_solar: parseFloat(document.getElementById('hvac-solar').value),
        portas_abertas: parseInt(document.getElementById('hvac-portas').value),
        potencia_real_kw: parseFloat(document.getElementById('manut-consumo').value),
        historico_residuos: historicoResiduos // Adicionada a chave exigida pela nova API
    };

    try {
        // 2. Fazer o pedido POST para o servidor Flask
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (data.status === 'success') {
            // Atualizar o histórico local com o retorno da API para a próxima chamada
            historicoResiduos = data.novo_historico_residuos;

            // 3. Atualizar o Card 1 (IA Otimizadora)
            const boxHvac = document.getElementById('resultado-hvac');
            boxHvac.style.opacity = 0;
            setTimeout(() => {
                boxHvac.innerText = data.potencia_esperada.toFixed(2) + ' kW';
                boxHvac.style.opacity = 1;
            }, 200);

            // 4. Preencher os inputs de Resíduo e Média Móvel (agora lendo 'residuo_atual')
            document.getElementById('manut-residuo').value = data.residuo_atual.toFixed(2);
            document.getElementById('manut-media15m').value = data.residuo_media.toFixed(2);

            // 5. Atualizar o Card 2 (Manutenção Preditiva)
            const boxResultManut = document.getElementById('resultado-manut');
            const boxProb = document.getElementById('probabilidade-manut');
            
            boxResultManut.style.opacity = 0;
            setTimeout(() => {
                if (data.alerta_manutencao) {
                    boxResultManut.innerText = '⚠️ FALHA MECÂNICA';
                    boxResultManut.className = 'result-value text-red';
                } else {
                    boxResultManut.innerText = '✔️ SISTEMA SAUDÁVEL';
                    boxResultManut.className = 'result-value text-green';
                }
                boxProb.innerText = `Probabilidade de Falha: ${data.probabilidade_falha.toFixed(1)}%`;
                boxResultManut.style.opacity = 1;
            }, 200);

        } else {
            alert('Erro na API: ' + data.message);
        }

    } catch (error) {
        console.error('Erro de ligação:', error);
        alert('Erro ao ligar ao servidor Flask. Verifique o console.');
    }
}