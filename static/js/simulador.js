// Aguarda que o documento carregue para garantir que os elementos existem
document.addEventListener('DOMContentLoaded', () => {
    
    const formSimulador = document.getElementById('form-simulador');

    formSimulador.addEventListener('submit', async (e) => {
        // Impede que a página recarregue ao submeter o formulário
        e.preventDefault();

        // 1. Capturar todos os dados inseridos pelo utilizador
        const payload = {
            hora: document.getElementById('hora').value,
            dia_semana: document.getElementById('dia_semana').value,
            is_horario_pico: document.getElementById('is_horario_pico').value,
            temp_externa: document.getElementById('temp_externa').value,
            incidencia_solar: document.getElementById('incidencia_solar').value,
            lotacao: document.getElementById('lotacao').value,
            portas_abertas: document.getElementById('portas_abertas').value,
            velocidade_kmh: document.getElementById('velocidade_kmh').value,
            temp_interna_atual: document.getElementById('temp_interna_atual').value,
            potencia_real_kw: document.getElementById('potencia_real_kw').value
        };

        // Alterar o texto do botão para dar feedback visual enquanto a IA processa
        const btnSubmit = formSimulador.querySelector('button[type="submit"]');
        const textoOriginal = btnSubmit.innerText;
        btnSubmit.innerText = 'A processar...';
        btnSubmit.disabled = true;

        try {
            // 2. Enviar os dados para o Backend (Flask)
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json' 
                },
                body: JSON.stringify(payload)
            });

            const result = await response.json();

            // 3. Atualizar a Interface (UI) com a resposta da IA
            if (result.status === 'success') {
                
                // Atualiza os valores numéricos dos resultados
                document.getElementById('result-potencia').innerText = result.potencia_esperada.toFixed(2) + ' kW';
                document.getElementById('result-falha').innerText = result.probabilidade_falha.toFixed(1) + '%';

                // Lógica de alerta visual para o card de Manutenção
                const cardManutencao = document.getElementById('card-manutencao');
                const statusText = document.getElementById('status-text');

                // Se a probabilidade for superior ao limite de segurança (25%)
                if (result.alerta_manutencao) {
                    cardManutencao.classList.add('alert');
                    statusText.innerText = '⚠️ NECESSITA MANUTENÇÃO';
                    statusText.style.color = '#ff4b2b'; // Vermelho de alerta
                } else {
                    cardManutencao.classList.remove('alert');
                    statusText.innerText = '✔️ OPERAÇÃO NORMAL';
                    statusText.style.color = '#00f2fe'; // Azul/Verde de sucesso
                }
            } else {
                alert("Erro da API: " + result.message);
            }

        } catch (error) {
            console.error("Erro na inferência da IA:", error);
            alert("Erro ao conectar com a IA. Verifica se o backend (app.py) está a correr.");
        } finally {
            // Restaurar o botão ao estado normal após terminar
            btnSubmit.innerText = textoOriginal;
            btnSubmit.disabled = false;
        }
    });
});