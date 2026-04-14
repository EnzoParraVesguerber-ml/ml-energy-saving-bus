import os
import logging
import random
from flask import Flask, render_template, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import joblib
import pandas as pd
import numpy as np

# 1. Configuração de Logs (Substitui os prints e não vaza erros para o usuário)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# 2. Segurança: Rate Limiting para proteger a API de inferência contra abusos (ex: DDoS)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["500 per day", "100 per hour"],
    storage_uri="memory://"
)

# 3. Configuração de caminhos e Carregamento de Modelos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_REG_PATH = os.path.join(BASE_DIR, 'models', 'hvac_regressor_gb.pkl')
MODEL_CLF_PATH = os.path.join(BASE_DIR, 'models', 'modelo_manutencao_gradient_boosting.pkl')

try:
    model_reg = joblib.load(MODEL_REG_PATH)
    model_clf = joblib.load(MODEL_CLF_PATH)
    logging.info("Modelos de Machine Learning carregados com sucesso.")
except Exception as e:
    logging.error(f"Erro crítico ao carregar modelos: {e}")

@app.route('/')
def index():
    return render_template('status.html')

@app.route('/simulador')
def simulador():
    return render_template('simulador.html')

@app.route('/economia')
def economia():
    return render_template('economia.html')

@app.route('/api/predict', methods=['POST'])
@limiter.limit("1 per second")
def predict():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'status': 'error', 'message': 'Nenhum dado fornecido no payload JSON.'}), 400

        # Validação de Inputs Obrigatórios
        campos_obrigatorios = ['temp_externa', 'lotacao', 'incidencia_solar', 'portas_abertas', 'potencia_real_kw', 'historico_residuos']
        for campo in campos_obrigatorios:
            if campo not in data:
                return jsonify({'status': 'error', 'message': f'Campo ausente: {campo}'}), 400

        # Validação de Tipagem para evitar injeção ou falha no Pandas
        input_data = pd.DataFrame([{
            'temp_externa': float(data['temp_externa']),
            'lotacao': int(data['lotacao']),
            'incidencia_solar': float(data['incidencia_solar']),
            'portas_abertas': int(data['portas_abertas'])
        }])
        
        potencia_real = float(data['potencia_real_kw'])
        
        # Recebendo o histórico do Frontend (Tornando a API Stateless e Segura)
        # O Frontend deve enviar um array com os últimos 14 resíduos
        historico = data.get('historico_residuos', [])
        if not isinstance(historico, list):
            return jsonify({'status': 'error', 'message': 'historico_residuos deve ser uma lista (array).'}), 400

        # 1. Predição da Potência Esperada (O que a física exige)
        potencia_esperada = model_reg.predict(input_data)[0]
        
        # 2. Cálculo do Resíduo Atual
        residuo_atual = potencia_real - potencia_esperada
        
        # 3. Atualização da Média Móvel (Backend calcula apenas para a requisição atual)
        historico.append(residuo_atual)
        if len(historico) > 15:
            historico = historico[-15:] # Mantém apenas os últimos 15 registros
            
        residuo_media_15m = np.mean(historico)
        
        # 4. Predição de Falha (Classificação)
        input_clf = input_data.copy()
        input_clf['residuo'] = residuo_atual
        input_clf['residuo_media_15m'] = residuo_media_15m
        
        prob_falha = model_clf.predict_proba(input_clf)[0][1]
        
        return jsonify({
            'status': 'success',
            'potencia_esperada': round(potencia_esperada, 3),
            'potencia_real': round(potencia_real, 3),
            'residuo_atual': round(residuo_atual, 3),
            'residuo_media': round(residuo_media_15m, 3),
            'novo_historico_residuos': [round(r, 3) for r in historico], # Devolve ao JS para a próxima chamada
            'probabilidade_falha': round(prob_falha * 100, 2),
            'alerta_manutencao': bool(prob_falha > 0.25)
        })

    except ValueError as ve:
        # Erros de conversão de tipo (ex: tentou converter string para float)
        logging.warning(f"Erro de validação de dados: {ve}")
        return jsonify({'status': 'error', 'message': 'Dados em formato inválido.'}), 400
    except Exception as e:
        # Erros inesperados genéricos (não vaza stacktrace pro cliente)
        logging.error(f"Erro interno no servidor: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Ocorreu um erro interno ao processar a predição.'}), 500

@app.route('/api/status_atual')
def status_atual():
    """ Simula a telemetria em tempo real do autocarro """
    consumo_real = round(12.0 + random.uniform(-1.0, 1.5), 2)
    residuo = round(random.uniform(0.1, 1.8), 2)
    alerta = residuo > 1.5

    return jsonify({
        'lotacao': random.randint(20, 65),
        'temp_externa': round(random.uniform(25.0, 35.0), 1),
        'consumo_real': consumo_real,
        'residuo': residuo,
        'alerta_manutencao': alerta
    })

@app.route('/api/kpis_economia')
def kpis_economia():
    """ Retorna os dados agregados para o dashboard financeiro/ambiental """
    return jsonify({
        'kpi_kwh': 14.5,
        'kpi_money': 12.32,
        'kpi_co2': 1.3,
        'chart_diario': {
            'labels': ['06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00'],
            'sem_ia': [15, 18, 22, 25, 28, 30, 32, 33, 31, 28, 24, 20, 16],
            'com_ia': [12, 14, 16, 18, 20, 22, 25, 26, 24, 21, 18, 15, 12]
        },
        'chart_semanal': [65, 70, 80, 55, 90, 40, 35],
        'chart_distribuicao': [40, 45, 15]
    })

if __name__ == '__main__':
    # Uso de variáveis de ambiente para decidir o modo debug
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() in ['true', '1', 't']
    app.run(debug=debug_mode, port=5000)