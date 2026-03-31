from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Configuração de caminhos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_REG_PATH = os.path.join(BASE_DIR, 'models', 'hvac_regressor_gb.pkl')
MODEL_CLF_PATH = os.path.join(BASE_DIR, 'models', 'modelo_manutencao_gradient_boosting.pkl')

# Carregar modelos ao iniciar o servidor
model_reg = joblib.load(MODEL_REG_PATH)
model_clf = joblib.load(MODEL_CLF_PATH)

# Cache simples para calcular a média móvel no backend (últimas 15 leituras)
historico_residuos = []

@app.route('/')
def index():
    # Rota principal para o dashboard de status
    return render_template('status.html')

@app.route('/simulador')
def simulador():
    return render_template('simulador.html')

@app.route('/economia')
def economia():
    return render_template('economia.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # 1. Preparar dados para o modelo de Regressão
        # Chaves do JSON agora batem com as features do DataFrame de treino
        input_data = pd.DataFrame([{
            'temp_externa': data['temp_externa'],
            'lotacao': data['lotacao'],
            'incidencia_solar': data['incidencia_solar'],
            'portas_abertas': data['portas_abertas']  # Correção: porta_aberta -> portas_abertas
        }])
        
        # 2. Predição de Consumo Esperado
        consumo_esperado = model_reg.predict(input_data)[0]
        
        # Correção: consumo_real -> consumo_kw
        consumo_real = float(data['consumo_kw']) 
        
        # 3. Cálculo do Resíduo
        residuo_atual = consumo_real - consumo_esperado
        
        # 4. Lógica da Média Móvel (Janela de 15 min)
        historico_residuos.append(residuo_atual)
        if len(historico_residuos) > 15:
            historico_residuos.pop(0)
        
        residuo_media_15m = np.mean(historico_residuos)
        
        # 5. Predição de Falha (Classificação)
        # O modelo de classificação espera o resíduo e a média
        input_clf = input_data.copy()
        input_clf['residuo'] = residuo_atual
        input_clf['residuo_media_15m'] = residuo_media_15m
        
        prob_falha = model_clf.predict_proba(input_clf)[0][1] # Probabilidade da classe 1
        
        return jsonify({
            'status': 'success',
            'consumo_esperado': round(consumo_esperado, 3),
            'residuo': round(residuo_atual, 3),
            'residuo_media': round(residuo_media_15m, 3),
            'probabilidade_falha': round(prob_falha * 100, 2),
            'alerta_manutencao': True if prob_falha > 0.25 else False # Seu threshold de 25%
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)