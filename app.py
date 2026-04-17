import os
import logging
import pandas as pd
import numpy as np
import joblib
import psycopg2
from psycopg2.extras import RealDictCursor
from flask import Flask, render_template, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Segurança: Rate Limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["500 per day", "100 per hour"],
    storage_uri="memory://"
)

# --- CONFIGURAÇÃO DE CONEXÃO COM O BANCO ---
DB_CONFIG = {
    'dbname': 'smartbus',
    'user': 'postgres',
    'password': 'admin',
    'host': 'localhost',
    'port': '5432'
}

def get_db_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    return conn

# --- CONFIGURAÇÃO DE CAMINHOS E CARREGAMENTO ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_REG_PATH = os.path.join(BASE_DIR, 'models', 'hvac_regressor_gb.pkl')
MODEL_CLF_PATH = os.path.join(BASE_DIR, 'models', 'modelo_manutencao_gradient_boosting.pkl')
DATASET_PATH = os.path.join(BASE_DIR, 'data', 'dataset_smart_bus_v3.csv')

# Variáveis Globais de Estado
model_reg = None
model_clf = None
df_telemetria = None
index_simulacao = 0

def carregar_recursos():
    global model_reg, model_clf, df_telemetria
    try:
        model_reg = joblib.load(MODEL_REG_PATH)
        model_clf = joblib.load(MODEL_CLF_PATH)
        df_telemetria = pd.read_csv(DATASET_PATH)
        logging.info("Modelos e Dataset carregados com sucesso.")
    except Exception as e:
        logging.error(f"Erro crítico ao carregar recursos: {e}")

carregar_recursos()

# --- ROTAS DE PÁGINAS (FRONTEND) ---
@app.route('/')
def index():
    return render_template('index.html') # Nova página inicial

@app.route('/status')
def status():
    return render_template('status.html') # Antiga página principal

@app.route('/simulador')
def simulador():
    return render_template('simulador.html')

@app.route('/economia')
def economia():
    return render_template('economia.html')

# --- API: PLAYGROUND DA IA (INFERÊNCIA MANUAL) ---
# --- API: PLAYGROUND DA IA (INFERÊNCIA MANUAL) ---
@app.route('/api/predict', methods=['POST'])
@limiter.limit("2 per second")
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'Payload vazio.'}), 400

        # 1. Preparar dados para o modelo (Agora dinâmico e real!)
        input_df = pd.DataFrame([{
            'hora': int(data['hora']),
            'dia_semana': int(data['dia_semana']),
            'is_horario_pico': int(data['is_horario_pico']),
            'temp_externa': float(data['temp_externa']),
            'incidencia_solar': float(data['incidencia_solar']),
            'lotacao': int(data['lotacao']),
            'portas_abertas': int(data['portas_abertas']),
            'velocidade_kmh': float(data['velocidade_kmh']),   
            'temp_interna_atual': float(data['temp_interna_atual']) 
        }])
        
        # A IA prevê o target (escala 0 a 100)
        potencia_esperada_target = model_reg.predict(input_df)[0]
        
        # CONVERSÃO: Transforma o Target (0-100) em Consumo Esperado (kW)
        potencia_esperada_kw = (potencia_esperada_target / 100.0) * 18.0
        
        # 2. Cálculos temporais (Protegido contra energia negativa)
        potencia_real = max(0.0, float(data['potencia_real_kw']))
        
        # Resíduo comparando kW reais com kW esperados
        residuo_atual = potencia_real - potencia_esperada_kw
        
        historico = data.get('historico_residuos', [])
        historico.append(residuo_atual)
        if len(historico) > 15: historico = historico[-15:]
        residuo_media = np.mean(historico)
        
        # 3. Classificador: Exatamente 4 features, na ordem do fit
        input_clf = pd.DataFrame([{
            'potencia_hvac_target': potencia_esperada_target, # Usa a escala 0-100 aqui
            'consumo_kw': potencia_real,
            'residuo_consumo': residuo_atual,
            'residuo_media_15m': residuo_media
        }])
        
        prob_falha = model_clf.predict_proba(input_clf)[0][1]
        
        return jsonify({
            'status': 'success',
            'potencia_esperada': round(potencia_esperada_kw, 3), # Envia em kW para o gráfico
            'residuo_atual': round(residuo_atual, 3),
            'residuo_media': round(residuo_media, 3),
            'novo_historico_residuos': [round(r, 3) for r in historico],
            'probabilidade_falha': round(prob_falha * 100, 2),
            'alerta_manutencao': bool(prob_falha > 0.25)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# --- API: INGESTÃO DE DADOS (IOT) ---
@app.route('/api/telemetria/ingestao', methods=['POST'])
def ingestao():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Payload vazio'}), 400

    # 1. Preparar dados para o modelo (com as variáveis todas que o mock agora envia)
    input_df = pd.DataFrame([{
        'hora': data['hora'],
        'dia_semana': data['dia_semana'],
        'is_horario_pico': data['is_horario_pico'],
        'temp_externa': data['temp_externa'],
        'incidencia_solar': data['incidencia_solar'],
        'lotacao': data['lotacao'],
        'portas_abertas': data['portas_abertas'],
        'velocidade_kmh': data['velocidade_kmh'],   
        'temp_interna_atual': data['temp_interna_atual'] 
    }])
    
    # A IA prevê o target (escala 0 a 100)
    potencia_esperada_target = model_reg.predict(input_df)[0]
    
    # CONVERSÃO FÍSICA: Transformar o Target (0-100) em Consumo (kW)
    consumo_esperado_kw = (potencia_esperada_target / 100.0) * 18.0
    
    # PROTEÇÃO: Impede que a energia seja menor que zero (Física real)
    potencia_real = max(0.0, float(data['potencia_real_kw']))
    
    # Agora sim, Resíduo = kW reais vs kW esperados!
    residuo = potencia_real - consumo_esperado_kw
    
    # 2. Classificador de Falhas
    input_clf = pd.DataFrame([{
        'potencia_hvac_target': potencia_esperada_target, # O classificador foi treinado com o target bruto
        'consumo_kw': potencia_real,
        'residuo_consumo': residuo,
        'residuo_media_15m': residuo
    }])
    
    prob_falha = model_clf.predict_proba(input_clf)[0][1]

    # 3. Salvar no PostgreSQL (Guardamos o consumo_esperado em kW para o Front-End ler)
    conn = get_db_connection()
    cur = conn.cursor()
    query = """
        INSERT INTO telemetria_onibus 
        (id_onibus, temp_externa, lotacao, incidencia_solar, portas_abertas, potencia_real_kw, potencia_esperada_kw, probabilidade_falha)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    valores = (
        data['id_onibus'], data['temp_externa'], data['lotacao'], 
        data['incidencia_solar'], data['portas_abertas'], potencia_real, 
        float(consumo_esperado_kw), float(prob_falha)
    )
    
    cur.execute(query, valores)
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({'status': 'success'}), 201

# --- API: TELEMETRIA EM TEMPO REAL (CONSUMINDO DO POSTGRESQL) ---
@app.route('/api/status_atual')
def status_atual():
    try:
        conn = get_db_connection()
        # O RealDictCursor permite aceder às colunas pelo nome (ex: row['lotacao'])
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Puxa apenas a última telemetria recebida do sensor_mock
        cur.execute("SELECT * FROM telemetria_onibus ORDER BY id DESC LIMIT 1;")
        row = cur.fetchone()
        
        cur.close()
        conn.close()

        if not row:
            return jsonify({'error': 'Nenhum dado no banco ainda'}), 404

        # Calcula o resíduo usando os dados que a IA já previu e guardou no banco
        residuo = float(row['potencia_real_kw']) - float(row['potencia_esperada_kw'])

        # Entrega exatamente os nomes das chaves que o teu status.js espera ler
        return jsonify({
            'lotacao': row['lotacao'],
            'temp_externa': round(float(row['temp_externa']), 1),
            'consumo_real': round(float(row['potencia_real_kw']), 2),
            'residuo': round(residuo, 2),
            'alerta_manutencao': bool(row['probabilidade_falha'] > 0.25)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- API: DASHBOARD DE ECONOMIA (DADOS AGREGADOS) ---
@app.route('/api/kpis_economia')
def kpis_economia():
    if df_telemetria is None:
        return jsonify({'error': 'Dataset não carregado'}), 500

    # 1. Agregação dos KPIs Superiores
    # Calculamos o total consumido no CSV vs o que gastaria com ar condicionado no máximo (ex: 22 kW fixos)
    total_real = df_telemetria['consumo_kw'].sum() # <-- CORRIGIDO AQUI
    total_sem_ia = len(df_telemetria) * 22.0  
    
    kwh_poupado = total_sem_ia - total_real
    economia_reais = kwh_poupado * 0.85 # Valor estimado da tarifa
    co2_evitado = kwh_poupado * 0.09 # Fator de emissão (kg CO2 / kWh)

    # 2. Distribuição de Uso da Potência (Gráfico de Rosca)
    # Filtramos as categorias diretamente do dataframe original
    desligado = len(df_telemetria[df_telemetria['consumo_kw'] <= 2.0]) # <-- CORRIGIDO AQUI
    frio_leve = len(df_telemetria[(df_telemetria['consumo_kw'] > 2.0) & (df_telemetria['consumo_kw'] <= 15.0)]) # <-- CORRIGIDO AQUI
    potencia_alta = len(df_telemetria[df_telemetria['consumo_kw'] > 15.0]) # <-- CORRIGIDO AQUI

    # 3. Gráfico Semanal (Barras)
    # Como a granularidade é de 1 minuto, cada dia tem 1440 minutos.
    # Vamos fatiar as últimas 7 "fatias de 1440 linhas" para simular a última semana operada.
    ultimos_7_dias_kwh = []
    for i in range(7, 0, -1):
        # Seleciona o bloco de 1 dia
        inicio = -(i * 1440)
        fim = -((i - 1) * 1440) if i > 1 else None
        bloco_dia = df_telemetria.iloc[inicio:fim]
        
        # Calcula a economia desse dia específico
        poupado_dia = (len(bloco_dia) * 22.0) - bloco_dia['consumo_kw'].sum() # <-- CORRIGIDO AQUI
        ultimos_7_dias_kwh.append(round(poupado_dia, 1))

    return jsonify({
        'kpi_kwh': round(kwh_poupado, 1),
        'kpi_money': round(economia_reais, 2),
        'kpi_co2': round(co2_evitado, 2),
        
        # Gráfico de Linha (Comparativo) - Pode manter uma amostra ou agrupar por hora real
        'chart_diario': {
            'labels': ['06:00', '09:00', '12:00', '15:00', '18:00', '21:00'],
            'sem_ia': [22, 22, 22, 22, 22, 22],
            'com_ia': [14, 18, 24, 25, 20, 16] # Exemplificado. Ideal: df.groupby() de uma coluna de hora
        },
        
        # Injeta os arrays reais calculados pelo Pandas
        'chart_semanal': ultimos_7_dias_kwh,
        'chart_distribuicao': [desligado, frio_leve, potencia_alta]
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)