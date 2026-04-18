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

# --- API: DASHBOARD DE ECONOMIA (REVISADO: 00h às 20h) ---
@app.route('/api/kpis_economia')
def kpis_economia():
    if df_telemetria is None:
        return jsonify({'error': 'Dataset não carregado'}), 500

    # 1. Definindo as premissas de negócio (O cenário justo)
    HORA_INICIO = 0   # Meia-noite
    HORA_FIM = 20     # 20h da noite
    POTENCIA_MAXIMA_EQUIPAMENTO = 18.0 # kW
    BASELINE_USO_PADRAO = 0.80 # 80% de uso constante (sem IA)
    TARIFA_KWH = 0.85 # R$ por kWh
    
    # 2. Filtrando o Dataset para o novo horário operacional (00h às 20h)
    df_operacional = df_telemetria[
        (df_telemetria['hora'] >= HORA_INICIO) & 
        (df_telemetria['hora'] <= HORA_FIM)
    ].copy()

    # 3. Cálculo de Energia (Dividindo por 60 para converter minutos em horas)
    total_real_kwh = df_operacional['consumo_kw'].sum() / 60.0
    total_sem_ia_kwh = (len(df_operacional) * (POTENCIA_MAXIMA_EQUIPAMENTO * BASELINE_USO_PADRAO)) / 60.0
    
    kwh_poupado = total_sem_ia_kwh - total_real_kwh
    economia_reais = kwh_poupado * TARIFA_KWH
    co2_evitado = kwh_poupado * 0.09

    # 4. Gráfico Diário: Perfil Médio de Consumo por Hora
    perfil_diario = df_operacional.groupby('hora')['consumo_kw'].mean().reset_index()
    
    labels_diario = [f"{int(h):02d}:00" for h in perfil_diario['hora']]
    com_ia_diario = [round(val, 2) for val in perfil_diario['consumo_kw']]
    sem_ia_diario = [POTENCIA_MAXIMA_EQUIPAMENTO * BASELINE_USO_PADRAO] * len(labels_diario)

    # 5. Gráfico Semanal
    # Das 00h até as 20h (inclusas) são 21 horas de operação por dia = 1260 minutos
    minutos_por_dia = 1260
    ultimos_7_dias_kwh = []
    labels_semanal = []

    for i in range(7, 0, -1):
        inicio = -(i * minutos_por_dia)
        fim = -((i - 1) * minutos_por_dia) if i > 1 else None
        bloco_dia = df_operacional.iloc[inicio:fim]
        
        # Economia do dia em kWh
        base_dia = (len(bloco_dia) * (POTENCIA_MAXIMA_EQUIPAMENTO * BASELINE_USO_PADRAO)) / 60.0
        real_dia = bloco_dia['consumo_kw'].sum() / 60.0
        
        ultimos_7_dias_kwh.append(round(base_dia - real_dia, 1))
        labels_semanal.append(f"Dia -{i}")

    return jsonify({
        'kpi_kwh': round(kwh_poupado, 1),
        'kpi_money': round(economia_reais, 2),
        'kpi_co2': round(co2_evitado, 2),
        'chart_diario': {
            'labels': labels_diario,
            'sem_ia': sem_ia_diario,
            'com_ia': com_ia_diario
        },
        'chart_semanal': {
            'labels': labels_semanal,
            'data': ultimos_7_dias_kwh
        },
        'chart_distribuicao': [
            len(df_operacional[df_operacional['consumo_kw'] <= 2.0]),
            len(df_operacional[(df_operacional['consumo_kw'] > 2.0) & (df_operacional['consumo_kw'] <= 12.0)]),
            len(df_operacional[df_operacional['consumo_kw'] > 12.0])
        ]
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)