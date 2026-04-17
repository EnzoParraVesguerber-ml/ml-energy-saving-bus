import pandas as pd
import requests
import time

# Lê o dataset gerado pelo teu script
df = pd.read_csv('data/dataset_smart_bus_v3.csv')

URL_INGESTAO = 'http://127.0.0.1:5000/api/telemetria/ingestao'

print("Iniciando simulação dos sensores IoT...")

for index, row in df.iterrows():
    payload = {
        "id_onibus": "BUS-001",
        "hora": int(row['hora']),
        "dia_semana": int(row['dia_semana']),
        "is_horario_pico": int(row['is_horario_pico']),
        "temp_externa": float(row['temp_externa']),
        "incidencia_solar": float(row['incidencia_solar']),
        "lotacao": int(row['lotacao']),
        "portas_abertas": int(row['portas_abertas']),
        "velocidade_kmh": float(row['velocidade_kmh']),
        "temp_interna_atual": float(row['temp_interna_atual']),
        "potencia_real_kw": float(row['consumo_kw']) # O sensor lê em kW!
    }
    
    try:
        response = requests.post(URL_INGESTAO, json=payload)
        print(f"[{index}] Enviado: {payload['potencia_real_kw']:.2f} kW | Status: {response.status_code}")
    except Exception as e:
        print(f"Erro de conexão: {e}")
    
    time.sleep(2.5)