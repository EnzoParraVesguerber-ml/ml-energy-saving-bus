import pandas as pd
import requests
import time

# Lê o dataset existente
df = pd.read_csv('data/dataset_smart_bus_v3.csv')

URL_INGESTAO = 'http://127.0.0.1:5000/api/telemetria/ingestao'

print("Iniciando simulação dos sensores IoT...")

for index, row in df.iterrows():
    # Monta o payload exatamente como um sensor ESP32 enviaria
    payload = {
        "id_onibus": "BUS-001",
        "temp_externa": float(row['temp_externa']),
        "lotacao": int(row['lotacao']),
        "incidencia_solar": float(row['incidencia_solar']),
        "portas_abertas": int(row['portas_abertas']),
        "potencia_real_kw": float(row['consumo_kw']) # <--- CORRIGIDO AQUI
    }
    
    try:
        response = requests.post(URL_INGESTAO, json=payload)
        print(f"[{index}] Enviado: {payload['potencia_real_kw']} kW | Status: {response.status_code}")
    except Exception as e:
        print(f"Erro de conexão: {e}")
    
    # Aguarda 2 segundos antes de enviar a próxima linha (simulando tempo real)
    time.sleep(2)