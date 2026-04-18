# 🚌 Smart Bus HVAC: Eficiência Energética & Manutenção Preditiva com IA

O **Smart Bus HVAC** é um sistema end-to-end que utiliza Machine Learning para otimizar o consumo de energia de sistemas de climatização em ônibus urbanos. O projeto integra sensores IoT simulados, modelos de regressão para eficiência e modelos de classificação para identificar falhas mecânicas antes que elas ocorram.

---

## 🚀 Funcionalidades Principais

O sistema é dividido em três módulos centrais acessíveis via um portal unificado:

1.  **Monitoramento Real (Telemetria):** Visualização segundo a segundo dos sensores (lotação, temperatura externa, incidência solar) e a resposta da IA em tempo real, consumindo dados diretamente de um banco **PostgreSQL**.
2.  **Playground da IA (Simulador):** Espaço para testes manuais onde é possível alterar variáveis de contexto (ex: dia de extremo calor ou lotação máxima) e observar como os modelos de ML reagem.
3.  **Dashboard de Economia:** Relatório consolidado que utiliza o histórico de dados para calcular a economia financeira (R$), energética (kWh) e a redução na emissão de CO2, baseando-se em premissas reais de operação.

---

## 🧠 Inteligência Artificial

O projeto utiliza uma arquitetura de dois modelos complementares:

* **Regressor (Gradient Boosting):** Treinado para prever a **Potência Ideal** de funcionamento do ar-condicionado baseando-se na carga térmica (temperatura externa, radiação solar e número de passageiros).
* **Classificador (Gradient Boosting):** Analisa o **Resíduo** (diferença entre o consumo real e o previsto pela IA). Caso o consumo real seja consistentemente maior que o ideal, o sistema gera um alerta de manutenção preditiva (ex: filtros sujos ou vazamento de gás).

---

## 📊 Premissas de Negócio e Cálculos

Para garantir que o dashboard de economia seja realista, o sistema aplica as seguintes regras:

* **Janela Operacional:** Considera o funcionamento do ônibus das **00:00 às 20:00**, cobrindo tanto períodos frios quanto horários de pico térmico para uma média justa.
* **Baseline de Comparação:** O sistema compara o uso da IA contra um ar-condicionado padrão operando fixo em **80% de sua capacidade** (uso comum sem automação inteligente).
* **Conversão Energética:** Os dados de telemetria coletados por minuto são convertidos para **kWh** para o cálculo preciso de faturamento e impacto ambiental.

---

## 🛠️ Tech Stack

* **Linguagem:** Python 3.x
* **Framework Web:** Flask
* **Banco de Dados:** PostgreSQL (psycopg2)
* **Data Science:** Pandas, NumPy, Scikit-learn, Joblib
* **Front-end:** HTML5, CSS3 (Glassmorphism), JavaScript (Chart.js)
* **Simulação:** Script Python para Mock de Sensores IoT via Requests

---

## 📦 Como Instalar e Rodar

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/EnzoParraVesguerber-ml/ml-energy-saving-bus.git
    cd ml-energy-saving-bus
    ```

2.  **Crie um ambiente virtual e instale as dependências:**
    ```bash
    python -m venv venv
    # No Windows:
    venv\Scripts\activate
    # No Linux/Mac:
    source venv/bin/activate

    pip install -r requirements.txt
    ```

3.  **Configure o Banco de Dados:**
    * Crie um banco de dados no PostgreSQL chamado `smartbus`.
    * Certifique-se de que a tabela `telemetria_onibus` exista com as colunas necessárias.
    * Ajuste as credenciais de conexão no arquivo `app.py`.

4.  **Inicie o Servidor Backend:**
    ```bash
    python app.py
    ```

5.  **Inicie o Simulador de Sensores (em outro terminal):**
    ```bash
    python sensor_mock.py
    ```

---

## 📁 Estrutura do Projeto

* `app.py`: Servidor Flask com as rotas de API e processamento de ML.
* `sensor_mock.py`: Simulador que lê o CSV e envia dados para o servidor.
* `/models`: Contém os modelos `.pkl` treinados.
* `/templates`: Páginas HTML (index, status, simulador, economia).
* `/static`: Arquivos CSS de estilo e lógica JavaScript.
* `/data`: Conjunto de dados sintéticos utilizado para treino e simulação.

---

**Autores:** Enzo Parra, Gustavo Yuji, Bruno Torres
**Instituição:** Facens - Engenharia de Computação (3º Semestre)
