import os
import matplotlib.pyplot as plt
from prophet import Prophet
import pandas as pd
import requests
from facebook_business.api import FacebookAdsApi
from facebook_business.adobjects.adset import AdSet
from facebook_business.adobjects.adaccount import AdAccount

# Load Environment Variables at the VERY beginning, after imports
my_app_id = os.environ.get('FACEBOOK_APP_ID')
my_app_secret = os.environ.get('FACEBOOK_APP_SECRET')
my_access_token = os.environ.get('FACEBOOK_ACCESS_TOKEN')

if not my_app_id or not my_app_secret or not my_access_token:
    raise Exception("Please set FACEBOOK_APP_ID, FACEBOOK_APP_SECRET, and FACEBOOK_ACCESS_TOKEN environment variables in Railway.")

# Initialize Facebook Ads API *after* loading environment variables
FacebookAdsApi.init(my_app_id, my_app_secret, my_access_token)

# Configuração de Variáveis (Variable Configuration)
# Limites de orçamento (em Euros) (Budget Limits in Euros)
min_budget = 5  # Orçamento mínimo (Minimum Budget)
max_budget = 100  # Orçamento máximo (Maximum Budget)

# Limite de CPL aceitável (Acceptable CPL Limit)
limite_cpl = 12  # Exemplo de limite de CPL aceitável (Example of acceptable CPL limit)

# Lista de contas de anúncio que você quer analisar (List of Ad Accounts to Analyze)
ad_accounts = ['act_300157559082101', 'act_705674745014619']  # IDs de múltiplas contas de anúncio (IDs of multiple ad accounts)


# Função para ajustar o orçamento com base no CPL previsto e real (Function to adjust budget based on predicted and real CPL)
def ajustar_budget(cpl_previsto, cpl_real, adset_id, current_budget):
    # Caso 1: Se o CPL previsto é maior que o CPL real, devemos reduzir o orçamento (Case 1: If predicted CPL is greater than real CPL, reduce budget)
    if cpl_previsto > cpl_real:
        new_budget = current_budget * 0.9  # Reduzir o orçamento em 10% (Reduce budget by 10%)
        action = "Reduzir"  # Reduce

    # Caso 2: Se o CPL previsto é menor que o CPL real e está abaixo do limite, podemos aumentar o orçamento (Case 2: If predicted CPL is less than real CPL and below limit, increase budget)
    elif cpl_previsto < cpl_real and cpl_previsto < limite_cpl:
        new_budget = current_budget * 1.1  # Aumentar o orçamento em 10% (Increase budget by 10%)
        action = "Aumentar"  # Increase

    # Caso 3: Se o CPL previsto está dentro do limite aceitável e estável, manter o orçamento (Case 3: If predicted CPL is within acceptable limit and stable, maintain budget)
    else:
        new_budget = current_budget
        action = "Manter"  # Maintain

    # Garantir que o orçamento não seja menor que o valor mínimo e não ultrapasse o valor máximo (Ensure budget is not less than minimum and does not exceed maximum)
    if new_budget < min_budget:
        new_budget = min_budget
    if new_budget > max_budget:
        new_budget = max_budget

    return new_budget, action


# Função para processar cada conta de anúncio (Function to process each ad account)
def processar_conta(ad_account_id):
    # Definir a conta de anúncios que você quer analisar (Define the ad account to analyze)
    ad_account = AdAccount(ad_account_id)

    # Definir o período de tempo e as métricas que deseja coletar (Define the time period and metrics to collect)
    fields = [
        'spend',
        'impressions',
        'clicks',
        'cpc',
        'cpm',
        'actions',               # Ações (incluindo leads) (Actions, including leads)
        'cost_per_action_type',   # Custo por cada tipo de ação, como leads (Cost per action type, like leads)
        'campaign_name',          # Nome da campanha para identificação (Campaign name for identification)
        'adset_name',             # Nome do adgroup (conjunto de anúncios) (Adset name)
        'adset_id'                # ID do adgroup para ajuste posterior (Adset ID for later adjustment)
    ]

    # Filtrar apenas campanhas ativas e adgroups ativos e garantir que os dados sejam diários (Filter only active campaigns and adsets and ensure daily data)
    params = {
        'date_preset': 'last_30d',
        'level': 'adset',  # Pega os dados a nível de adgroup (conjunto de anúncios) (Get data at adset level)
        'time_increment': 1,  # Garante que os dados sejam diários (Ensure daily data)
        'filtering': [{'field': 'adset.effective_status', 'operator': 'IN', 'value': ['ACTIVE']}]
    }

    # Coletar os insights do Facebook Ads (Collect Facebook Ads insights)
    try:
        insights = ad_account.get_insights(fields=fields, params=params)

        # Converter os dados em DataFrame do Pandas para análise (Convert data to Pandas DataFrame for analysis)
        data = pd.DataFrame(insights)

        # Função para filtrar o custo por lead (CPL) a partir do campo cost_per_action_type (Function to extract CPL from cost_per_action_type field)
        def extract_cpl(row):
            for action in row.get('cost_per_action_type', []):
                if action['action_type'] == 'lead':
                    return action['value']
            return None

        # Criar uma nova coluna para o Custo por Lead (CPL) (Create a new column for Cost per Lead (CPL))
        data['CPL'] = data.apply(extract_cpl, axis=1)

        # Converter a coluna CPL para numérico, forçando a conversão e substituindo erros por NaN (Convert CPL column to numeric, forcing conversion and replacing errors with NaN)
        data['CPL'] = pd.to_numeric(data['CPL'], errors='coerce')

        # Preencher os valores de CPL ausentes com a média do CPL para o adgroup (Fill missing CPL values with the average CPL for the adset)
        data['CPL'] = data.groupby('adset_id')['CPL'].transform(lambda x: x.fillna(x.mean()))

        # Verificar se a coluna 'date_start' está no formato correto (convertendo para datetime) (Verify if 'date_start' column is in correct format (converting to datetime))
        data['date_start'] = pd.to_datetime(data['date_start'])

        # Verificar os dados após a conversão (Verify data after conversion)
        print(f"Dados após conversão para a conta {ad_account_id}:")
        print(data[['adset_id', 'adset_name', 'CPL']].head())

        # Iterar por cada adgroup ativo (Iterate through each active adset)
        for adset_id, group_data in data.groupby('adset_id'):

            # Extrair o nome do adgroup (adset_name) para usar no log e gráfico (Extract adset name for logs and graph)
            adgroup_name = group_data['adset_name'].iloc[0]  # Pegar o nome do adgroup (Get adset name)

            # Preparar os dados para o Prophet (Prepare data for Prophet)
            df_prophet = group_data[['date_start', 'CPL']].rename(columns={'date_start': 'ds', 'CPL': 'y'})

            # Verificar se o adgroup tem pelo menos 2 linhas válidas (Check if adset has at least 2 valid rows)
            if df_prophet['y'].notnull().sum() < 2:
                print(f"Adgroup {adgroup_name} tem menos de 2 linhas com dados válidos. Pulando...")
                continue

            # Criar e ajustar o modelo Prophet (Create and fit Prophet model)
            model = Prophet()
            model.fit(df_prophet)

            # Fazer previsões para os próximos 30 dias (Make predictions for the next 30 days)
            future = model.make_future_dataframe(periods=7)
            forecast = model.predict(future)

            # Exibir a previsão para verificação (Display forecast for verification)
            # Ajuste para usar a média dos próximos 7 dias de previsão (Adjust to use the average of the next 7 days of forecast)
            cpl_previsto_7dias = forecast['yhat'].iloc[-7:].mean()
            cpl_real_atual = data[data['adset_id'] == adset_id]['CPL'].iloc[-1]

            # Obter o orçamento atual do adgroup (Get current adset budget)
            adset = AdSet(adset_id).api_get(fields=['daily_budget'])
            current_budget = int(adset['daily_budget']) / 100  # O orçamento é armazenado em centavos (Budget is stored in cents)

            # Ajustar o orçamento com base na previsão, com limites de orçamento mínimo e máximo (Adjust budget based on forecast, with min and max budget limits)
            new_budget, action = ajustar_budget(cpl_previsto_7dias, cpl_real_atual, adset_id, current_budget)

            # Atualizar o orçamento do adgroup via API do Facebook (Update adset budget via Facebook API)
            try:
                adset.api_update(params={
                    'daily_budget': int(new_budget * 100),  # O orçamento deve ser fornecido em centavos (Budget must be provided in cents)
                })
                print(f"Orçamento atualizado para o Adgroup {adgroup_name}: {new_budget:.2f} € ({action})")
            except Exception as e:
                print(f"Erro ao atualizar o orçamento do Adgroup {adgroup_name}: {e}")

            # Exibir a previsão (Display forecast)
            print(f"Previsão de CPL para o Adgroup {adgroup_name}:")
            print(forecast[['ds', 'yhat']].tail())

            # Gerar e exibir o gráfico de previsão para o adgroup usando o nome (Generate and display forecast graph for adset using name)
            plt.figure(figsize=(10, 6))
            model.plot(forecast)

            # Melhorias no gráfico: adicionar intervalo de confiança e dados reais (Graph improvements: add confidence interval and real data)
            plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.2)
            plt.scatter(df_prophet['ds'], df_prophet['y'], color='red', label='Dados reais (Real Data)')
            plt.title(f"Previsão de CPL para o Adgroup: {adgroup_name}")
            plt.xlabel("Data")
            plt.ylabel("Custo por Lead (CPL)")
            plt.legend()
            plt.grid(True)
            plt.show()  # Exibir o gráfico (Display graph)

    except Exception as e:
        print(f"Erro ao acessar a API do Meta Ads para a conta {ad_account_id}: Erro: {e}")  # Improved error message, printing the exception 'e'


# Processar todas as contas de anúncio (Process all ad accounts)
for account_id in ad_accounts:
    processar_conta(account_id)
