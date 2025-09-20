from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64, String
import pandas as pd

# Definir entidades principais
customer = Entity(
    name="customer_id", 
    value_type=ValueType.STRING,
    description="Identificador único do cliente"
)

merchant = Entity(
    name="merchant_id",
    value_type=ValueType.STRING, 
    description="Identificador único do estabelecimento"
)

# Fonte de dados offline (arquivos Parquet)
transaction_source = FileSource(
    name="transaction_data",
    path="../offline_data/transactions.parquet",
    timestamp_field="event_timestamp",
)

# Feature Views para agregações de transações por cliente
customer_transaction_features = FeatureView(
    name="customer_transaction_features",
    entities=[customer],
    ttl=timedelta(hours=24),
    schema=[
        Field(name="txn_amount_sum_60s", dtype=Float32, description="Soma do valor transacionado nos últimos 60 segundos"),
        Field(name="txn_amount_sum_5m", dtype=Float32, description="Soma do valor transacionado nos últimos 5 minutos"),
        Field(name="txn_amount_sum_1h", dtype=Float32, description="Soma do valor transacionado na última hora"),
        Field(name="txn_count_60s", dtype=Int64, description="Número de transações nos últimos 60 segundos"),
        Field(name="txn_count_5m", dtype=Int64, description="Número de transações nos últimos 5 minutos"),
        Field(name="txn_count_10m", dtype=Int64, description="Número de transações nos últimos 10 minutos"),
        Field(name="txn_count_1h", dtype=Int64, description="Número de transações na última hora"),
        Field(name="unique_merchants_1h", dtype=Int64, description="Número de estabelecimentos únicos na última hora"),
        Field(name="avg_txn_amount_1h", dtype=Float32, description="Valor médio das transações na última hora"),
        Field(name="max_txn_amount_1h", dtype=Float32, description="Maior valor de transação na última hora"),
    ],
    online=True,
    source=transaction_source,
    tags={"team": "aml_team", "domain": "fraud_detection"},
)

# Feature Views para agregações por estabelecimento
merchant_transaction_features = FeatureView(
    name="merchant_transaction_features", 
    entities=[merchant],
    ttl=timedelta(hours=24),
    schema=[
        Field(name="merchant_txn_count_1h", dtype=Int64, description="Número de transações do estabelecimento na última hora"),
        Field(name="merchant_txn_amount_sum_1h", dtype=Float32, description="Soma das transações do estabelecimento na última hora"),
        Field(name="merchant_unique_customers_1h", dtype=Int64, description="Número de clientes únicos do estabelecimento na última hora"),
        Field(name="merchant_avg_txn_amount_1h", dtype=Float32, description="Valor médio das transações do estabelecimento na última hora"),
    ],
    online=True,
    source=transaction_source,
    tags={"team": "aml_team", "domain": "merchant_analysis"},
)

# Feature Views para padrões comportamentais avançados
customer_behavioral_features = FeatureView(
    name="customer_behavioral_features",
    entities=[customer], 
    ttl=timedelta(hours=24),
    schema=[
        Field(name="unique_ips_1h", dtype=Int64, description="Número de IPs únicos utilizados na última hora"),
        Field(name="night_txn_count_24h", dtype=Int64, description="Número de transações noturnas (22h-6h) nas últimas 24 horas"),
        Field(name="weekend_txn_count_7d", dtype=Int64, description="Número de transações em finais de semana nos últimos 7 dias"),
        Field(name="velocity_score_1h", dtype=Float32, description="Score de velocidade de transações na última hora"),
        Field(name="amount_deviation_score_1h", dtype=Float32, description="Score de desvio do padrão de valores na última hora"),
    ],
    online=True,
    source=transaction_source,
    tags={"team": "aml_team", "domain": "behavioral_analysis"},
)
