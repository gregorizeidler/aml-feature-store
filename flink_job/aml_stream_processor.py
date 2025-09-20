"""
Job Flink para processamento de stream de transações e cálculo de features em tempo real
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Iterable

from pyflink.common import Types, Time, WatermarkStrategy
from pyflink.common.serialization import SimpleStringSchema
from pyflink.datastream import StreamExecutionEnvironment, TimeCharacteristic
from pyflink.datastream.connectors import FlinkKafkaConsumer, FlinkKafkaProducer
from pyflink.datastream.functions import KeyedProcessFunction, RuntimeContext
from pyflink.datastream.state import ValueStateDescriptor, ListStateDescriptor
from pyflink.datastream.window import TumblingEventTimeWindows, SlidingEventTimeWindows
from pyflink.table import StreamTableEnvironment

import redis

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransactionAggregator(KeyedProcessFunction):
    """
    Função para agregar transações por cliente e calcular features em tempo real
    """
    
    def __init__(self, redis_host='redis', redis_port=6379):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_client = None
        
        # Descritores de estado
        self.txn_history_state = None
        self.ip_history_state = None
        self.merchant_history_state = None
        
    def open(self, runtime_context: RuntimeContext):
        """Inicialização da função"""
        # Conectar ao Redis
        self.redis_client = redis.Redis(
            host=self.redis_host, 
            port=self.redis_port, 
            decode_responses=True
        )
        
        # Configurar estados
        self.txn_history_state = runtime_context.get_list_state(
            ListStateDescriptor("txn_history", Types.STRING())
        )
        
        self.ip_history_state = runtime_context.get_list_state(
            ListStateDescriptor("ip_history", Types.STRING())
        )
        
        self.merchant_history_state = runtime_context.get_list_state(
            ListStateDescriptor("merchant_history", Types.STRING())
        )
        
        logger.info("TransactionAggregator inicializado")

    def process_element(self, value, ctx):
        """Processa cada transação"""
        try:
            # Parse da transação
            transaction = json.loads(value)
            customer_id = transaction['customer_id']
            amount = float(transaction['amount'])
            timestamp = datetime.fromisoformat(transaction['event_timestamp'].replace('Z', '+00:00'))
            ip_address = transaction['ip_address']
            merchant_id = transaction['merchant_id']
            
            current_time = ctx.timestamp()
            
            # Obter histórico de transações
            txn_history = list(self.txn_history_state.get())
            ip_history = list(self.ip_history_state.get())
            merchant_history = list(self.merchant_history_state.get())
            
            # Adicionar transação atual
            txn_record = {
                'amount': amount,
                'timestamp': current_time,
                'ip': ip_address,
                'merchant': merchant_id
            }
            
            txn_history.append(json.dumps(txn_record))
            ip_history.append(json.dumps({'ip': ip_address, 'timestamp': current_time}))
            merchant_history.append(json.dumps({'merchant': merchant_id, 'timestamp': current_time}))
            
            # Limpar registros antigos (manter apenas última hora)
            cutoff_time = current_time - 3600000  # 1 hora em ms
            
            txn_history = [t for t in txn_history 
                          if json.loads(t)['timestamp'] > cutoff_time]
            ip_history = [i for i in ip_history 
                         if json.loads(i)['timestamp'] > cutoff_time]
            merchant_history = [m for m in merchant_history 
                               if json.loads(m)['timestamp'] > cutoff_time]
            
            # Atualizar estados
            self.txn_history_state.clear()
            self.txn_history_state.add_all(txn_history)
            
            self.ip_history_state.clear()
            self.ip_history_state.add_all(ip_history)
            
            self.merchant_history_state.clear()
            self.merchant_history_state.add_all(merchant_history)
            
            # Calcular features
            features = self._calculate_features(txn_history, ip_history, merchant_history, current_time)
            
            # Salvar no Redis
            self._save_features_to_redis(customer_id, features)
            
            # Emitir resultado
            yield json.dumps({
                'customer_id': customer_id,
                'features': features,
                'timestamp': current_time
            })
            
        except Exception as e:
            logger.error(f"Erro ao processar transação: {e}")

    def _calculate_features(self, txn_history, ip_history, merchant_history, current_time):
        """Calcula features baseadas no histórico"""
        features = {}
        
        # Parse dos históricos
        transactions = [json.loads(t) for t in txn_history]
        ips = [json.loads(i) for i in ip_history]
        merchants = [json.loads(m) for m in merchant_history]
        
        # Features de janelas temporais
        time_windows = {
            '60s': current_time - 60000,    # 60 segundos
            '5m': current_time - 300000,    # 5 minutos
            '10m': current_time - 600000,   # 10 minutos
            '1h': current_time - 3600000    # 1 hora
        }
        
        for window_name, cutoff in time_windows.items():
            # Transações na janela
            window_txns = [t for t in transactions if t['timestamp'] > cutoff]
            
            # Soma dos valores
            features[f'txn_amount_sum_{window_name}'] = sum(t['amount'] for t in window_txns)
            
            # Contagem de transações
            features[f'txn_count_{window_name}'] = len(window_txns)
            
            # Valor médio
            if window_txns:
                features[f'avg_txn_amount_{window_name}'] = features[f'txn_amount_sum_{window_name}'] / len(window_txns)
                features[f'max_txn_amount_{window_name}'] = max(t['amount'] for t in window_txns)
            else:
                features[f'avg_txn_amount_{window_name}'] = 0.0
                features[f'max_txn_amount_{window_name}'] = 0.0
        
        # Features de IPs únicos na última hora
        unique_ips_1h = set(ip['ip'] for ip in ips if ip['timestamp'] > time_windows['1h'])
        features['unique_ips_1h'] = len(unique_ips_1h)
        
        # Features de estabelecimentos únicos na última hora
        unique_merchants_1h = set(m['merchant'] for m in merchants if m['timestamp'] > time_windows['1h'])
        features['unique_merchants_1h'] = len(unique_merchants_1h)
        
        # Features comportamentais avançadas
        if transactions:
            # Score de velocidade (transações por minuto na última hora)
            txns_last_hour = [t for t in transactions if t['timestamp'] > time_windows['1h']]
            features['velocity_score_1h'] = len(txns_last_hour) / 60.0  # transações por minuto
            
            # Score de desvio de valores
            if len(txns_last_hour) > 1:
                amounts = [t['amount'] for t in txns_last_hour]
                avg_amount = sum(amounts) / len(amounts)
                variance = sum((x - avg_amount) ** 2 for x in amounts) / len(amounts)
                features['amount_deviation_score_1h'] = variance ** 0.5  # desvio padrão
            else:
                features['amount_deviation_score_1h'] = 0.0
        
        # Features de transações noturnas (últimas 24h - simulado com 1h)
        night_txns = [t for t in transactions 
                     if t['timestamp'] > time_windows['1h'] and 
                     (datetime.fromtimestamp(t['timestamp']/1000).hour < 6 or 
                      datetime.fromtimestamp(t['timestamp']/1000).hour > 22)]
        features['night_txn_count_24h'] = len(night_txns)
        
        # Features de fim de semana (últimos 7 dias - simulado com 1h)
        weekend_txns = [t for t in transactions 
                       if t['timestamp'] > time_windows['1h'] and 
                       datetime.fromtimestamp(t['timestamp']/1000).weekday() >= 5]
        features['weekend_txn_count_7d'] = len(weekend_txns)
        
        return features

    def _save_features_to_redis(self, customer_id, features):
        """Salva features no Redis para acesso rápido"""
        try:
            # Chave no formato esperado pelo Feast
            redis_key = f"aml_feature_store:customer_transaction_features:{customer_id}"
            
            # Salvar features com TTL de 24 horas
            pipeline = self.redis_client.pipeline()
            
            for feature_name, value in features.items():
                pipeline.hset(redis_key, feature_name, str(value))
            
            pipeline.expire(redis_key, 86400)  # 24 horas
            pipeline.execute()
            
            logger.debug(f"Features salvas no Redis para cliente {customer_id}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar no Redis: {e}")

class MerchantAggregator(KeyedProcessFunction):
    """
    Função para agregar transações por estabelecimento
    """
    
    def __init__(self, redis_host='redis', redis_port=6379):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_client = None
        self.merchant_txn_state = None
        self.merchant_customer_state = None
        
    def open(self, runtime_context: RuntimeContext):
        """Inicialização da função"""
        self.redis_client = redis.Redis(
            host=self.redis_host, 
            port=self.redis_port, 
            decode_responses=True
        )
        
        self.merchant_txn_state = runtime_context.get_list_state(
            ListStateDescriptor("merchant_txns", Types.STRING())
        )
        
        self.merchant_customer_state = runtime_context.get_list_state(
            ListStateDescriptor("merchant_customers", Types.STRING())
        )

    def process_element(self, value, ctx):
        """Processa transações por estabelecimento"""
        try:
            transaction = json.loads(value)
            merchant_id = transaction['merchant_id']
            customer_id = transaction['customer_id']
            amount = float(transaction['amount'])
            current_time = ctx.timestamp()
            
            # Obter histórico
            txn_history = list(self.merchant_txn_state.get())
            customer_history = list(self.merchant_customer_state.get())
            
            # Adicionar nova transação
            txn_record = {
                'amount': amount,
                'customer_id': customer_id,
                'timestamp': current_time
            }
            
            txn_history.append(json.dumps(txn_record))
            customer_history.append(json.dumps({'customer_id': customer_id, 'timestamp': current_time}))
            
            # Limpar registros antigos (última hora)
            cutoff_time = current_time - 3600000
            
            txn_history = [t for t in txn_history 
                          if json.loads(t)['timestamp'] > cutoff_time]
            customer_history = [c for c in customer_history 
                               if json.loads(c)['timestamp'] > cutoff_time]
            
            # Atualizar estados
            self.merchant_txn_state.clear()
            self.merchant_txn_state.add_all(txn_history)
            
            self.merchant_customer_state.clear()
            self.merchant_customer_state.add_all(customer_history)
            
            # Calcular features do estabelecimento
            features = self._calculate_merchant_features(txn_history, customer_history)
            
            # Salvar no Redis
            self._save_merchant_features_to_redis(merchant_id, features)
            
        except Exception as e:
            logger.error(f"Erro ao processar transação do estabelecimento: {e}")

    def _calculate_merchant_features(self, txn_history, customer_history):
        """Calcula features do estabelecimento"""
        transactions = [json.loads(t) for t in txn_history]
        customers = [json.loads(c) for c in customer_history]
        
        features = {
            'merchant_txn_count_1h': len(transactions),
            'merchant_txn_amount_sum_1h': sum(t['amount'] for t in transactions),
            'merchant_unique_customers_1h': len(set(c['customer_id'] for c in customers))
        }
        
        if transactions:
            features['merchant_avg_txn_amount_1h'] = features['merchant_txn_amount_sum_1h'] / len(transactions)
        else:
            features['merchant_avg_txn_amount_1h'] = 0.0
            
        return features

    def _save_merchant_features_to_redis(self, merchant_id, features):
        """Salva features do estabelecimento no Redis"""
        try:
            redis_key = f"aml_feature_store:merchant_transaction_features:{merchant_id}"
            
            pipeline = self.redis_client.pipeline()
            for feature_name, value in features.items():
                pipeline.hset(redis_key, feature_name, str(value))
            
            pipeline.expire(redis_key, 86400)
            pipeline.execute()
            
        except Exception as e:
            logger.error(f"Erro ao salvar features do estabelecimento no Redis: {e}")

def create_flink_job():
    """Cria e configura o job Flink"""
    
    # Configurar ambiente
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_stream_time_characteristic(TimeCharacteristic.EventTime)
    env.set_parallelism(2)
    
    # Configurar Kafka Consumer
    kafka_props = {
        'bootstrap.servers': 'kafka:29092',
        'group.id': 'aml-feature-processor',
        'auto.offset.reset': 'latest'
    }
    
    kafka_consumer = FlinkKafkaConsumer(
        topics='transactions',
        deserialization_schema=SimpleStringSchema(),
        properties=kafka_props
    )
    
    # Configurar watermark strategy
    watermark_strategy = WatermarkStrategy.for_bounded_out_of_orderness(
        Time.seconds(10)
    ).with_timestamp_assigner(lambda event, timestamp: int(datetime.now().timestamp() * 1000))
    
    kafka_consumer.assign_timestamps_and_watermarks(watermark_strategy)
    
    # Criar stream de dados
    transaction_stream = env.add_source(kafka_consumer)
    
    # Processar por cliente
    customer_features = (transaction_stream
                        .key_by(lambda x: json.loads(x)['customer_id'])
                        .process(TransactionAggregator()))
    
    # Processar por estabelecimento  
    merchant_features = (transaction_stream
                        .key_by(lambda x: json.loads(x)['merchant_id'])
                        .process(MerchantAggregator()))
    
    # Sink para logs (opcional)
    customer_features.print("Customer Features")
    merchant_features.print("Merchant Features")
    
    return env

def main():
    """Função principal"""
    logger.info("Iniciando job Flink para processamento AML")
    
    try:
        # Criar job
        env = create_flink_job()
        
        # Executar
        env.execute("AML Feature Store Stream Processor")
        
    except Exception as e:
        logger.error(f"Erro na execução do job Flink: {e}")
        raise

if __name__ == "__main__":
    main()
