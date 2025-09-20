"""
Real-time AML Feature Processor usando PyFlink
Processa stream do Kafka e calcula features em tempo real
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

from pyflink.common import Types, Time, WatermarkStrategy, Configuration
from pyflink.common.serialization import SimpleStringSchema, JsonRowSerializationSchema
from pyflink.datastream import StreamExecutionEnvironment, TimeCharacteristic
from pyflink.datastream.connectors import FlinkKafkaConsumer, FlinkKafkaProducer
from pyflink.datastream.functions import KeyedProcessFunction, RuntimeContext, ProcessFunction
from pyflink.datastream.state import ValueStateDescriptor, MapStateDescriptor
from pyflink.datastream.window import TumblingEventTimeWindows, SlidingEventTimeWindows
from pyflink.table import StreamTableEnvironment, EnvironmentSettings
from pyflink.table.expressions import col, lit
import redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedFeatureProcessor(KeyedProcessFunction):
    """
    Processador avançado de features com múltiplas janelas temporais
    """
    
    def __init__(self, redis_config: Dict[str, Any]):
        self.redis_config = redis_config
        self.redis_client = None
        
        # Estados para diferentes janelas temporais
        self.transaction_state = None
        self.ip_state = None
        self.merchant_state = None
        self.amount_state = None
        
        # Janelas temporais em milissegundos
        self.windows = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '24h': 24 * 60 * 60 * 1000
        }
    
    def open(self, runtime_context: RuntimeContext):
        """Inicialização da função"""
        # Conectar ao Redis
        self.redis_client = redis.Redis(
            host=self.redis_config.get('host', 'localhost'),
            port=self.redis_config.get('port', 6379),
            decode_responses=True
        )
        
        # Configurar estados
        self.transaction_state = runtime_context.get_map_state(
            MapStateDescriptor("transactions", Types.LONG(), Types.STRING())
        )
        
        self.ip_state = runtime_context.get_map_state(
            MapStateDescriptor("ips", Types.LONG(), Types.STRING())
        )
        
        self.merchant_state = runtime_context.get_map_state(
            MapStateDescriptor("merchants", Types.LONG(), Types.STRING())
        )
        
        self.amount_state = runtime_context.get_map_state(
            MapStateDescriptor("amounts", Types.LONG(), Types.DOUBLE())
        )
        
        logger.info("AdvancedFeatureProcessor inicializado")
    
    def process_element(self, value, ctx):
        """Processa cada transação"""
        try:
            # Parse da transação
            if isinstance(value, str):
                transaction = json.loads(value)
            else:
                transaction = value
            
            customer_id = transaction['customer_id']
            amount = float(transaction['amount'])
            ip_address = transaction['ip_address']
            merchant_id = transaction['merchant_id']
            current_time = ctx.timestamp()
            
            # Limpar estados antigos (manter apenas últimas 24h)
            self._cleanup_old_states(current_time)
            
            # Adicionar nova transação aos estados
            self.transaction_state.put(current_time, json.dumps({
                'amount': amount,
                'ip': ip_address,
                'merchant': merchant_id,
                'timestamp': current_time
            }))
            
            self.ip_state.put(current_time, ip_address)
            self.merchant_state.put(current_time, merchant_id)
            self.amount_state.put(current_time, amount)
            
            # Calcular features para todas as janelas
            features = self._calculate_all_features(current_time)
            
            # Adicionar informações da transação atual
            features.update({
                'customer_id': customer_id,
                'current_amount': amount,
                'current_merchant': merchant_id,
                'current_ip': ip_address,
                'processing_timestamp': current_time
            })
            
            # Salvar no Redis
            self._save_to_redis(customer_id, features)
            
            # Emitir resultado
            yield json.dumps(features)
            
        except Exception as e:
            logger.error(f"Erro ao processar transação: {e}")
    
    def _cleanup_old_states(self, current_time):
        """Remove estados antigos (>24h)"""
        cutoff_time = current_time - self.windows['24h']
        
        # Limpar transações antigas
        for timestamp in list(self.transaction_state.keys()):
            if timestamp < cutoff_time:
                self.transaction_state.remove(timestamp)
        
        # Limpar IPs antigos
        for timestamp in list(self.ip_state.keys()):
            if timestamp < cutoff_time:
                self.ip_state.remove(timestamp)
        
        # Limpar merchants antigos
        for timestamp in list(self.merchant_state.keys()):
            if timestamp < cutoff_time:
                self.merchant_state.remove(timestamp)
        
        # Limpar amounts antigos
        for timestamp in list(self.amount_state.keys()):
            if timestamp < cutoff_time:
                self.amount_state.remove(timestamp)
    
    def _calculate_all_features(self, current_time):
        """Calcula features para todas as janelas temporais"""
        features = {}
        
        for window_name, window_ms in self.windows.items():
            cutoff_time = current_time - window_ms
            
            # Filtrar dados na janela
            window_transactions = []
            window_ips = set()
            window_merchants = set()
            window_amounts = []
            
            # Coletar dados da janela
            for timestamp in self.transaction_state.keys():
                if timestamp > cutoff_time:
                    txn_data = json.loads(self.transaction_state.get(timestamp))
                    window_transactions.append(txn_data)
                    window_ips.add(txn_data['ip'])
                    window_merchants.add(txn_data['merchant'])
                    window_amounts.append(txn_data['amount'])
            
            # Calcular features básicas
            features.update(self._calculate_window_features(window_name, window_transactions, window_ips, window_merchants, window_amounts))
        
        return features
    
    def _calculate_window_features(self, window_name, transactions, ips, merchants, amounts):
        """Calcula features para uma janela específica"""
        features = {}
        
        # Features básicas
        features[f'txn_count_{window_name}'] = len(transactions)
        features[f'txn_amount_sum_{window_name}'] = sum(amounts) if amounts else 0.0
        features[f'unique_ips_{window_name}'] = len(ips)
        features[f'unique_merchants_{window_name}'] = len(merchants)
        
        if amounts:
            features[f'avg_txn_amount_{window_name}'] = sum(amounts) / len(amounts)
            features[f'max_txn_amount_{window_name}'] = max(amounts)
            features[f'min_txn_amount_{window_name}'] = min(amounts)
            
            # Desvio padrão
            if len(amounts) > 1:
                mean_amount = sum(amounts) / len(amounts)
                variance = sum((x - mean_amount) ** 2 for x in amounts) / len(amounts)
                features[f'std_txn_amount_{window_name}'] = variance ** 0.5
            else:
                features[f'std_txn_amount_{window_name}'] = 0.0
        else:
            features[f'avg_txn_amount_{window_name}'] = 0.0
            features[f'max_txn_amount_{window_name}'] = 0.0
            features[f'min_txn_amount_{window_name}'] = 0.0
            features[f'std_txn_amount_{window_name}'] = 0.0
        
        # Features de velocidade
        window_hours = int(window_name.replace('m', '').replace('h', '')) / 60.0 if 'm' in window_name else int(window_name.replace('h', ''))
        features[f'velocity_score_{window_name}'] = len(transactions) / window_hours if window_hours > 0 else 0.0
        
        # Features de concentração (Gini coefficient)
        if len(amounts) > 1:
            sorted_amounts = sorted(amounts)
            n = len(sorted_amounts)
            cumsum = 0
            for i, amount in enumerate(sorted_amounts):
                cumsum += amount
            
            total_sum = sum(amounts)
            if total_sum > 0:
                gini_sum = sum((i + 1) * amount for i, amount in enumerate(sorted_amounts))
                gini = (2 * gini_sum) / (n * total_sum) - (n + 1) / n
                features[f'amount_gini_{window_name}'] = gini
            else:
                features[f'amount_gini_{window_name}'] = 0.0
        else:
            features[f'amount_gini_{window_name}'] = 0.0
        
        return features
    
    def _save_to_redis(self, customer_id, features):
        """Salva features no Redis"""
        try:
            redis_key = f"aml_features:{customer_id}"
            
            # Salvar features com TTL de 24 horas
            pipeline = self.redis_client.pipeline()
            
            for feature_name, value in features.items():
                if feature_name != 'customer_id':  # Não salvar a chave como valor
                    pipeline.hset(redis_key, feature_name, str(value))
            
            pipeline.expire(redis_key, 86400)  # 24 horas
            pipeline.execute()
            
            logger.debug(f"Features salvas no Redis para cliente {customer_id}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar no Redis: {e}")

class GraphFeatureProcessor(ProcessFunction):
    """
    Processador de features baseadas em grafos
    """
    
    def __init__(self, redis_config: Dict[str, Any]):
        self.redis_config = redis_config
        self.redis_client = None
        self.graph_state = None
    
    def open(self, runtime_context: RuntimeContext):
        """Inicialização"""
        self.redis_client = redis.Redis(
            host=self.redis_config.get('host', 'localhost'),
            port=self.redis_config.get('port', 6379),
            decode_responses=True
        )
        
        # Estado para manter grafo de relacionamentos
        self.graph_state = runtime_context.get_map_state(
            MapStateDescriptor("graph_edges", Types.STRING(), Types.STRING())
        )
    
    def process_element(self, value, ctx):
        """Processa transação para features de grafo"""
        try:
            if isinstance(value, str):
                transaction = json.loads(value)
            else:
                transaction = value
            
            customer_id = transaction['customer_id']
            merchant_id = transaction['merchant_id']
            ip_address = transaction['ip_address']
            
            # Criar edges no grafo
            # Customer -> Merchant
            customer_merchant_key = f"cm:{customer_id}:{merchant_id}"
            existing_weight = self.graph_state.get(customer_merchant_key)
            new_weight = (int(existing_weight) if existing_weight else 0) + 1
            self.graph_state.put(customer_merchant_key, str(new_weight))
            
            # Customer -> IP
            customer_ip_key = f"ci:{customer_id}:{ip_address}"
            existing_weight = self.graph_state.get(customer_ip_key)
            new_weight = (int(existing_weight) if existing_weight else 0) + 1
            self.graph_state.put(customer_ip_key, str(new_weight))
            
            # Calcular features de grafo
            graph_features = self._calculate_graph_features(customer_id)
            
            # Salvar no Redis
            self._save_graph_features_to_redis(customer_id, graph_features)
            
            yield json.dumps({
                'customer_id': customer_id,
                'graph_features': graph_features
            })
            
        except Exception as e:
            logger.error(f"Erro no processamento de grafo: {e}")
    
    def _calculate_graph_features(self, customer_id):
        """Calcula features baseadas em grafo"""
        features = {}
        
        # Contar conexões diretas
        merchant_connections = 0
        ip_connections = 0
        
        for key in self.graph_state.keys():
            if key.startswith(f"cm:{customer_id}:"):
                merchant_connections += 1
            elif key.startswith(f"ci:{customer_id}:"):
                ip_connections += 1
        
        features['graph_merchant_degree'] = merchant_connections
        features['graph_ip_degree'] = ip_connections
        features['graph_total_degree'] = merchant_connections + ip_connections
        
        return features
    
    def _save_graph_features_to_redis(self, customer_id, features):
        """Salva features de grafo no Redis"""
        try:
            redis_key = f"aml_graph_features:{customer_id}"
            
            pipeline = self.redis_client.pipeline()
            for feature_name, value in features.items():
                pipeline.hset(redis_key, feature_name, str(value))
            
            pipeline.expire(redis_key, 86400)
            pipeline.execute()
            
        except Exception as e:
            logger.error(f"Erro ao salvar features de grafo: {e}")

def create_advanced_flink_job():
    """Cria job Flink avançado com múltiplos processadores"""
    
    # Configuração do ambiente
    config = Configuration()
    config.set_string("python.fn-execution.bundle.size", "1000")
    config.set_string("python.fn-execution.bundle.time", "1000")
    
    env = StreamExecutionEnvironment.get_execution_environment(config)
    env.set_stream_time_characteristic(TimeCharacteristic.EventTime)
    env.set_parallelism(2)
    
    # Configurações do Kafka
    kafka_props = {
        'bootstrap.servers': 'localhost:9094',
        'group.id': 'aml-advanced-processor',
        'auto.offset.reset': 'latest'
    }
    
    # Consumer do Kafka
    kafka_consumer = FlinkKafkaConsumer(
        topics='transactions',
        deserialization_schema=SimpleStringSchema(),
        properties=kafka_props
    )
    
    # Configurar watermarks
    watermark_strategy = WatermarkStrategy.for_bounded_out_of_orderness(
        Time.seconds(5)
    ).with_timestamp_assigner(
        lambda event, timestamp: int(datetime.now().timestamp() * 1000)
    )
    
    kafka_consumer.assign_timestamps_and_watermarks(watermark_strategy)
    
    # Stream principal
    transaction_stream = env.add_source(kafka_consumer)
    
    # Configuração do Redis
    redis_config = {
        'host': 'localhost',
        'port': 6379
    }
    
    # Processamento de features por cliente
    customer_features = (transaction_stream
                        .key_by(lambda x: json.loads(x)['customer_id'])
                        .process(AdvancedFeatureProcessor(redis_config)))
    
    # Processamento de features de grafo
    graph_features = (transaction_stream
                     .process(GraphFeatureProcessor(redis_config)))
    
    # Output streams
    customer_features.print("Customer Features")
    graph_features.print("Graph Features")
    
    return env

def main():
    """Função principal"""
    logger.info("Iniciando job Flink avançado para AML")
    
    try:
        # Criar e executar job
        env = create_advanced_flink_job()
        env.execute("Advanced AML Real-Time Feature Processor")
        
    except Exception as e:
        logger.error(f"Erro na execução do job Flink: {e}")
        raise

if __name__ == "__main__":
    main()
