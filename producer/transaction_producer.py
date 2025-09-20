import json
import time
import random
import uuid
from datetime import datetime, timedelta
from kafka import KafkaProducer
from kafka.errors import KafkaError
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransactionProducer:
    def __init__(self, bootstrap_servers='localhost:9094', topic='transactions'):
        """
        Inicializa o produtor de transações para Kafka
        
        Args:
            bootstrap_servers: Endereço do servidor Kafka
            topic: Tópico para enviar as mensagens
        """
        self.topic = topic
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            acks='all',  # Aguardar confirmação de todos os replicas
            retries=3,
            batch_size=16384,
            linger_ms=10,  # Aguardar 10ms para batch
            buffer_memory=33554432
        )
        
        # Dados sintéticos para simulação realista
        self.customer_ids = [f"CUST_{i:06d}" for i in range(1, 1001)]
        self.merchant_ids = [f"MERCH_{i:05d}" for i in range(1, 501)]
        self.ip_ranges = [
            "192.168.1.",  # Rede doméstica comum
            "10.0.0.",     # Rede corporativa
            "172.16.0.",   # Rede privada
            "203.0.113.",  # IP público exemplo
        ]
        
        # Padrões de comportamento por cliente (alguns são mais suspeitos)
        self.customer_profiles = self._generate_customer_profiles()
        
        logger.info(f"Produtor inicializado para tópico '{topic}'")

    def _generate_customer_profiles(self):
        """Gera perfis de comportamento para clientes"""
        profiles = {}
        
        for customer_id in self.customer_ids:
            # 5% dos clientes têm comportamento suspeito
            is_high_risk = random.random() < 0.05
            
            profile = {
                'is_high_risk': is_high_risk,
                'avg_transaction_amount': random.uniform(50, 500) if not is_high_risk else random.uniform(1000, 5000),
                'transaction_frequency': random.uniform(0.1, 2.0) if not is_high_risk else random.uniform(5.0, 20.0),
                'preferred_merchants': random.sample(self.merchant_ids, random.randint(3, 10)),
                'usual_ips': [f"{random.choice(self.ip_ranges)}{random.randint(1, 254)}" for _ in range(random.randint(1, 3))]
            }
            
            profiles[customer_id] = profile
            
        return profiles

    def generate_transaction(self, customer_id=None):
        """
        Gera uma transação sintética realista
        
        Args:
            customer_id: ID específico do cliente (opcional)
            
        Returns:
            dict: Dados da transação
        """
        if customer_id is None:
            # Distribuição realista: alguns clientes são muito mais ativos
            if random.random() < 0.3:  # 30% das transações vêm dos top 10% clientes
                customer_id = random.choice(self.customer_ids[:100])
            else:
                customer_id = random.choice(self.customer_ids)
        
        profile = self.customer_profiles.get(customer_id, {})
        
        # Valor da transação baseado no perfil
        base_amount = profile.get('avg_transaction_amount', 100)
        
        if profile.get('is_high_risk', False):
            # Clientes de alto risco: valores mais variáveis e potencialmente altos
            if random.random() < 0.2:  # 20% das transações são muito altas
                amount = random.uniform(base_amount * 5, base_amount * 20)
            else:
                amount = random.uniform(base_amount * 0.5, base_amount * 2)
        else:
            # Clientes normais: distribuição mais estável
            amount = max(1.0, random.normalvariate(base_amount, base_amount * 0.3))
        
        amount = round(amount, 2)
        
        # Estabelecimento baseado no perfil
        preferred_merchants = profile.get('preferred_merchants', self.merchant_ids)
        if random.random() < 0.7:  # 70% das transações em estabelecimentos conhecidos
            merchant_id = random.choice(preferred_merchants)
        else:
            merchant_id = random.choice(self.merchant_ids)
        
        # IP baseado no perfil
        usual_ips = profile.get('usual_ips', [])
        if usual_ips and random.random() < 0.8:  # 80% das transações de IPs conhecidos
            ip_address = random.choice(usual_ips)
        else:
            # IP novo/suspeito
            ip_range = random.choice(self.ip_ranges)
            ip_address = f"{ip_range}{random.randint(1, 254)}"
        
        # Timestamp atual
        event_timestamp = datetime.utcnow()
        
        # Adicionar ruído temporal para simular latência de rede
        network_delay = random.uniform(0.001, 0.1)  # 1ms a 100ms
        
        transaction = {
            'transaction_id': str(uuid.uuid4()),
            'customer_id': customer_id,
            'merchant_id': merchant_id,
            'amount': amount,
            'ip_address': ip_address,
            'event_timestamp': event_timestamp.isoformat() + 'Z',
            'processing_timestamp': (event_timestamp + timedelta(seconds=network_delay)).isoformat() + 'Z',
            'is_weekend': event_timestamp.weekday() >= 5,
            'hour_of_day': event_timestamp.hour,
            'is_night_transaction': event_timestamp.hour < 6 or event_timestamp.hour > 22,
            'profile_risk_level': 'HIGH' if profile.get('is_high_risk', False) else 'NORMAL'
        }
        
        return transaction

    def send_transaction(self, transaction):
        """
        Envia uma transação para o Kafka
        
        Args:
            transaction: Dados da transação
        """
        try:
            # Usar customer_id como chave para garantir ordem por cliente
            future = self.producer.send(
                self.topic, 
                key=transaction['customer_id'],
                value=transaction
            )
            
            # Callback para sucesso/erro
            future.add_callback(self._on_send_success)
            future.add_errback(self._on_send_error)
            
            return future
            
        except Exception as e:
            logger.error(f"Erro ao enviar transação: {e}")
            raise

    def _on_send_success(self, record_metadata):
        """Callback para envio bem-sucedido"""
        logger.debug(f"Transação enviada para tópico {record_metadata.topic} "
                    f"partição {record_metadata.partition} "
                    f"offset {record_metadata.offset}")

    def _on_send_error(self, excp):
        """Callback para erro no envio"""
        logger.error(f"Erro ao enviar transação: {excp}")

    def run_continuous_stream(self, transactions_per_second=10, duration_minutes=None):
        """
        Executa um stream contínuo de transações
        
        Args:
            transactions_per_second: Taxa de transações por segundo
            duration_minutes: Duração em minutos (None = infinito)
        """
        logger.info(f"Iniciando stream de {transactions_per_second} transações/segundo")
        
        start_time = time.time()
        transaction_count = 0
        
        try:
            while True:
                # Verificar duração se especificada
                if duration_minutes and (time.time() - start_time) > (duration_minutes * 60):
                    break
                
                # Gerar e enviar transação
                transaction = self.generate_transaction()
                self.send_transaction(transaction)
                
                transaction_count += 1
                
                # Log periódico
                if transaction_count % 100 == 0:
                    logger.info(f"Transações enviadas: {transaction_count}")
                
                # Controlar taxa de envio
                time.sleep(1.0 / transactions_per_second)
                
        except KeyboardInterrupt:
            logger.info("Interrompido pelo usuário")
        except Exception as e:
            logger.error(f"Erro no stream: {e}")
        finally:
            # Garantir que todas as mensagens sejam enviadas
            self.producer.flush()
            logger.info(f"Stream finalizado. Total de transações: {transaction_count}")

    def run_burst_scenario(self, customer_id, burst_size=50, burst_duration_seconds=30):
        """
        Simula um cenário de rajada suspeita para um cliente específico
        
        Args:
            customer_id: ID do cliente
            burst_size: Número de transações na rajada
            burst_duration_seconds: Duração da rajada em segundos
        """
        logger.info(f"Iniciando cenário de rajada para cliente {customer_id}")
        
        interval = burst_duration_seconds / burst_size
        
        for i in range(burst_size):
            transaction = self.generate_transaction(customer_id)
            # Tornar transações da rajada mais suspeitas
            transaction['amount'] = random.uniform(1000, 5000)
            transaction['burst_sequence'] = i + 1
            
            self.send_transaction(transaction)
            
            if i < burst_size - 1:  # Não aguardar após a última transação
                time.sleep(interval)
        
        self.producer.flush()
        logger.info(f"Rajada concluída: {burst_size} transações em {burst_duration_seconds}s")

    def close(self):
        """Fecha o produtor"""
        self.producer.close()
        logger.info("Produtor fechado")

def main():
    """Função principal para execução standalone"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Produtor de transações para Kafka')
    parser.add_argument('--kafka-servers', default='localhost:9094', 
                       help='Endereço dos servidores Kafka')
    parser.add_argument('--topic', default='transactions', 
                       help='Tópico Kafka')
    parser.add_argument('--rate', type=float, default=10.0,
                       help='Transações por segundo')
    parser.add_argument('--duration', type=int, 
                       help='Duração em minutos (padrão: infinito)')
    parser.add_argument('--burst-customer', 
                       help='ID do cliente para cenário de rajada')
    parser.add_argument('--burst-size', type=int, default=50,
                       help='Tamanho da rajada')
    
    args = parser.parse_args()
    
    # Criar produtor
    producer = TransactionProducer(
        bootstrap_servers=args.kafka_servers,
        topic=args.topic
    )
    
    try:
        if args.burst_customer:
            # Cenário de rajada
            producer.run_burst_scenario(
                customer_id=args.burst_customer,
                burst_size=args.burst_size
            )
        else:
            # Stream contínuo
            producer.run_continuous_stream(
                transactions_per_second=args.rate,
                duration_minutes=args.duration
            )
    finally:
        producer.close()

if __name__ == "__main__":
    main()
