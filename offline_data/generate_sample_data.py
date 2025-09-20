import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import uuid

def generate_sample_transactions(num_transactions=10000):
    """Gera dados de transações sintéticas para o offline store"""
    
    # Configurar seed para reprodutibilidade
    np.random.seed(42)
    random.seed(42)
    
    # Listas de dados sintéticos
    customer_ids = [f"CUST_{i:06d}" for i in range(1, 1001)]  # 1000 clientes
    merchant_ids = [f"MERCH_{i:05d}" for i in range(1, 501)]  # 500 estabelecimentos
    
    # IPs simulados
    ip_addresses = [f"192.168.{random.randint(1,255)}.{random.randint(1,255)}" for _ in range(200)]
    
    transactions = []
    
    # Data base - últimos 30 dias
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    for _ in range(num_transactions):
        # Timestamp aleatório nos últimos 30 dias
        random_timestamp = start_date + timedelta(
            seconds=random.randint(0, int((end_date - start_date).total_seconds()))
        )
        
        # Cliente com distribuição realista (alguns clientes mais ativos)
        if random.random() < 0.2:  # 20% dos clientes são muito ativos
            customer_id = random.choice(customer_ids[:100])  # Top 100 clientes
        else:
            customer_id = random.choice(customer_ids)
        
        # Valor da transação com distribuição log-normal (mais realista)
        amount = max(1.0, np.random.lognormal(mean=3.0, sigma=1.5))
        amount = round(amount, 2)
        
        # Estabelecimento
        merchant_id = random.choice(merchant_ids)
        
        # IP com padrão realista (a maioria dos clientes usa poucos IPs)
        if random.random() < 0.8:  # 80% das transações vêm de IPs conhecidos
            ip_address = f"192.168.1.{random.randint(1,50)}"  # IPs domésticos
        else:
            ip_address = random.choice(ip_addresses)
        
        # Adicionar alguns padrões suspeitos intencionalmente
        is_suspicious = random.random() < 0.05  # 5% de transações suspeitas
        
        if is_suspicious:
            # Transações suspeitas: valores altos, horários estranhos, etc.
            if random.random() < 0.5:
                amount = random.uniform(5000, 50000)  # Valores muito altos
            
            # Horários noturnos
            if random.random() < 0.3:
                hour = random.randint(2, 5)  # Madrugada
                random_timestamp = random_timestamp.replace(hour=hour)
        
        transaction = {
            'transaction_id': str(uuid.uuid4()),
            'customer_id': customer_id,
            'merchant_id': merchant_id,
            'amount': amount,
            'ip_address': ip_address,
            'event_timestamp': random_timestamp,
            'is_weekend': random_timestamp.weekday() >= 5,
            'hour_of_day': random_timestamp.hour,
            'is_suspicious': is_suspicious
        }
        
        transactions.append(transaction)
    
    # Criar DataFrame e ordenar por timestamp
    df = pd.DataFrame(transactions)
    df = df.sort_values('event_timestamp').reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    print("Gerando dados de transações sintéticas...")
    
    # Gerar dados
    df = generate_sample_transactions(50000)
    
    # Salvar como Parquet
    output_path = "transactions.parquet"
    df.to_parquet(output_path, index=False)
    
    print(f"Dados salvos em {output_path}")
    print(f"Total de transações: {len(df)}")
    print(f"Período: {df['event_timestamp'].min()} até {df['event_timestamp'].max()}")
    print(f"Clientes únicos: {df['customer_id'].nunique()}")
    print(f"Estabelecimentos únicos: {df['merchant_id'].nunique()}")
    print(f"Transações suspeitas: {df['is_suspicious'].sum()} ({df['is_suspicious'].mean()*100:.1f}%)")
    
    # Mostrar estatísticas básicas
    print("\nEstatísticas dos valores:")
    print(df['amount'].describe())
