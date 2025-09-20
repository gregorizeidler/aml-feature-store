"""
Script para testar a API de inferência AML
"""

import requests
import json
import time
import random
from datetime import datetime

# Configuração da API
API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """Testa o endpoint de health check"""
    print("🔍 Testando health check...")
    
    response = requests.get(f"{API_BASE_URL}/health")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Health check OK: {data['status']}")
        print(f"   Serviços: {data['services']}")
    else:
        print(f"❌ Health check falhou: {response.status_code}")
    
    return response.status_code == 200

def test_single_prediction():
    """Testa predição individual"""
    print("\n🔍 Testando predição individual...")
    
    # Transação normal
    normal_transaction = {
        "customer_id": "CUST_000001",
        "merchant_id": "MERCH_00001", 
        "amount": 150.50,
        "ip_address": "192.168.1.100"
    }
    
    response = requests.post(f"{API_BASE_URL}/predict", json=normal_transaction)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Predição normal:")
        print(f"   Score de risco: {data['risk_score']:.3f}")
        print(f"   Nível: {data['risk_level']}")
        print(f"   Tempo de processamento: {data['processing_time_ms']:.2f}ms")
        print(f"   Explicação: {data['explanation']}")
    else:
        print(f"❌ Predição falhou: {response.status_code}")
        print(f"   Erro: {response.text}")
    
    return response.status_code == 200

def test_suspicious_transaction():
    """Testa transação suspeita"""
    print("\n🔍 Testando transação suspeita...")
    
    # Transação suspeita (valor alto)
    suspicious_transaction = {
        "customer_id": "CUST_000002",
        "merchant_id": "MERCH_00002",
        "amount": 15000.00,  # Valor alto
        "ip_address": "203.0.113.50"  # IP diferente
    }
    
    response = requests.post(f"{API_BASE_URL}/predict", json=suspicious_transaction)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Predição suspeita:")
        print(f"   Score de risco: {data['risk_score']:.3f}")
        print(f"   Nível: {data['risk_level']}")
        print(f"   Tempo de processamento: {data['processing_time_ms']:.2f}ms")
        print(f"   Explicação: {data['explanation']}")
    else:
        print(f"❌ Predição suspeita falhou: {response.status_code}")
    
    return response.status_code == 200

def test_get_features():
    """Testa busca de features"""
    print("\n🔍 Testando busca de features...")
    
    customer_id = "CUST_000001"
    response = requests.get(f"{API_BASE_URL}/features/{customer_id}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Features do cliente {customer_id}:")
        print(f"   Timestamp: {data['feature_timestamp']}")
        print(f"   Número de features: {len(data['features'])}")
        
        # Mostrar algumas features importantes
        important_features = [
            'txn_count_1h', 'txn_amount_sum_1h', 'unique_ips_1h', 'velocity_score_1h'
        ]
        
        for feature in important_features:
            if feature in data['features']:
                print(f"   {feature}: {data['features'][feature]}")
    else:
        print(f"❌ Busca de features falhou: {response.status_code}")
    
    return response.status_code == 200

def test_batch_prediction():
    """Testa predição em lote"""
    print("\n🔍 Testando predição em lote...")
    
    transactions = [
        {
            "customer_id": "CUST_000003",
            "merchant_id": "MERCH_00001",
            "amount": 50.00,
            "ip_address": "192.168.1.101"
        },
        {
            "customer_id": "CUST_000004", 
            "merchant_id": "MERCH_00002",
            "amount": 8000.00,  # Valor alto
            "ip_address": "10.0.0.50"
        },
        {
            "customer_id": "CUST_000005",
            "merchant_id": "MERCH_00003",
            "amount": 200.00,
            "ip_address": "192.168.1.102"
        }
    ]
    
    response = requests.post(f"{API_BASE_URL}/batch-predict", json=transactions)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Predição em lote:")
        print(f"   Total processado: {data['total_processed']}")
        
        for i, prediction in enumerate(data['predictions']):
            print(f"   Transação {i+1}: {prediction['risk_level']} "
                  f"(score: {prediction['risk_score']:.3f})")
    else:
        print(f"❌ Predição em lote falhou: {response.status_code}")
    
    return response.status_code == 200

def test_performance():
    """Testa performance da API"""
    print("\n🔍 Testando performance...")
    
    num_requests = 50
    times = []
    
    print(f"   Enviando {num_requests} requisições...")
    
    for i in range(num_requests):
        transaction = {
            "customer_id": f"CUST_{i:06d}",
            "merchant_id": f"MERCH_{random.randint(1, 100):05d}",
            "amount": round(random.uniform(10, 1000), 2),
            "ip_address": f"192.168.1.{random.randint(1, 254)}"
        }
        
        start_time = time.time()
        response = requests.post(f"{API_BASE_URL}/predict", json=transaction)
        end_time = time.time()
        
        if response.status_code == 200:
            times.append((end_time - start_time) * 1000)  # Converter para ms
        
        if (i + 1) % 10 == 0:
            print(f"   Processadas: {i + 1}/{num_requests}")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"✅ Resultados de performance:")
        print(f"   Requisições bem-sucedidas: {len(times)}/{num_requests}")
        print(f"   Tempo médio: {avg_time:.2f}ms")
        print(f"   Tempo mínimo: {min_time:.2f}ms")
        print(f"   Tempo máximo: {max_time:.2f}ms")
        print(f"   Throughput: {1000/avg_time:.1f} req/s")
    else:
        print("❌ Nenhuma requisição bem-sucedida")
    
    return len(times) > 0

def main():
    """Executa todos os testes"""
    print("🚀 Iniciando testes da API AML Feature Store")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_check),
        ("Predição Individual", test_single_prediction),
        ("Transação Suspeita", test_suspicious_transaction),
        ("Busca de Features", test_get_features),
        ("Predição em Lote", test_batch_prediction),
        ("Performance", test_performance)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Erro no teste {test_name}: {e}")
            results[test_name] = False
    
    # Resumo dos resultados
    print("\n" + "=" * 50)
    print("📊 RESUMO DOS TESTES")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASSOU" if passed_test else "❌ FALHOU"
        print(f"{test_name}: {status}")
    
    print(f"\nResultado geral: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 Todos os testes passaram!")
    else:
        print("⚠️  Alguns testes falharam. Verifique a configuração.")

if __name__ == "__main__":
    main()
