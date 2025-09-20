"""
Script de teste de carga para a API AML Feature Store
"""

import asyncio
import aiohttp
import time
import random
import argparse
from datetime import datetime
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
import threading

class LoadTester:
    def __init__(self, base_url="http://localhost:8000", concurrent_users=10):
        self.base_url = base_url
        self.concurrent_users = concurrent_users
        self.results = []
        self.errors = []
        self.lock = threading.Lock()
        
    def generate_transaction(self):
        """Gera uma transação aleatória para teste"""
        return {
            "customer_id": f"CUST_{random.randint(1, 1000):06d}",
            "merchant_id": f"MERCH_{random.randint(1, 500):05d}",
            "amount": round(random.uniform(10, 5000), 2),
            "ip_address": f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
        }
    
    async def make_request(self, session, endpoint="/predict"):
        """Faz uma requisição HTTP"""
        transaction = self.generate_transaction()
        
        start_time = time.time()
        
        try:
            async with session.post(f"{self.base_url}{endpoint}", json=transaction) as response:
                end_time = time.time()
                response_time = (end_time - start_time) * 1000  # ms
                
                if response.status == 200:
                    data = await response.json()
                    
                    with self.lock:
                        self.results.append({
                            'response_time': response_time,
                            'status_code': response.status,
                            'risk_score': data.get('risk_score', 0),
                            'processing_time_ms': data.get('processing_time_ms', 0),
                            'timestamp': datetime.now()
                        })
                else:
                    with self.lock:
                        self.errors.append({
                            'status_code': response.status,
                            'response_time': response_time,
                            'timestamp': datetime.now()
                        })
                        
        except Exception as e:
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            
            with self.lock:
                self.errors.append({
                    'error': str(e),
                    'response_time': response_time,
                    'timestamp': datetime.now()
                })
    
    async def run_user_session(self, user_id, duration_seconds, requests_per_second):
        """Simula uma sessão de usuário"""
        print(f"🚀 Iniciando usuário {user_id}")
        
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            request_count = 0
            
            while (time.time() - start_time) < duration_seconds:
                await self.make_request(session)
                request_count += 1
                
                # Controlar taxa de requisições
                if requests_per_second > 0:
                    await asyncio.sleep(1.0 / requests_per_second)
                
                # Log progresso a cada 50 requisições
                if request_count % 50 == 0:
                    elapsed = time.time() - start_time
                    print(f"👤 Usuário {user_id}: {request_count} requisições em {elapsed:.1f}s")
        
        print(f"✅ Usuário {user_id} concluído: {request_count} requisições")
    
    async def run_load_test(self, duration_seconds=60, requests_per_second=1):
        """Executa o teste de carga"""
        print(f"🔥 Iniciando teste de carga:")
        print(f"   • Usuários simultâneos: {self.concurrent_users}")
        print(f"   • Duração: {duration_seconds}s")
        print(f"   • Taxa por usuário: {requests_per_second} req/s")
        print(f"   • Taxa total estimada: {self.concurrent_users * requests_per_second} req/s")
        print()
        
        # Verificar se API está disponível
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health") as response:
                    if response.status != 200:
                        print(f"❌ API não está saudável: {response.status}")
                        return
        except Exception as e:
            print(f"❌ Erro ao conectar com API: {e}")
            return
        
        print("✅ API está disponível, iniciando teste...")
        
        # Executar usuários simultâneos
        tasks = []
        for user_id in range(self.concurrent_users):
            task = asyncio.create_task(
                self.run_user_session(user_id, duration_seconds, requests_per_second)
            )
            tasks.append(task)
        
        # Aguardar todos os usuários terminarem
        await asyncio.gather(*tasks)
        
        print("\n🏁 Teste de carga concluído!")
    
    def generate_report(self):
        """Gera relatório dos resultados"""
        if not self.results:
            print("❌ Nenhum resultado para gerar relatório")
            return
        
        # Calcular estatísticas
        response_times = [r['response_time'] for r in self.results]
        processing_times = [r['processing_time_ms'] for r in self.results if r['processing_time_ms'] > 0]
        risk_scores = [r['risk_score'] for r in self.results]
        
        total_requests = len(self.results)
        total_errors = len(self.errors)
        success_rate = (total_requests / (total_requests + total_errors)) * 100 if (total_requests + total_errors) > 0 else 0
        
        # Calcular percentis
        p50_response = statistics.median(response_times)
        p95_response = statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times)
        p99_response = statistics.quantiles(response_times, n=100)[98] if len(response_times) > 100 else max(response_times)
        
        # Calcular throughput
        if self.results:
            test_duration = (max(r['timestamp'] for r in self.results) - 
                           min(r['timestamp'] for r in self.results)).total_seconds()
            throughput = total_requests / test_duration if test_duration > 0 else 0
        else:
            throughput = 0
        
        print("\n" + "="*60)
        print("📊 RELATÓRIO DE TESTE DE CARGA")
        print("="*60)
        
        print(f"\n📈 ESTATÍSTICAS GERAIS:")
        print(f"   • Total de requisições: {total_requests:,}")
        print(f"   • Total de erros: {total_errors:,}")
        print(f"   • Taxa de sucesso: {success_rate:.1f}%")
        print(f"   • Throughput: {throughput:.1f} req/s")
        
        print(f"\n⏱️  TEMPOS DE RESPOSTA (ms):")
        print(f"   • Média: {statistics.mean(response_times):.1f}")
        print(f"   • Mediana (P50): {p50_response:.1f}")
        print(f"   • P95: {p95_response:.1f}")
        print(f"   • P99: {p99_response:.1f}")
        print(f"   • Mínimo: {min(response_times):.1f}")
        print(f"   • Máximo: {max(response_times):.1f}")
        
        if processing_times:
            print(f"\n🔄 TEMPO DE PROCESSAMENTO DA API (ms):")
            print(f"   • Média: {statistics.mean(processing_times):.1f}")
            print(f"   • Mediana: {statistics.median(processing_times):.1f}")
            print(f"   • Mínimo: {min(processing_times):.1f}")
            print(f"   • Máximo: {max(processing_times):.1f}")
        
        print(f"\n🎯 SCORES DE RISCO:")
        print(f"   • Score médio: {statistics.mean(risk_scores):.3f}")
        print(f"   • Score mediano: {statistics.median(risk_scores):.3f}")
        print(f"   • Transações de alto risco (>0.6): {sum(1 for s in risk_scores if s > 0.6)}")
        
        if self.errors:
            print(f"\n❌ ANÁLISE DE ERROS:")
            error_types = {}
            for error in self.errors:
                error_key = error.get('status_code', 'Exception')
                error_types[error_key] = error_types.get(error_key, 0) + 1
            
            for error_type, count in error_types.items():
                print(f"   • {error_type}: {count} ocorrências")
        
        # Recomendações
        print(f"\n💡 RECOMENDAÇÕES:")
        
        if p95_response > 100:
            print(f"   ⚠️  P95 de resposta alto ({p95_response:.1f}ms) - considere otimização")
        
        if success_rate < 99:
            print(f"   ⚠️  Taxa de sucesso baixa ({success_rate:.1f}%) - investigar erros")
        
        if throughput < self.concurrent_users * 0.8:
            print(f"   ⚠️  Throughput baixo - possível gargalo no sistema")
        
        if statistics.mean(response_times) > 50:
            print(f"   💡 Tempo médio de resposta pode ser melhorado")
        else:
            print(f"   ✅ Tempos de resposta estão bons")
        
        if success_rate > 99 and p95_response < 100:
            print(f"   🎉 Sistema está performando muito bem!")
        
        print("\n" + "="*60)

async def main():
    parser = argparse.ArgumentParser(description='Teste de carga para AML Feature Store API')
    parser.add_argument('--url', default='http://localhost:8000', help='URL base da API')
    parser.add_argument('--concurrent', type=int, default=10, help='Número de usuários simultâneos')
    parser.add_argument('--duration', type=int, default=60, help='Duração do teste em segundos')
    parser.add_argument('--rate', type=float, default=1.0, help='Requisições por segundo por usuário')
    
    args = parser.parse_args()
    
    # Criar e executar teste
    tester = LoadTester(
        base_url=args.url,
        concurrent_users=args.concurrent
    )
    
    try:
        await tester.run_load_test(
            duration_seconds=args.duration,
            requests_per_second=args.rate
        )
        
        # Gerar relatório
        tester.generate_report()
        
    except KeyboardInterrupt:
        print("\n⏹️  Teste interrompido pelo usuário")
        if tester.results:
            tester.generate_report()
    except Exception as e:
        print(f"\n❌ Erro durante o teste: {e}")

if __name__ == "__main__":
    asyncio.run(main())
