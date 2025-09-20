"""
Sistema de Monitoramento para AML Feature Store
Monitora performance, saúde dos componentes e métricas de negócio
"""

import time
import json
import logging
import asyncio
import aiohttp
import redis
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import psutil
import requests
from dataclasses import dataclass, asdict
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class HealthMetric:
    """Métrica de saúde de um componente"""
    component: str
    status: str  # healthy, warning, critical
    response_time_ms: float
    error_message: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class PerformanceMetric:
    """Métrica de performance"""
    metric_name: str
    value: float
    unit: str
    threshold_warning: float
    threshold_critical: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @property
    def status(self) -> str:
        if self.value >= self.threshold_critical:
            return "critical"
        elif self.value >= self.threshold_warning:
            return "warning"
        return "healthy"

class ComponentMonitor:
    """Monitor para componentes individuais do sistema"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.redis_client = None
        self.last_metrics = {}
        
    async def check_api_health(self) -> HealthMetric:
        """Verifica saúde da API FastAPI"""
        api_url = self.config.get('api_url', 'http://localhost:8000')
        
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{api_url}/health", timeout=10) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        # Verificar se todos os serviços estão saudáveis
                        services_status = data.get('services', {})
                        unhealthy_services = [k for k, v in services_status.items() if v != 'healthy']
                        
                        if unhealthy_services:
                            return HealthMetric(
                                component="api",
                                status="warning",
                                response_time_ms=response_time,
                                error_message=f"Serviços não saudáveis: {unhealthy_services}"
                            )
                        
                        return HealthMetric(
                            component="api",
                            status="healthy",
                            response_time_ms=response_time
                        )
                    else:
                        return HealthMetric(
                            component="api",
                            status="critical",
                            response_time_ms=response_time,
                            error_message=f"HTTP {response.status}"
                        )
                        
        except Exception as e:
            return HealthMetric(
                component="api",
                status="critical",
                response_time_ms=0,
                error_message=str(e)
            )
    
    def check_redis_health(self) -> HealthMetric:
        """Verifica saúde do Redis"""
        try:
            if not self.redis_client:
                self.redis_client = redis.Redis(
                    host=self.config.get('redis_host', 'localhost'),
                    port=self.config.get('redis_port', 6379),
                    decode_responses=True
                )
            
            start_time = time.time()
            self.redis_client.ping()
            response_time = (time.time() - start_time) * 1000
            
            # Verificar uso de memória
            info = self.redis_client.info('memory')
            used_memory_mb = info['used_memory'] / (1024 * 1024)
            
            if used_memory_mb > 1000:  # 1GB threshold
                return HealthMetric(
                    component="redis",
                    status="warning",
                    response_time_ms=response_time,
                    error_message=f"Alto uso de memória: {used_memory_mb:.1f}MB"
                )
            
            return HealthMetric(
                component="redis",
                status="healthy",
                response_time_ms=response_time
            )
            
        except Exception as e:
            return HealthMetric(
                component="redis",
                status="critical",
                response_time_ms=0,
                error_message=str(e)
            )
    
    def check_flink_health(self) -> HealthMetric:
        """Verifica saúde do Flink"""
        flink_url = self.config.get('flink_url', 'http://localhost:8081')
        
        try:
            start_time = time.time()
            response = requests.get(f"{flink_url}/overview", timeout=10)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                
                # Verificar se há jobs rodando
                running_jobs = data.get('jobs-running', 0)
                failed_jobs = data.get('jobs-failed', 0)
                
                if failed_jobs > 0:
                    return HealthMetric(
                        component="flink",
                        status="critical",
                        response_time_ms=response_time,
                        error_message=f"{failed_jobs} jobs falharam"
                    )
                
                if running_jobs == 0:
                    return HealthMetric(
                        component="flink",
                        status="warning",
                        response_time_ms=response_time,
                        error_message="Nenhum job em execução"
                    )
                
                return HealthMetric(
                    component="flink",
                    status="healthy",
                    response_time_ms=response_time
                )
            else:
                return HealthMetric(
                    component="flink",
                    status="critical",
                    response_time_ms=response_time,
                    error_message=f"HTTP {response.status_code}"
                )
                
        except Exception as e:
            return HealthMetric(
                component="flink",
                status="critical",
                response_time_ms=0,
                error_message=str(e)
            )
    
    def check_kafka_health(self) -> HealthMetric:
        """Verifica saúde do Kafka (via Docker)"""
        try:
            import docker
            client = docker.from_env()
            
            start_time = time.time()
            
            # Verificar se container Kafka está rodando
            kafka_container = client.containers.get('kafka')
            response_time = (time.time() - start_time) * 1000
            
            if kafka_container.status == 'running':
                return HealthMetric(
                    component="kafka",
                    status="healthy",
                    response_time_ms=response_time
                )
            else:
                return HealthMetric(
                    component="kafka",
                    status="critical",
                    response_time_ms=response_time,
                    error_message=f"Container status: {kafka_container.status}"
                )
                
        except Exception as e:
            return HealthMetric(
                component="kafka",
                status="critical",
                response_time_ms=0,
                error_message=str(e)
            )

class PerformanceMonitor:
    """Monitor de performance do sistema"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.redis_client = None
    
    def get_system_metrics(self) -> List[PerformanceMetric]:
        """Coleta métricas do sistema"""
        metrics = []
        
        # CPU Usage
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics.append(PerformanceMetric(
            metric_name="cpu_usage",
            value=cpu_percent,
            unit="percent",
            threshold_warning=70.0,
            threshold_critical=90.0
        ))
        
        # Memory Usage
        memory = psutil.virtual_memory()
        metrics.append(PerformanceMetric(
            metric_name="memory_usage",
            value=memory.percent,
            unit="percent",
            threshold_warning=80.0,
            threshold_critical=95.0
        ))
        
        # Disk Usage
        disk = psutil.disk_usage('/')
        metrics.append(PerformanceMetric(
            metric_name="disk_usage",
            value=disk.percent,
            unit="percent",
            threshold_warning=80.0,
            threshold_critical=95.0
        ))
        
        return metrics
    
    async def get_api_performance_metrics(self) -> List[PerformanceMetric]:
        """Coleta métricas de performance da API"""
        metrics = []
        api_url = self.config.get('api_url', 'http://localhost:8000')
        
        try:
            # Teste de latência
            start_time = time.time()
            
            test_transaction = {
                "customer_id": "CUST_MONITOR",
                "merchant_id": "MERCH_MONITOR",
                "amount": 100.0,
                "ip_address": "192.168.1.1"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{api_url}/predict", json=test_transaction) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    metrics.append(PerformanceMetric(
                        metric_name="api_response_time",
                        value=response_time,
                        unit="ms",
                        threshold_warning=100.0,
                        threshold_critical=500.0
                    ))
                    
                    if response.status == 200:
                        data = await response.json()
                        processing_time = data.get('processing_time_ms', 0)
                        
                        metrics.append(PerformanceMetric(
                            metric_name="api_processing_time",
                            value=processing_time,
                            unit="ms",
                            threshold_warning=50.0,
                            threshold_critical=200.0
                        ))
                        
        except Exception as e:
            logger.error(f"Erro ao coletar métricas da API: {e}")
        
        return metrics
    
    def get_redis_performance_metrics(self) -> List[PerformanceMetric]:
        """Coleta métricas de performance do Redis"""
        metrics = []
        
        try:
            if not self.redis_client:
                self.redis_client = redis.Redis(
                    host=self.config.get('redis_host', 'localhost'),
                    port=self.config.get('redis_port', 6379),
                    decode_responses=True
                )
            
            info = self.redis_client.info()
            
            # Conexões
            connected_clients = info.get('connected_clients', 0)
            metrics.append(PerformanceMetric(
                metric_name="redis_connections",
                value=connected_clients,
                unit="count",
                threshold_warning=100,
                threshold_critical=500
            ))
            
            # Operações por segundo
            ops_per_sec = info.get('instantaneous_ops_per_sec', 0)
            metrics.append(PerformanceMetric(
                metric_name="redis_ops_per_sec",
                value=ops_per_sec,
                unit="ops/sec",
                threshold_warning=10000,
                threshold_critical=50000
            ))
            
            # Uso de memória
            used_memory_mb = info.get('used_memory', 0) / (1024 * 1024)
            metrics.append(PerformanceMetric(
                metric_name="redis_memory_usage",
                value=used_memory_mb,
                unit="MB",
                threshold_warning=500,
                threshold_critical=1000
            ))
            
        except Exception as e:
            logger.error(f"Erro ao coletar métricas do Redis: {e}")
        
        return metrics

class AlertManager:
    """Gerenciador de alertas"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.alert_history = []
    
    def should_send_alert(self, metric) -> bool:
        """Verifica se deve enviar alerta (evita spam)"""
        # Implementar lógica de throttling
        recent_alerts = [
            alert for alert in self.alert_history 
            if alert['timestamp'] > datetime.now() - timedelta(minutes=15)
            and alert['component'] == getattr(metric, 'component', metric.metric_name)
        ]
        
        return len(recent_alerts) < 3  # Máximo 3 alertas por componente em 15 min
    
    def send_alert(self, metric, alert_type="warning"):
        """Envia alerta"""
        if not self.should_send_alert(metric):
            return
        
        alert_data = {
            'timestamp': datetime.now(),
            'component': getattr(metric, 'component', metric.metric_name),
            'type': alert_type,
            'message': self._format_alert_message(metric, alert_type)
        }
        
        self.alert_history.append(alert_data)
        
        # Enviar para diferentes canais
        if self.config.get('slack_webhook'):
            self._send_slack_alert(alert_data)
        
        if self.config.get('email_alerts'):
            self._send_email_alert(alert_data)
        
        # Log local
        logger.warning(f"ALERTA {alert_type.upper()}: {alert_data['message']}")
    
    def _format_alert_message(self, metric, alert_type) -> str:
        """Formata mensagem de alerta"""
        if hasattr(metric, 'component'):  # HealthMetric
            return f"🚨 {metric.component.upper()} - {alert_type.upper()}: {metric.error_message or 'Status crítico'}"
        else:  # PerformanceMetric
            return f"📊 {metric.metric_name.upper()} - {alert_type.upper()}: {metric.value:.2f}{metric.unit} (limite: {metric.threshold_warning})"
    
    def _send_slack_alert(self, alert_data):
        """Envia alerta para Slack"""
        try:
            webhook_url = self.config.get('slack_webhook')
            if not webhook_url:
                return
            
            payload = {
                "text": f"AML Feature Store Alert",
                "attachments": [{
                    "color": "danger" if "critical" in alert_data['type'] else "warning",
                    "fields": [{
                        "title": f"{alert_data['component']} - {alert_data['type'].upper()}",
                        "value": alert_data['message'],
                        "short": False
                    }],
                    "ts": int(alert_data['timestamp'].timestamp())
                }]
            }
            
            requests.post(webhook_url, json=payload, timeout=10)
            logger.info("Alerta enviado para Slack")
            
        except Exception as e:
            logger.error(f"Erro ao enviar alerta para Slack: {e}")
    
    def _send_email_alert(self, alert_data):
        """Envia alerta por email"""
        try:
            email_config = self.config.get('email_alerts', {})
            if not email_config.get('enabled'):
                return
            
            msg = MimeMultipart()
            msg['From'] = email_config['from']
            msg['To'] = ', '.join(email_config['to'])
            msg['Subject'] = f"AML Feature Store Alert - {alert_data['type'].upper()}"
            
            body = f"""
            Alerta do Sistema AML Feature Store
            
            Componente: {alert_data['component']}
            Tipo: {alert_data['type'].upper()}
            Timestamp: {alert_data['timestamp']}
            
            Detalhes:
            {alert_data['message']}
            
            ---
            Sistema de Monitoramento Automático
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            if email_config.get('use_tls'):
                server.starttls()
            if email_config.get('username'):
                server.login(email_config['username'], email_config['password'])
            
            server.send_message(msg)
            server.quit()
            
            logger.info("Alerta enviado por email")
            
        except Exception as e:
            logger.error(f"Erro ao enviar alerta por email: {e}")

class AMLMonitor:
    """Monitor principal do sistema AML"""
    
    def __init__(self, config_file='monitoring_config.json'):
        self.config = self._load_config(config_file)
        self.component_monitor = ComponentMonitor(self.config)
        self.performance_monitor = PerformanceMonitor(self.config)
        self.alert_manager = AlertManager(self.config)
        self.running = False
    
    def _load_config(self, config_file) -> Dict:
        """Carrega configuração do arquivo"""
        default_config = {
            'api_url': 'http://localhost:8000',
            'redis_host': 'localhost',
            'redis_port': 6379,
            'flink_url': 'http://localhost:8081',
            'monitoring_interval': 30,  # segundos
            'slack_webhook': None,
            'email_alerts': {
                'enabled': False,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'use_tls': True,
                'from': 'alerts@aml-system.com',
                'to': ['admin@aml-system.com'],
                'username': None,
                'password': None
            }
        }
        
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        except FileNotFoundError:
            logger.info(f"Arquivo de configuração {config_file} não encontrado, usando configuração padrão")
            # Criar arquivo de exemplo
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
        
        return default_config
    
    async def run_monitoring_cycle(self):
        """Executa um ciclo completo de monitoramento"""
        logger.info("🔍 Iniciando ciclo de monitoramento...")
        
        # Verificar saúde dos componentes
        health_checks = [
            await self.component_monitor.check_api_health(),
            self.component_monitor.check_redis_health(),
            self.component_monitor.check_flink_health(),
            self.component_monitor.check_kafka_health()
        ]
        
        # Coletar métricas de performance
        performance_metrics = []
        performance_metrics.extend(self.performance_monitor.get_system_metrics())
        performance_metrics.extend(await self.performance_monitor.get_api_performance_metrics())
        performance_metrics.extend(self.performance_monitor.get_redis_performance_metrics())
        
        # Processar alertas
        for health_metric in health_checks:
            if health_metric.status in ['warning', 'critical']:
                self.alert_manager.send_alert(health_metric, health_metric.status)
        
        for perf_metric in performance_metrics:
            if perf_metric.status in ['warning', 'critical']:
                self.alert_manager.send_alert(perf_metric, perf_metric.status)
        
        # Log status geral
        healthy_components = sum(1 for h in health_checks if h.status == 'healthy')
        total_components = len(health_checks)
        
        logger.info(f"✅ Ciclo concluído: {healthy_components}/{total_components} componentes saudáveis")
        
        # Retornar métricas para dashboard
        return {
            'health_metrics': [asdict(h) for h in health_checks],
            'performance_metrics': [asdict(p) for p in performance_metrics],
            'timestamp': datetime.now().isoformat()
        }
    
    async def start_monitoring(self):
        """Inicia monitoramento contínuo"""
        self.running = True
        logger.info("🚀 Iniciando monitoramento contínuo do AML Feature Store...")
        
        while self.running:
            try:
                metrics = await self.run_monitoring_cycle()
                
                # Salvar métricas (opcional - pode ser enviado para InfluxDB, Prometheus, etc.)
                with open(f"metrics_{datetime.now().strftime('%Y%m%d')}.json", 'a') as f:
                    f.write(json.dumps(metrics) + '\n')
                
                # Aguardar próximo ciclo
                await asyncio.sleep(self.config['monitoring_interval'])
                
            except KeyboardInterrupt:
                logger.info("⏹️ Monitoramento interrompido pelo usuário")
                break
            except Exception as e:
                logger.error(f"❌ Erro no ciclo de monitoramento: {e}")
                await asyncio.sleep(10)  # Aguardar antes de tentar novamente
        
        self.running = False
    
    def stop_monitoring(self):
        """Para o monitoramento"""
        self.running = False

async def main():
    """Função principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor do AML Feature Store')
    parser.add_argument('--config', default='monitoring_config.json', help='Arquivo de configuração')
    parser.add_argument('--once', action='store_true', help='Executar apenas um ciclo')
    
    args = parser.parse_args()
    
    monitor = AMLMonitor(args.config)
    
    if args.once:
        # Executar apenas um ciclo
        metrics = await monitor.run_monitoring_cycle()
        print(json.dumps(metrics, indent=2, default=str))
    else:
        # Monitoramento contínuo
        try:
            await monitor.start_monitoring()
        except KeyboardInterrupt:
            logger.info("Monitoramento finalizado")

if __name__ == "__main__":
    asyncio.run(main())
