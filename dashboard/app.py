"""
Dashboard Executivo Real-Time para AML Feature Store
Interface web interativa com métricas em tempo real
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import redis
import json
import requests
from datetime import datetime, timedelta
import time
import asyncio
from typing import Dict, List, Any
import logging

# Configurar página
st.set_page_config(
    page_title="AML Feature Store Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .alert-low {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

class DashboardDataProvider:
    """Provedor de dados para o dashboard"""
    
    def __init__(self):
        self.redis_client = None
        self.api_base_url = "http://localhost:8000"
        self._connect_redis()
    
    def _connect_redis(self):
        """Conecta ao Redis"""
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.redis_client.ping()
        except Exception as e:
            st.error(f"❌ Erro ao conectar ao Redis: {e}")
    
    def get_api_health(self) -> Dict[str, Any]:
        """Verifica saúde da API"""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def get_recent_predictions(self, limit: int = 100) -> List[Dict]:
        """Simula predições recentes (em produção viria de um log/database)"""
        # Para demonstração, vamos gerar dados sintéticos
        predictions = []
        
        for i in range(limit):
            timestamp = datetime.now() - timedelta(minutes=i*2)
            risk_score = np.random.beta(2, 5)  # Distribuição realista de scores
            
            if risk_score > 0.7:
                risk_level = "HIGH"
            elif risk_score > 0.4:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            predictions.append({
                'timestamp': timestamp,
                'customer_id': f"CUST_{np.random.randint(1, 1000):06d}",
                'amount': np.random.lognormal(6, 1.5),
                'risk_score': risk_score,
                'risk_level': risk_level,
                'processing_time_ms': np.random.normal(25, 10)
            })
        
        return sorted(predictions, key=lambda x: x['timestamp'], reverse=True)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Obtém métricas do sistema"""
        health = self.get_api_health()
        
        # Simular métricas (em produção viria do monitoramento)
        return {
            'api_status': health.get('status', 'unknown'),
            'total_predictions_today': np.random.randint(5000, 15000),
            'avg_response_time_ms': np.random.normal(25, 5),
            'high_risk_alerts': np.random.randint(50, 200),
            'system_uptime_hours': np.random.uniform(720, 744),  # ~30 dias
            'throughput_per_second': np.random.normal(100, 20)
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Obtém importância das features do modelo atual"""
        # Simular importância de features
        features = [
            'unique_ips_1h', 'velocity_score_1h', 'txn_amount_sum_5m',
            'graph_degree_centrality', 'temporal_burst_ratio', 'behavioral_amount_zscore',
            'txn_count_1h', 'amount_deviation_score_1h', 'night_txn_ratio_1h',
            'merchant_concentration_hhi', 'graph_clustering_coefficient'
        ]
        
        # Gerar importâncias que somam 1
        importances = np.random.dirichlet(np.ones(len(features)) * 2)
        
        return dict(zip(features, importances))

class DashboardComponents:
    """Componentes do dashboard"""
    
    def __init__(self, data_provider: DashboardDataProvider):
        self.data_provider = data_provider
    
    def render_header(self):
        """Renderiza cabeçalho"""
        st.markdown('<h1 class="main-header">🏦 AML Feature Store Dashboard</h1>', unsafe_allow_html=True)
        
        # Status da API
        health = self.data_provider.get_api_health()
        status = health.get('status', 'unknown')
        
        if status == 'healthy':
            st.success("✅ Sistema operacional")
        else:
            st.error(f"❌ Sistema com problemas: {health.get('error', 'Erro desconhecido')}")
    
    def render_metrics_overview(self):
        """Renderiza visão geral das métricas"""
        st.subheader("📊 Visão Geral do Sistema")
        
        metrics = self.data_provider.get_system_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Predições Hoje",
                value=f"{metrics['total_predictions_today']:,}",
                delta=f"+{np.random.randint(100, 500)} vs ontem"
            )
        
        with col2:
            st.metric(
                label="Tempo Resposta Médio",
                value=f"{metrics['avg_response_time_ms']:.1f}ms",
                delta=f"{np.random.uniform(-2, 2):.1f}ms"
            )
        
        with col3:
            st.metric(
                label="Alertas Alto Risco",
                value=f"{metrics['high_risk_alerts']}",
                delta=f"{np.random.randint(-20, 30)}"
            )
        
        with col4:
            st.metric(
                label="Uptime",
                value=f"{metrics['system_uptime_hours']:.1f}h",
                delta="99.9% disponibilidade"
            )
    
    def render_real_time_alerts(self):
        """Renderiza alertas em tempo real"""
        st.subheader("🚨 Alertas em Tempo Real")
        
        predictions = self.data_provider.get_recent_predictions(20)
        high_risk_predictions = [p for p in predictions if p['risk_level'] == 'HIGH']
        
        if high_risk_predictions:
            for pred in high_risk_predictions[:5]:  # Top 5 alertas
                with st.container():
                    st.markdown(f"""
                    <div class="alert-high">
                        <strong>🚨 ALTO RISCO</strong><br>
                        Cliente: {pred['customer_id']}<br>
                        Valor: R$ {pred['amount']:,.2f}<br>
                        Score: {pred['risk_score']:.3f}<br>
                        Timestamp: {pred['timestamp'].strftime('%H:%M:%S')}
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.info("✅ Nenhum alerta de alto risco no momento")
    
    def render_risk_distribution(self):
        """Renderiza distribuição de risco"""
        st.subheader("📈 Distribuição de Risco - Últimas 2 Horas")
        
        predictions = self.data_provider.get_recent_predictions(200)
        df = pd.DataFrame(predictions)
        
        # Gráfico de distribuição por nível de risco
        risk_counts = df['risk_level'].value_counts()
        
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color_discrete_map={
                'LOW': '#4CAF50',
                'MEDIUM': '#FF9800', 
                'HIGH': '#F44336'
            },
            title="Distribuição por Nível de Risco"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Histograma de scores
        fig2 = px.histogram(
            df, x='risk_score', nbins=30,
            title="Distribuição de Risk Scores",
            labels={'risk_score': 'Risk Score', 'count': 'Frequência'}
        )
        fig2.update_layout(showlegend=False)
        
        st.plotly_chart(fig2, use_container_width=True)
    
    def render_time_series_analysis(self):
        """Renderiza análise de séries temporais"""
        st.subheader("⏰ Análise Temporal")
        
        predictions = self.data_provider.get_recent_predictions(500)
        df = pd.DataFrame(predictions)
        
        # Agrupar por intervalos de tempo
        df['time_bucket'] = pd.to_datetime(df['timestamp']).dt.floor('10min')
        
        time_agg = df.groupby('time_bucket').agg({
            'risk_score': ['mean', 'count'],
            'amount': 'sum'
        }).reset_index()
        
        time_agg.columns = ['timestamp', 'avg_risk_score', 'transaction_count', 'total_amount']
        
        # Gráfico de múltiplas séries
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Risk Score Médio', 'Número de Transações', 'Volume Total'),
            vertical_spacing=0.08
        )
        
        # Risk Score
        fig.add_trace(
            go.Scatter(
                x=time_agg['timestamp'],
                y=time_agg['avg_risk_score'],
                mode='lines+markers',
                name='Risk Score Médio',
                line=dict(color='#FF6B6B')
            ),
            row=1, col=1
        )
        
        # Contagem de transações
        fig.add_trace(
            go.Scatter(
                x=time_agg['timestamp'],
                y=time_agg['transaction_count'],
                mode='lines+markers',
                name='Transações',
                line=dict(color='#4ECDC4')
            ),
            row=2, col=1
        )
        
        # Volume total
        fig.add_trace(
            go.Scatter(
                x=time_agg['timestamp'],
                y=time_agg['total_amount'],
                mode='lines+markers',
                name='Volume (R$)',
                line=dict(color='#45B7D1')
            ),
            row=3, col=1
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_feature_importance(self):
        """Renderiza importância das features"""
        st.subheader("🎯 Importância das Features")
        
        feature_importance = self.data_provider.get_feature_importance()
        
        # Ordenar por importância
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        features, importances = zip(*sorted_features[:10])  # Top 10
        
        fig = px.bar(
            x=list(importances),
            y=list(features),
            orientation='h',
            title="Top 10 Features Mais Importantes",
            labels={'x': 'Importância', 'y': 'Feature'}
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_performance_metrics(self):
        """Renderiza métricas de performance"""
        st.subheader("⚡ Performance do Sistema")
        
        predictions = self.data_provider.get_recent_predictions(100)
        df = pd.DataFrame(predictions)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribuição de tempo de processamento
            fig = px.histogram(
                df, x='processing_time_ms',
                title="Distribuição do Tempo de Processamento",
                labels={'processing_time_ms': 'Tempo (ms)', 'count': 'Frequência'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Métricas de performance
            avg_time = df['processing_time_ms'].mean()
            p95_time = df['processing_time_ms'].quantile(0.95)
            p99_time = df['processing_time_ms'].quantile(0.99)
            
            st.markdown(f"""
            **Métricas de Latência:**
            - Tempo médio: {avg_time:.1f}ms
            - P95: {p95_time:.1f}ms  
            - P99: {p99_time:.1f}ms
            
            **Throughput:**
            - Atual: ~100 req/s
            - Pico: ~500 req/s
            - Capacidade: 1000+ req/s
            """)

def main():
    """Função principal do dashboard"""
    
    # Inicializar provedor de dados
    data_provider = DashboardDataProvider()
    components = DashboardComponents(data_provider)
    
    # Sidebar para controles
    with st.sidebar:
        st.header("⚙️ Controles")
        
        # Auto-refresh
        auto_refresh = st.checkbox("🔄 Auto-refresh (30s)", value=False)
        
        if auto_refresh:
            time.sleep(30)
            st.experimental_rerun()
        
        # Botão de refresh manual
        if st.button("🔄 Atualizar Dados"):
            st.experimental_rerun()
        
        st.markdown("---")
        
        # Filtros
        st.subheader("🔍 Filtros")
        
        time_range = st.selectbox(
            "Período de análise:",
            ["Última hora", "Últimas 2 horas", "Últimas 6 horas", "Último dia"]
        )
        
        risk_filter = st.multiselect(
            "Níveis de risco:",
            ["LOW", "MEDIUM", "HIGH"],
            default=["LOW", "MEDIUM", "HIGH"]
        )
        
        st.markdown("---")
        
        # Informações do sistema
        st.subheader("ℹ️ Sistema")
        st.info(f"""
        **Versão:** 2.0.0
        **Última atualização:** {datetime.now().strftime('%H:%M:%S')}
        **Modelos ativos:** 5
        **Features:** 90+
        """)
    
    # Renderizar componentes principais
    components.render_header()
    
    # Layout em abas
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🚨 Alertas", "📈 Análises", "⚡ Performance"])
    
    with tab1:
        components.render_metrics_overview()
        st.markdown("---")
        components.render_risk_distribution()
    
    with tab2:
        components.render_real_time_alerts()
    
    with tab3:
        components.render_time_series_analysis()
        st.markdown("---")
        components.render_feature_importance()
    
    with tab4:
        components.render_performance_metrics()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        🏦 AML Feature Store Dashboard v2.0 | 
        Desenvolvido com Streamlit e Plotly | 
        © 2024 Advanced AML Systems
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
