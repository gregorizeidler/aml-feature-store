"""
Advanced Feature Engineering para AML
Inclui features baseadas em grafos, séries temporais e análise comportamental
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import networkx as nx
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

class GraphFeatureExtractor:
    """
    Extrator de features baseadas em análise de grafos
    """
    
    def __init__(self):
        self.customer_graph = nx.Graph()
        self.merchant_graph = nx.Graph()
        self.ip_graph = nx.Graph()
    
    def build_graphs(self, df: pd.DataFrame):
        """Constrói grafos de relacionamentos"""
        
        # Grafo Customer-Merchant
        for _, row in df.iterrows():
            customer = row['customer_id']
            merchant = row['merchant_id']
            amount = row['amount']
            
            if self.customer_graph.has_edge(customer, merchant):
                self.customer_graph[customer][merchant]['weight'] += amount
                self.customer_graph[customer][merchant]['count'] += 1
            else:
                self.customer_graph.add_edge(customer, merchant, weight=amount, count=1)
        
        # Grafo Customer-IP
        for _, row in df.iterrows():
            customer = row['customer_id']
            ip = row['ip_address']
            
            if self.ip_graph.has_edge(customer, ip):
                self.ip_graph[customer][ip]['count'] += 1
            else:
                self.ip_graph.add_edge(customer, ip, count=1)
    
    def extract_customer_graph_features(self, customer_id: str) -> Dict[str, float]:
        """Extrai features de grafo para um cliente"""
        features = {}
        
        try:
            # Degree centrality
            if customer_id in self.customer_graph:
                features['graph_degree_centrality'] = nx.degree_centrality(self.customer_graph)[customer_id]
                features['graph_betweenness_centrality'] = nx.betweenness_centrality(self.customer_graph)[customer_id]
                features['graph_closeness_centrality'] = nx.closeness_centrality(self.customer_graph)[customer_id]
                
                # Número de conexões diretas
                features['graph_direct_connections'] = len(list(self.customer_graph.neighbors(customer_id)))
                
                # Peso total das conexões
                total_weight = sum([self.customer_graph[customer_id][neighbor]['weight'] 
                                  for neighbor in self.customer_graph.neighbors(customer_id)])
                features['graph_total_weight'] = total_weight
                
                # Clustering coefficient
                features['graph_clustering_coefficient'] = nx.clustering(self.customer_graph, customer_id)
                
            else:
                # Cliente novo - features zeradas
                features.update({
                    'graph_degree_centrality': 0.0,
                    'graph_betweenness_centrality': 0.0,
                    'graph_closeness_centrality': 0.0,
                    'graph_direct_connections': 0.0,
                    'graph_total_weight': 0.0,
                    'graph_clustering_coefficient': 0.0
                })
            
            # Features de IP
            if customer_id in self.ip_graph:
                ip_connections = len(list(self.ip_graph.neighbors(customer_id)))
                features['graph_ip_diversity'] = ip_connections
            else:
                features['graph_ip_diversity'] = 0.0
                
        except Exception as e:
            print(f"Erro ao calcular features de grafo para {customer_id}: {e}")
            # Retornar features zeradas em caso de erro
            features.update({
                'graph_degree_centrality': 0.0,
                'graph_betweenness_centrality': 0.0,
                'graph_closeness_centrality': 0.0,
                'graph_direct_connections': 0.0,
                'graph_total_weight': 0.0,
                'graph_clustering_coefficient': 0.0,
                'graph_ip_diversity': 0.0
            })
        
        return features

class TimeSeriesFeatureExtractor:
    """
    Extrator de features de séries temporais
    """
    
    def __init__(self):
        pass
    
    def extract_temporal_features(self, customer_transactions: pd.DataFrame) -> Dict[str, float]:
        """Extrai features temporais avançadas"""
        features = {}
        
        if len(customer_transactions) == 0:
            return self._get_zero_features()
        
        # Ordenar por timestamp
        df = customer_transactions.sort_values('event_timestamp')
        
        # Features de sazonalidade
        df['hour'] = pd.to_datetime(df['event_timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['event_timestamp']).dt.dayofweek
        df['day_of_month'] = pd.to_datetime(df['event_timestamp']).dt.day
        
        # Distribuição por hora
        hour_counts = df['hour'].value_counts()
        features['temporal_hour_entropy'] = self._calculate_entropy(hour_counts.values)
        features['temporal_peak_hour_ratio'] = hour_counts.max() / len(df) if len(df) > 0 else 0
        
        # Distribuição por dia da semana
        dow_counts = df['day_of_week'].value_counts()
        features['temporal_dow_entropy'] = self._calculate_entropy(dow_counts.values)
        features['temporal_weekend_ratio'] = len(df[df['day_of_week'] >= 5]) / len(df) if len(df) > 0 else 0
        
        # Features de intervalos entre transações
        if len(df) > 1:
            time_diffs = pd.to_datetime(df['event_timestamp']).diff().dt.total_seconds().dropna()
            
            features['temporal_avg_interval'] = time_diffs.mean()
            features['temporal_std_interval'] = time_diffs.std()
            features['temporal_min_interval'] = time_diffs.min()
            features['temporal_max_interval'] = time_diffs.max()
            
            # Regularidade (coeficiente de variação)
            features['temporal_regularity'] = time_diffs.std() / time_diffs.mean() if time_diffs.mean() > 0 else 0
            
            # Detecção de burst (intervalos muito pequenos)
            burst_threshold = 300  # 5 minutos
            burst_count = (time_diffs < burst_threshold).sum()
            features['temporal_burst_ratio'] = burst_count / len(time_diffs)
        else:
            features.update({
                'temporal_avg_interval': 0.0,
                'temporal_std_interval': 0.0,
                'temporal_min_interval': 0.0,
                'temporal_max_interval': 0.0,
                'temporal_regularity': 0.0,
                'temporal_burst_ratio': 0.0
            })
        
        # Features de tendência
        if len(df) >= 3:
            # Tendência de valores
            amounts = df['amount'].values
            time_index = np.arange(len(amounts))
            slope, _, r_value, _, _ = stats.linregress(time_index, amounts)
            
            features['temporal_amount_trend'] = slope
            features['temporal_amount_trend_strength'] = abs(r_value)
            
            # Tendência de frequência (transações por dia)
            df['date'] = pd.to_datetime(df['event_timestamp']).dt.date
            daily_counts = df.groupby('date').size()
            if len(daily_counts) >= 3:
                time_index = np.arange(len(daily_counts))
                slope, _, r_value, _, _ = stats.linregress(time_index, daily_counts.values)
                features['temporal_frequency_trend'] = slope
                features['temporal_frequency_trend_strength'] = abs(r_value)
            else:
                features['temporal_frequency_trend'] = 0.0
                features['temporal_frequency_trend_strength'] = 0.0
        else:
            features.update({
                'temporal_amount_trend': 0.0,
                'temporal_amount_trend_strength': 0.0,
                'temporal_frequency_trend': 0.0,
                'temporal_frequency_trend_strength': 0.0
            })
        
        return features
    
    def _calculate_entropy(self, values: np.ndarray) -> float:
        """Calcula entropia de Shannon"""
        if len(values) == 0:
            return 0.0
        
        probabilities = values / values.sum()
        probabilities = probabilities[probabilities > 0]  # Remove zeros
        
        if len(probabilities) <= 1:
            return 0.0
        
        return -np.sum(probabilities * np.log2(probabilities))
    
    def _get_zero_features(self) -> Dict[str, float]:
        """Retorna features zeradas"""
        return {
            'temporal_hour_entropy': 0.0,
            'temporal_peak_hour_ratio': 0.0,
            'temporal_dow_entropy': 0.0,
            'temporal_weekend_ratio': 0.0,
            'temporal_avg_interval': 0.0,
            'temporal_std_interval': 0.0,
            'temporal_min_interval': 0.0,
            'temporal_max_interval': 0.0,
            'temporal_regularity': 0.0,
            'temporal_burst_ratio': 0.0,
            'temporal_amount_trend': 0.0,
            'temporal_amount_trend_strength': 0.0,
            'temporal_frequency_trend': 0.0,
            'temporal_frequency_trend_strength': 0.0
        }

class BehavioralFeatureExtractor:
    """
    Extrator de features comportamentais avançadas
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def extract_behavioral_features(self, customer_transactions: pd.DataFrame, 
                                  all_transactions: pd.DataFrame) -> Dict[str, float]:
        """Extrai features comportamentais comparativas"""
        features = {}
        
        if len(customer_transactions) == 0:
            return self._get_zero_behavioral_features()
        
        # Features de desvio da norma
        customer_amounts = customer_transactions['amount'].values
        all_amounts = all_transactions['amount'].values
        
        # Z-score do valor médio do cliente vs população
        customer_avg = customer_amounts.mean()
        population_avg = all_amounts.mean()
        population_std = all_amounts.std()
        
        if population_std > 0:
            features['behavioral_amount_zscore'] = (customer_avg - population_avg) / population_std
        else:
            features['behavioral_amount_zscore'] = 0.0
        
        # Percentil do cliente na população
        features['behavioral_amount_percentile'] = stats.percentileofscore(all_amounts, customer_avg) / 100.0
        
        # Features de variabilidade
        features['behavioral_amount_cv'] = customer_amounts.std() / customer_avg if customer_avg > 0 else 0
        
        # Skewness e kurtosis
        if len(customer_amounts) >= 3:
            features['behavioral_amount_skewness'] = stats.skew(customer_amounts)
            features['behavioral_amount_kurtosis'] = stats.kurtosis(customer_amounts)
        else:
            features['behavioral_amount_skewness'] = 0.0
            features['behavioral_amount_kurtosis'] = 0.0
        
        # Features de clustering (detecção de anomalias)
        if len(customer_transactions) >= 5:
            # Preparar dados para clustering
            cluster_features = customer_transactions[['amount']].copy()
            cluster_features['hour'] = pd.to_datetime(customer_transactions['event_timestamp']).dt.hour
            
            # Normalizar
            cluster_features_scaled = self.scaler.fit_transform(cluster_features)
            
            # DBSCAN para detectar outliers
            try:
                dbscan = DBSCAN(eps=0.5, min_samples=2)
                clusters = dbscan.fit_predict(cluster_features_scaled)
                
                # Proporção de outliers (cluster -1)
                outlier_ratio = (clusters == -1).sum() / len(clusters)
                features['behavioral_outlier_ratio'] = outlier_ratio
                
                # Número de clusters
                n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
                features['behavioral_n_clusters'] = n_clusters
                
            except Exception:
                features['behavioral_outlier_ratio'] = 0.0
                features['behavioral_n_clusters'] = 1.0
        else:
            features['behavioral_outlier_ratio'] = 0.0
            features['behavioral_n_clusters'] = 1.0
        
        # Features de merchant diversity
        unique_merchants = customer_transactions['merchant_id'].nunique()
        total_transactions = len(customer_transactions)
        features['behavioral_merchant_diversity'] = unique_merchants / total_transactions if total_transactions > 0 else 0
        
        # Merchant concentration (HHI)
        merchant_counts = customer_transactions['merchant_id'].value_counts()
        merchant_shares = merchant_counts / merchant_counts.sum()
        hhi = (merchant_shares ** 2).sum()
        features['behavioral_merchant_hhi'] = hhi
        
        return features
    
    def _get_zero_behavioral_features(self) -> Dict[str, float]:
        """Retorna features comportamentais zeradas"""
        return {
            'behavioral_amount_zscore': 0.0,
            'behavioral_amount_percentile': 0.0,
            'behavioral_amount_cv': 0.0,
            'behavioral_amount_skewness': 0.0,
            'behavioral_amount_kurtosis': 0.0,
            'behavioral_outlier_ratio': 0.0,
            'behavioral_n_clusters': 0.0,
            'behavioral_merchant_diversity': 0.0,
            'behavioral_merchant_hhi': 0.0
        }

class AdvancedFeatureEngineering:
    """
    Classe principal para engenharia de features avançada
    """
    
    def __init__(self):
        self.graph_extractor = GraphFeatureExtractor()
        self.temporal_extractor = TimeSeriesFeatureExtractor()
        self.behavioral_extractor = BehavioralFeatureExtractor()
    
    def fit(self, df: pd.DataFrame):
        """Treina os extratores com os dados"""
        print("🔧 Construindo grafos de relacionamentos...")
        self.graph_extractor.build_graphs(df)
        print("✅ Grafos construídos com sucesso!")
    
    def extract_all_features(self, customer_id: str, customer_transactions: pd.DataFrame, 
                           all_transactions: pd.DataFrame) -> Dict[str, float]:
        """Extrai todas as features avançadas para um cliente"""
        
        features = {}
        
        # Features de grafo
        graph_features = self.graph_extractor.extract_customer_graph_features(customer_id)
        features.update(graph_features)
        
        # Features temporais
        temporal_features = self.temporal_extractor.extract_temporal_features(customer_transactions)
        features.update(temporal_features)
        
        # Features comportamentais
        behavioral_features = self.behavioral_extractor.extract_behavioral_features(
            customer_transactions, all_transactions
        )
        features.update(behavioral_features)
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Retorna lista de nomes das features"""
        return [
            # Graph features
            'graph_degree_centrality', 'graph_betweenness_centrality', 'graph_closeness_centrality',
            'graph_direct_connections', 'graph_total_weight', 'graph_clustering_coefficient',
            'graph_ip_diversity',
            
            # Temporal features
            'temporal_hour_entropy', 'temporal_peak_hour_ratio', 'temporal_dow_entropy',
            'temporal_weekend_ratio', 'temporal_avg_interval', 'temporal_std_interval',
            'temporal_min_interval', 'temporal_max_interval', 'temporal_regularity',
            'temporal_burst_ratio', 'temporal_amount_trend', 'temporal_amount_trend_strength',
            'temporal_frequency_trend', 'temporal_frequency_trend_strength',
            
            # Behavioral features
            'behavioral_amount_zscore', 'behavioral_amount_percentile', 'behavioral_amount_cv',
            'behavioral_amount_skewness', 'behavioral_amount_kurtosis', 'behavioral_outlier_ratio',
            'behavioral_n_clusters', 'behavioral_merchant_diversity', 'behavioral_merchant_hhi'
        ]

def create_advanced_dataset(df: pd.DataFrame, sample_size: int = 5000) -> pd.DataFrame:
    """
    Cria dataset com features avançadas
    """
    print("🚀 Iniciando criação de dataset com features avançadas...")
    
    # Inicializar feature engineering
    feature_eng = AdvancedFeatureEngineering()
    feature_eng.fit(df)
    
    # Amostra para processamento
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42).sort_values('event_timestamp')
    
    advanced_data = []
    
    for idx, (_, row) in enumerate(df_sample.iterrows()):
        if idx % 1000 == 0:
            print(f"   Processando transação {idx}/{len(df_sample)}...")
        
        customer_id = row['customer_id']
        timestamp = row['event_timestamp']
        
        # Transações do cliente até este momento
        customer_txns = df[
            (df['customer_id'] == customer_id) & 
            (df['event_timestamp'] <= timestamp)
        ]
        
        # Extrair features avançadas
        advanced_features = feature_eng.extract_all_features(
            customer_id, customer_txns, df
        )
        
        # Combinar com dados básicos
        record = {
            'transaction_id': row['transaction_id'],
            'customer_id': customer_id,
            'merchant_id': row['merchant_id'],
            'amount': row['amount'],
            'event_timestamp': timestamp,
            'is_suspicious': row['is_suspicious'],
            **advanced_features
        }
        
        advanced_data.append(record)
    
    result_df = pd.DataFrame(advanced_data)
    print(f"✅ Dataset avançado criado: {len(result_df)} amostras com {len(advanced_features)} features avançadas")
    
    return result_df
