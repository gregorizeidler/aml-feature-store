"""
API FastAPI para inferência em tempo real usando Feature Store
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import redis
from feast import FeatureStore

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modelos Pydantic
class TransactionRequest(BaseModel):
    """Modelo para requisição de transação"""
    customer_id: str = Field(..., description="ID único do cliente")
    merchant_id: str = Field(..., description="ID único do estabelecimento")
    amount: float = Field(..., gt=0, description="Valor da transação")
    ip_address: Optional[str] = Field(None, description="Endereço IP da transação")
    
class FeatureVector(BaseModel):
    """Modelo para vetor de features"""
    customer_id: str
    features: Dict[str, float]
    feature_timestamp: datetime
    
class RiskPrediction(BaseModel):
    """Modelo para resposta de predição de risco"""
    transaction_id: str
    customer_id: str
    merchant_id: str
    amount: float
    risk_score: float = Field(..., ge=0, le=1, description="Score de risco entre 0 e 1")
    risk_level: str = Field(..., description="Nível de risco: LOW, MEDIUM, HIGH")
    features_used: Dict[str, float]
    explanation: List[str]
    processing_time_ms: float

class HealthCheck(BaseModel):
    """Modelo para health check"""
    status: str
    timestamp: datetime
    services: Dict[str, str]

# Inicializar aplicação
app = FastAPI(
    title="AML Feature Store API",
    description="API para inferência em tempo real de risco de lavagem de dinheiro",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variáveis globais
feature_store = None
redis_client = None

class AMLRiskModel:
    """
    Modelo simples de risco AML baseado em regras
    Em um ambiente real, seria um modelo ML treinado (XGBoost, etc.)
    """
    
    def __init__(self):
        # Pesos das features (simulando um modelo treinado)
        self.feature_weights = {
            'txn_amount_sum_60s': 0.15,
            'txn_amount_sum_5m': 0.12,
            'txn_amount_sum_1h': 0.10,
            'txn_count_60s': 0.08,
            'txn_count_5m': 0.07,
            'txn_count_10m': 0.06,
            'txn_count_1h': 0.05,
            'unique_ips_1h': 0.20,
            'unique_merchants_1h': 0.05,
            'velocity_score_1h': 0.12,
            'amount_deviation_score_1h': 0.08,
            'night_txn_count_24h': 0.06,
            'weekend_txn_count_7d': 0.03,
            'avg_txn_amount_1h': 0.03
        }
        
        # Thresholds para classificação
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 1.0
        }
    
    def predict(self, features: Dict[str, float], transaction_amount: float) -> Dict:
        """
        Prediz o risco de uma transação baseado nas features
        
        Args:
            features: Dicionário com features calculadas
            transaction_amount: Valor da transação atual
            
        Returns:
            Dict com score, nível e explicação
        """
        
        # Normalizar features
        normalized_features = self._normalize_features(features, transaction_amount)
        
        # Calcular score baseado nos pesos
        risk_score = 0.0
        feature_contributions = {}
        
        for feature_name, weight in self.feature_weights.items():
            if feature_name in normalized_features:
                contribution = normalized_features[feature_name] * weight
                risk_score += contribution
                feature_contributions[feature_name] = contribution
        
        # Aplicar regras de negócio adicionais
        risk_score = self._apply_business_rules(risk_score, features, transaction_amount)
        
        # Classificar nível de risco
        risk_level = self._classify_risk_level(risk_score)
        
        # Gerar explicação
        explanation = self._generate_explanation(features, feature_contributions, transaction_amount)
        
        return {
            'risk_score': min(max(risk_score, 0.0), 1.0),  # Garantir entre 0 e 1
            'risk_level': risk_level,
            'explanation': explanation,
            'feature_contributions': feature_contributions
        }
    
    def _normalize_features(self, features: Dict[str, float], transaction_amount: float) -> Dict[str, float]:
        """Normaliza features para escala 0-1"""
        normalized = {}
        
        # Normalização baseada em thresholds esperados
        normalizers = {
            'txn_amount_sum_60s': 10000.0,    # Valores acima de 10k são suspeitos
            'txn_amount_sum_5m': 25000.0,
            'txn_amount_sum_1h': 50000.0,
            'txn_count_60s': 10.0,            # Mais de 10 transações por minuto é suspeito
            'txn_count_5m': 20.0,
            'txn_count_10m': 30.0,
            'txn_count_1h': 50.0,
            'unique_ips_1h': 5.0,             # Mais de 5 IPs diferentes é suspeito
            'unique_merchants_1h': 10.0,
            'velocity_score_1h': 2.0,         # Mais de 2 transações por minuto
            'amount_deviation_score_1h': 5000.0,  # Desvio padrão alto
            'night_txn_count_24h': 5.0,
            'weekend_txn_count_7d': 10.0,
            'avg_txn_amount_1h': 5000.0
        }
        
        for feature_name, value in features.items():
            if feature_name in normalizers:
                normalized[feature_name] = min(value / normalizers[feature_name], 1.0)
        
        return normalized
    
    def _apply_business_rules(self, base_score: float, features: Dict[str, float], transaction_amount: float) -> float:
        """Aplica regras de negócio específicas"""
        
        # Regra 1: Transações muito altas são automaticamente suspeitas
        if transaction_amount > 10000:
            base_score += 0.3
        elif transaction_amount > 5000:
            base_score += 0.15
        
        # Regra 2: Múltiplos IPs em pouco tempo
        if features.get('unique_ips_1h', 0) > 3:
            base_score += 0.25
        
        # Regra 3: Velocidade muito alta
        if features.get('velocity_score_1h', 0) > 1.5:  # Mais de 1.5 transações por minuto
            base_score += 0.2
        
        # Regra 4: Padrão de valores muito irregular
        if features.get('amount_deviation_score_1h', 0) > 3000:
            base_score += 0.15
        
        # Regra 5: Transações noturnas frequentes
        if features.get('night_txn_count_24h', 0) > 3:
            base_score += 0.1
        
        return base_score
    
    def _classify_risk_level(self, risk_score: float) -> str:
        """Classifica o nível de risco"""
        if risk_score <= self.risk_thresholds['low']:
            return 'LOW'
        elif risk_score <= self.risk_thresholds['medium']:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def _generate_explanation(self, features: Dict[str, float], contributions: Dict[str, float], transaction_amount: float) -> List[str]:
        """Gera explicação legível para a predição"""
        explanations = []
        
        # Ordenar contribuições por impacto
        sorted_contributions = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Top 3 fatores mais importantes
        for feature_name, contribution in sorted_contributions[:3]:
            if contribution > 0.05:  # Apenas contribuições significativas
                feature_value = features.get(feature_name, 0)
                
                if 'amount_sum' in feature_name:
                    explanations.append(f"Alto volume transacionado: R$ {feature_value:,.2f}")
                elif 'count' in feature_name:
                    explanations.append(f"Muitas transações: {int(feature_value)} transações")
                elif 'unique_ips' in feature_name:
                    explanations.append(f"Múltiplos IPs utilizados: {int(feature_value)} IPs diferentes")
                elif 'velocity_score' in feature_name:
                    explanations.append(f"Alta velocidade de transações: {feature_value:.2f} txn/min")
                elif 'deviation_score' in feature_name:
                    explanations.append(f"Padrão irregular de valores: desvio de R$ {feature_value:,.2f}")
        
        # Adicionar explicações específicas da transação atual
        if transaction_amount > 10000:
            explanations.append(f"Transação de alto valor: R$ {transaction_amount:,.2f}")
        
        if not explanations:
            explanations.append("Padrão de transação normal")
        
        return explanations

# Instanciar modelo
risk_model = AMLRiskModel()

@app.on_event("startup")
async def startup_event():
    """Inicialização da aplicação"""
    global feature_store, redis_client
    
    try:
        # Conectar ao Redis
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        redis_client.ping()  # Testar conexão
        logger.info("Conectado ao Redis")
        
        # Inicializar Feast Feature Store
        feature_store = FeatureStore(repo_path="../feature_repo")
        logger.info("Feature Store inicializado")
        
    except Exception as e:
        logger.error(f"Erro na inicialização: {e}")
        raise

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Endpoint de health check"""
    services = {}
    
    # Verificar Redis
    try:
        redis_client.ping()
        services["redis"] = "healthy"
    except:
        services["redis"] = "unhealthy"
    
    # Verificar Feature Store
    try:
        # Tentar listar feature views
        feature_store.list_feature_views()
        services["feast"] = "healthy"
    except:
        services["feast"] = "unhealthy"
    
    overall_status = "healthy" if all(status == "healthy" for status in services.values()) else "unhealthy"
    
    return HealthCheck(
        status=overall_status,
        timestamp=datetime.now(),
        services=services
    )

@app.post("/predict", response_model=RiskPrediction)
async def predict_transaction_risk(transaction: TransactionRequest):
    """
    Prediz o risco de uma transação em tempo real
    """
    start_time = datetime.now()
    transaction_id = f"txn_{int(start_time.timestamp() * 1000)}"
    
    try:
        # Buscar features do Feature Store
        features = await get_customer_features(transaction.customer_id)
        
        # Fazer predição
        prediction = risk_model.predict(features, transaction.amount)
        
        # Calcular tempo de processamento
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return RiskPrediction(
            transaction_id=transaction_id,
            customer_id=transaction.customer_id,
            merchant_id=transaction.merchant_id,
            amount=transaction.amount,
            risk_score=prediction['risk_score'],
            risk_level=prediction['risk_level'],
            features_used=features,
            explanation=prediction['explanation'],
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

async def get_customer_features(customer_id: str) -> Dict[str, float]:
    """
    Busca features do cliente do Feature Store (Redis)
    """
    try:
        # Tentar buscar do Redis primeiro (mais rápido)
        redis_key = f"aml_feature_store:customer_transaction_features:{customer_id}"
        redis_features = redis_client.hgetall(redis_key)
        
        if redis_features:
            # Converter strings para float
            features = {k: float(v) for k, v in redis_features.items()}
            logger.debug(f"Features encontradas no Redis para {customer_id}")
            return features
        
        # Se não encontrar no Redis, usar valores padrão
        logger.warning(f"Features não encontradas para cliente {customer_id}, usando valores padrão")
        
        default_features = {
            'txn_amount_sum_60s': 0.0,
            'txn_amount_sum_5m': 0.0,
            'txn_amount_sum_1h': 0.0,
            'txn_count_60s': 0,
            'txn_count_5m': 0,
            'txn_count_10m': 0,
            'txn_count_1h': 0,
            'unique_ips_1h': 0,
            'unique_merchants_1h': 0,
            'velocity_score_1h': 0.0,
            'amount_deviation_score_1h': 0.0,
            'night_txn_count_24h': 0,
            'weekend_txn_count_7d': 0,
            'avg_txn_amount_1h': 0.0,
            'max_txn_amount_1h': 0.0
        }
        
        return default_features
        
    except Exception as e:
        logger.error(f"Erro ao buscar features: {e}")
        raise

@app.get("/features/{customer_id}", response_model=FeatureVector)
async def get_features(customer_id: str):
    """
    Endpoint para consultar features de um cliente
    """
    try:
        features = await get_customer_features(customer_id)
        
        return FeatureVector(
            customer_id=customer_id,
            features=features,
            feature_timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Erro ao buscar features: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.post("/batch-predict")
async def batch_predict(transactions: List[TransactionRequest]):
    """
    Predição em lote para múltiplas transações
    """
    results = []
    
    for transaction in transactions:
        try:
            # Reutilizar a lógica do endpoint individual
            prediction = await predict_transaction_risk(transaction)
            results.append(prediction)
        except Exception as e:
            logger.error(f"Erro na predição em lote para {transaction.customer_id}: {e}")
            # Continuar com as outras transações
            continue
    
    return {"predictions": results, "total_processed": len(results)}

@app.get("/")
async def root():
    """Endpoint raiz"""
    return {
        "message": "AML Feature Store API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
