"""
Sistema de Feedback Loop e Aprendizado Contínuo
Coleta feedback de analistas e retreina modelos automaticamente
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import sqlite3
import json
import logging
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeedbackDatabase:
    """
    Database para armazenar feedback dos analistas
    """
    
    def __init__(self, db_path: str = "feedback.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Inicializa o banco de dados"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabela de feedback
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                predicted_score REAL NOT NULL,
                predicted_label INTEGER NOT NULL,
                actual_label INTEGER,
                analyst_id TEXT,
                feedback_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                confidence_level INTEGER,
                comments TEXT,
                investigation_outcome TEXT
            )
        """)
        
        # Tabela de modelo performance
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT NOT NULL,
                training_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                validation_auc REAL,
                test_auc REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                feature_count INTEGER,
                training_samples INTEGER,
                is_active BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Tabela de drift detection
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drift_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                feature_name TEXT NOT NULL,
                drift_score REAL,
                drift_detected BOOLEAN,
                reference_period_start DATETIME,
                reference_period_end DATETIME,
                current_period_start DATETIME,
                current_period_end DATETIME
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("Database inicializado")
    
    def add_feedback(self, transaction_id: str, customer_id: str, 
                    predicted_score: float, predicted_label: int,
                    actual_label: int, analyst_id: str,
                    confidence_level: int = 5, comments: str = "",
                    investigation_outcome: str = ""):
        """Adiciona feedback de um analista"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO feedback 
            (transaction_id, customer_id, predicted_score, predicted_label, 
             actual_label, analyst_id, confidence_level, comments, investigation_outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (transaction_id, customer_id, predicted_score, predicted_label,
              actual_label, analyst_id, confidence_level, comments, investigation_outcome))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Feedback adicionado para transação {transaction_id}")
    
    def get_feedback_data(self, days_back: int = 30) -> pd.DataFrame:
        """Obtém dados de feedback dos últimos N dias"""
        
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT * FROM feedback 
            WHERE feedback_timestamp >= datetime('now', '-{} days')
            AND actual_label IS NOT NULL
        """.format(days_back)
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def get_model_performance_history(self) -> pd.DataFrame:
        """Obtém histórico de performance dos modelos"""
        
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM model_performance ORDER BY training_timestamp DESC", conn)
        conn.close()
        
        return df
    
    def save_model_performance(self, model_version: str, metrics: Dict[str, float],
                              feature_count: int, training_samples: int):
        """Salva métricas de performance de um modelo"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO model_performance 
            (model_version, validation_auc, test_auc, precision, recall, f1_score, 
             feature_count, training_samples)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (model_version, metrics.get('validation_auc'), metrics.get('test_auc'),
              metrics.get('precision'), metrics.get('recall'), metrics.get('f1_score'),
              feature_count, training_samples))
        
        conn.commit()
        conn.close()

class DriftDetector:
    """
    Detector de drift em features e performance
    """
    
    def __init__(self, feedback_db: FeedbackDatabase):
        self.feedback_db = feedback_db
    
    def detect_performance_drift(self, window_days: int = 7) -> Dict[str, Any]:
        """Detecta drift na performance do modelo"""
        
        feedback_df = self.feedback_db.get_feedback_data(days_back=window_days*2)
        
        if len(feedback_df) < 50:  # Mínimo de amostras
            return {"drift_detected": False, "reason": "Insufficient data"}
        
        # Dividir em duas janelas temporais
        feedback_df['feedback_timestamp'] = pd.to_datetime(feedback_df['feedback_timestamp'])
        cutoff_date = datetime.now() - timedelta(days=window_days)
        
        recent_data = feedback_df[feedback_df['feedback_timestamp'] >= cutoff_date]
        older_data = feedback_df[feedback_df['feedback_timestamp'] < cutoff_date]
        
        if len(recent_data) < 20 or len(older_data) < 20:
            return {"drift_detected": False, "reason": "Insufficient data in time windows"}
        
        # Calcular AUC para cada período
        recent_auc = roc_auc_score(recent_data['actual_label'], recent_data['predicted_score'])
        older_auc = roc_auc_score(older_data['actual_label'], older_data['predicted_score'])
        
        # Detectar drift significativo (>5% de degradação)
        auc_degradation = older_auc - recent_auc
        drift_threshold = 0.05
        
        drift_detected = auc_degradation > drift_threshold
        
        return {
            "drift_detected": drift_detected,
            "recent_auc": recent_auc,
            "older_auc": older_auc,
            "auc_degradation": auc_degradation,
            "threshold": drift_threshold,
            "recent_samples": len(recent_data),
            "older_samples": len(older_data)
        }
    
    def detect_feature_drift(self, feature_data: pd.DataFrame, 
                           reference_data: pd.DataFrame) -> Dict[str, Any]:
        """Detecta drift nas features usando KS test"""
        
        from scipy import stats
        
        drift_results = {}
        
        for column in feature_data.columns:
            if column in reference_data.columns:
                # Kolmogorov-Smirnov test
                ks_stat, p_value = stats.ks_2samp(
                    reference_data[column].dropna(),
                    feature_data[column].dropna()
                )
                
                # Drift detectado se p-value < 0.05
                drift_detected = p_value < 0.05
                
                drift_results[column] = {
                    "ks_statistic": ks_stat,
                    "p_value": p_value,
                    "drift_detected": drift_detected
                }
        
        return drift_results

class ModelRetrainer:
    """
    Sistema de retreinamento automático de modelos
    """
    
    def __init__(self, feedback_db: FeedbackDatabase, model_path: str = "models/"):
        self.feedback_db = feedback_db
        self.model_path = Path(model_path)
        self.model_path.mkdir(exist_ok=True)
    
    def should_retrain(self) -> Tuple[bool, str]:
        """Determina se o modelo deve ser retreinado"""
        
        # Verificar quantidade de feedback novo
        feedback_df = self.feedback_db.get_feedback_data(days_back=7)
        
        if len(feedback_df) < 100:
            return False, "Insufficient feedback data"
        
        # Verificar drift de performance
        drift_detector = DriftDetector(self.feedback_db)
        drift_result = drift_detector.detect_performance_drift()
        
        if drift_result["drift_detected"]:
            return True, f"Performance drift detected: AUC degradation of {drift_result['auc_degradation']:.3f}"
        
        # Verificar se há feedback suficiente para melhorar o modelo
        feedback_accuracy = (feedback_df['predicted_label'] == feedback_df['actual_label']).mean()
        
        if feedback_accuracy < 0.85:  # Accuracy baixa
            return True, f"Low accuracy detected: {feedback_accuracy:.3f}"
        
        # Verificar última vez que foi retreinado
        performance_history = self.feedback_db.get_model_performance_history()
        
        if len(performance_history) > 0:
            last_training = pd.to_datetime(performance_history.iloc[0]['training_timestamp'])
            days_since_training = (datetime.now() - last_training).days
            
            if days_since_training > 30:  # Retreinar a cada 30 dias
                return True, f"Scheduled retraining: {days_since_training} days since last training"
        
        return False, "No retraining needed"
    
    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepara dados de treinamento com feedback"""
        
        # Obter feedback dos últimos 90 dias
        feedback_df = self.feedback_db.get_feedback_data(days_back=90)
        
        if len(feedback_df) < 100:
            raise ValueError("Insufficient feedback data for retraining")
        
        # Simular features (em produção viria do feature store)
        # Para demonstração, vamos gerar features sintéticas baseadas nos dados
        np.random.seed(42)
        n_samples = len(feedback_df)
        n_features = 50
        
        # Gerar features correlacionadas com o target
        X = np.random.randn(n_samples, n_features)
        
        # Adicionar correlação com target
        for i in range(n_features):
            correlation_strength = np.random.uniform(0.1, 0.5)
            if np.random.random() > 0.5:  # 50% das features correlacionadas
                X[:, i] += feedback_df['actual_label'].values * correlation_strength
        
        # Criar DataFrame de features
        feature_names = [f"feature_{i}" for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        
        # Target
        y_df = feedback_df[['actual_label']].copy()
        
        return X_df, y_df
    
    def retrain_model(self) -> Dict[str, Any]:
        """Retreina o modelo com novos dados"""
        
        logger.info("Iniciando retreinamento do modelo...")
        
        # Preparar dados
        X, y = self.prepare_training_data()
        
        # Split dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y['actual_label'], test_size=0.3, random_state=42, stratify=y['actual_label']
        )
        
        # Treinar modelo (usando XGBoost como exemplo)
        import xgboost as xgb
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        )
        
        # Treinar
        model.fit(X_train, y_train)
        
        # Avaliar
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        # Métricas
        test_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Validação cruzada para AUC de validação
        from sklearn.model_selection import cross_val_score
        val_auc_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        val_auc = val_auc_scores.mean()
        
        # Report detalhado
        report = classification_report(y_test, y_pred, output_dict=True)
        
        metrics = {
            'validation_auc': val_auc,
            'test_auc': test_auc,
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score']
        }
        
        # Salvar modelo
        model_version = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_file = self.model_path / f"{model_version}.pkl"
        
        joblib.dump({
            'model': model,
            'feature_names': X.columns.tolist(),
            'metrics': metrics,
            'training_timestamp': datetime.now()
        }, model_file)
        
        # Salvar métricas no database
        self.feedback_db.save_model_performance(
            model_version, metrics, len(X.columns), len(X_train)
        )
        
        logger.info(f"Modelo retreinado com sucesso: {model_version}")
        logger.info(f"Test AUC: {test_auc:.4f}, Validation AUC: {val_auc:.4f}")
        
        return {
            'model_version': model_version,
            'model_file': str(model_file),
            'metrics': metrics,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }

class FeedbackCollector:
    """
    Coletor de feedback dos analistas
    """
    
    def __init__(self, feedback_db: FeedbackDatabase):
        self.feedback_db = feedback_db
    
    def collect_feedback_batch(self, feedback_data: List[Dict]) -> int:
        """Coleta feedback em lote"""
        
        count = 0
        for feedback in feedback_data:
            try:
                self.feedback_db.add_feedback(**feedback)
                count += 1
            except Exception as e:
                logger.error(f"Erro ao adicionar feedback: {e}")
        
        logger.info(f"Coletados {count} feedbacks")
        return count
    
    def simulate_analyst_feedback(self, n_samples: int = 100) -> List[Dict]:
        """Simula feedback de analistas (para demonstração)"""
        
        feedback_data = []
        
        for i in range(n_samples):
            # Simular dados de transação
            transaction_id = f"txn_{i:06d}"
            customer_id = f"CUST_{np.random.randint(1, 1000):06d}"
            
            # Score predito (beta distribution para realismo)
            predicted_score = np.random.beta(2, 5)
            predicted_label = 1 if predicted_score > 0.5 else 0
            
            # Label real (com algum ruído)
            # Modelos bons têm ~85% de accuracy
            if np.random.random() < 0.85:
                actual_label = predicted_label
            else:
                actual_label = 1 - predicted_label
            
            # Simular analista
            analyst_id = f"analyst_{np.random.randint(1, 10)}"
            confidence = np.random.randint(3, 6)  # 3-5 escala de confiança
            
            # Comentários simulados
            comments = [
                "Transação suspeita confirmada",
                "Falso positivo - cliente legítimo", 
                "Padrão típico de lavagem",
                "Necessita investigação adicional",
                "Cliente conhecido - baixo risco"
            ]
            
            feedback_data.append({
                'transaction_id': transaction_id,
                'customer_id': customer_id,
                'predicted_score': predicted_score,
                'predicted_label': predicted_label,
                'actual_label': actual_label,
                'analyst_id': analyst_id,
                'confidence_level': confidence,
                'comments': np.random.choice(comments),
                'investigation_outcome': 'completed'
            })
        
        return feedback_data

class ContinuousLearningPipeline:
    """
    Pipeline completo de aprendizado contínuo
    """
    
    def __init__(self, db_path: str = "feedback.db"):
        self.feedback_db = FeedbackDatabase(db_path)
        self.drift_detector = DriftDetector(self.feedback_db)
        self.model_retrainer = ModelRetrainer(self.feedback_db)
        self.feedback_collector = FeedbackCollector(self.feedback_db)
    
    def run_continuous_learning_cycle(self) -> Dict[str, Any]:
        """Executa um ciclo completo de aprendizado contínuo"""
        
        results = {
            'timestamp': datetime.now(),
            'feedback_collected': 0,
            'drift_detected': False,
            'model_retrained': False,
            'performance_metrics': {}
        }
        
        logger.info("Iniciando ciclo de aprendizado contínuo...")
        
        # 1. Coletar feedback (simulado para demonstração)
        simulated_feedback = self.feedback_collector.simulate_analyst_feedback(50)
        feedback_count = self.feedback_collector.collect_feedback_batch(simulated_feedback)
        results['feedback_collected'] = feedback_count
        
        # 2. Detectar drift
        drift_result = self.drift_detector.detect_performance_drift()
        results['drift_detected'] = drift_result.get('drift_detected', False)
        results['drift_metrics'] = drift_result
        
        # 3. Verificar se deve retreinar
        should_retrain, reason = self.model_retrainer.should_retrain()
        results['retrain_decision'] = {'should_retrain': should_retrain, 'reason': reason}
        
        # 4. Retreinar se necessário
        if should_retrain:
            try:
                retrain_result = self.model_retrainer.retrain_model()
                results['model_retrained'] = True
                results['retrain_results'] = retrain_result
                results['performance_metrics'] = retrain_result['metrics']
                
                logger.info("Modelo retreinado com sucesso!")
                
            except Exception as e:
                logger.error(f"Erro no retreinamento: {e}")
                results['retrain_error'] = str(e)
        
        # 5. Gerar relatório
        self._generate_cycle_report(results)
        
        return results
    
    def _generate_cycle_report(self, results: Dict[str, Any]):
        """Gera relatório do ciclo de aprendizado"""
        
        report = f"""
        
        📊 RELATÓRIO DO CICLO DE APRENDIZADO CONTÍNUO
        ============================================
        
        🕐 Timestamp: {results['timestamp']}
        
        📝 Feedback Coletado: {results['feedback_collected']} amostras
        
        🔍 Drift Detection:
           - Drift detectado: {results['drift_detected']}
           - Métricas: {results.get('drift_metrics', {})}
        
        🤖 Retreinamento:
           - Deve retreinar: {results['retrain_decision']['should_retrain']}
           - Razão: {results['retrain_decision']['reason']}
           - Modelo retreinado: {results['model_retrained']}
        
        """
        
        if results['model_retrained']:
            metrics = results.get('performance_metrics', {})
            report += f"""
        📈 Performance do Novo Modelo:
           - Test AUC: {metrics.get('test_auc', 0):.4f}
           - Validation AUC: {metrics.get('validation_auc', 0):.4f}
           - Precision: {metrics.get('precision', 0):.4f}
           - Recall: {metrics.get('recall', 0):.4f}
           - F1-Score: {metrics.get('f1_score', 0):.4f}
            """
        
        logger.info(report)

def main():
    """Função principal para demonstração"""
    
    print("🔄 Demonstração do Sistema de Aprendizado Contínuo")
    print("=" * 60)
    
    # Inicializar pipeline
    pipeline = ContinuousLearningPipeline()
    
    # Executar ciclo
    results = pipeline.run_continuous_learning_cycle()
    
    print("\n✅ Ciclo de aprendizado contínuo concluído!")
    print(f"📊 Resumo: {results['feedback_collected']} feedbacks coletados")
    
    if results['model_retrained']:
        print("🤖 Modelo foi retreinado com sucesso!")
        metrics = results.get('performance_metrics', {})
        print(f"   AUC: {metrics.get('test_auc', 0):.4f}")
    else:
        print("ℹ️  Retreinamento não foi necessário")

if __name__ == "__main__":
    main()
