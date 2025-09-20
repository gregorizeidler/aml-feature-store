"""
Modelos ML Avançados para AML
Inclui Deep Learning, Ensemble Methods e AutoML
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    VotingClassifier, BaggingClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    RandomizedSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline

# XGBoost e LightGBM
import xgboost as xgb
import lightgbm as lgb
try:
    import catboost as cb
except ImportError:
    cb = None

# Imbalanced learning
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

# Deep Learning (TensorFlow/Keras)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow não disponível. Modelos de Deep Learning serão desabilitados.")

# Optuna para hyperparameter tuning
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna não disponível. AutoML será limitado.")

class DeepLearningModels:
    """
    Modelos de Deep Learning para detecção de fraudes
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    def create_feedforward_nn(self, input_dim: int, name: str = "feedforward") -> Optional[Any]:
        """Cria rede neural feedforward"""
        if not TENSORFLOW_AVAILABLE:
            return None
            
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(), 
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.models[name] = model
        return model
    
    def create_autoencoder_anomaly_detector(self, input_dim: int, name: str = "autoencoder") -> Optional[Any]:
        """Cria autoencoder para detecção de anomalias"""
        if not TENSORFLOW_AVAILABLE:
            return None
            
        # Encoder
        encoder_input = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(128, activation='relu')(encoder_input)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(64, activation='relu')(encoded)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dense(32, activation='relu')(encoded)  # Bottleneck
        
        # Decoder
        decoded = layers.Dense(64, activation='relu')(encoded)
        decoded = layers.BatchNormalization()(decoded)
        decoded = layers.Dense(128, activation='relu')(decoded)
        decoded = layers.BatchNormalization()(decoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)
        
        # Autoencoder completo
        autoencoder = keras.Model(encoder_input, decoded)
        autoencoder.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Encoder separado para features
        encoder = keras.Model(encoder_input, encoded)
        
        self.models[name] = autoencoder
        self.models[f"{name}_encoder"] = encoder
        
        return autoencoder
    
    def create_lstm_sequence_model(self, sequence_length: int, n_features: int, name: str = "lstm") -> Optional[Any]:
        """Cria modelo LSTM para sequências temporais"""
        if not TENSORFLOW_AVAILABLE:
            return None
            
        model = keras.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=(sequence_length, n_features)),
            layers.Dropout(0.3),
            
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            
            layers.LSTM(32),
            layers.Dropout(0.2),
            
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.models[name] = model
        return model
    
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray = None, y_val: np.ndarray = None,
                   epochs: int = 100, batch_size: int = 32) -> Dict[str, Any]:
        """Treina modelo de deep learning"""
        if not TENSORFLOW_AVAILABLE or model_name not in self.models:
            return {}
            
        model = self.models[model_name]
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7
        )
        
        # Preparar dados de validação
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        # Treinar
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        return {
            'history': history.history,
            'final_epoch': len(history.history['loss'])
        }

class EnsembleModels:
    """
    Modelos de Ensemble avançados
    """
    
    def __init__(self):
        self.models = {}
        self.meta_model = None
    
    def create_voting_ensemble(self, X_train: np.ndarray, y_train: np.ndarray) -> VotingClassifier:
        """Cria ensemble de voting"""
        
        # Modelos base
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, scale_pos_weight=10)
        lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight='balanced')
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)
        
        # Ensemble de voting
        voting_clf = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('xgb', xgb_model),
                ('lgb', lgb_model),
                ('gb', gb)
            ],
            voting='soft'  # Usa probabilidades
        )
        
        self.models['voting_ensemble'] = voting_clf
        return voting_clf
    
    def create_stacking_ensemble(self, X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
        """Cria ensemble de stacking"""
        
        # Modelos de nível 1 (base learners)
        level1_models = [
            ('rf', RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)),
            ('xgb', xgb.XGBClassifier(n_estimators=50, max_depth=4, random_state=42)),
            ('lgb', lgb.LGBMClassifier(n_estimators=50, max_depth=4, random_state=42)),
            ('svm', SVC(probability=True, random_state=42)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500))
        ]
        
        # Treinar modelos de nível 1 e gerar meta-features
        meta_features = np.zeros((X_train.shape[0], len(level1_models)))
        
        # Cross-validation para gerar meta-features
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for i, (name, model) in enumerate(level1_models):
            print(f"   Treinando {name} para stacking...")
            
            fold_predictions = np.zeros(X_train.shape[0])
            
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train = y_train[train_idx]
                
                # Treinar modelo no fold
                model.fit(X_fold_train, y_fold_train)
                
                # Predizer no validation set
                fold_predictions[val_idx] = model.predict_proba(X_fold_val)[:, 1]
            
            meta_features[:, i] = fold_predictions
            
            # Retreinar no dataset completo para uso posterior
            model.fit(X_train, y_train)
        
        # Meta-modelo (nível 2)
        meta_model = LogisticRegression(random_state=42, class_weight='balanced')
        meta_model.fit(meta_features, y_train)
        
        # Salvar componentes
        self.models['level1_models'] = dict(level1_models)
        self.meta_model = meta_model
        
        return meta_model
    
    def predict_stacking(self, X_test: np.ndarray) -> np.ndarray:
        """Faz predição com ensemble de stacking"""
        if 'level1_models' not in self.models or self.meta_model is None:
            raise ValueError("Stacking ensemble não foi treinado")
        
        # Gerar meta-features com modelos de nível 1
        meta_features = np.zeros((X_test.shape[0], len(self.models['level1_models'])))
        
        for i, (name, model) in enumerate(self.models['level1_models'].items()):
            meta_features[:, i] = model.predict_proba(X_test)[:, 1]
        
        # Predição final com meta-modelo
        return self.meta_model.predict_proba(meta_features)[:, 1]

class AutoMLOptimizer:
    """
    Otimização automática de hiperparâmetros usando Optuna
    """
    
    def __init__(self):
        self.study = None
        self.best_model = None
    
    def optimize_xgboost(self, X_train: np.ndarray, y_train: np.ndarray, 
                        n_trials: int = 50) -> Dict[str, Any]:
        """Otimiza hiperparâmetros do XGBoost"""
        if not OPTUNA_AVAILABLE:
            print("⚠️ Optuna não disponível. Usando parâmetros padrão.")
            return {}
        
        def objective(trial):
            # Definir espaço de busca
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42
            }
            
            # Modelo
            model = xgb.XGBClassifier(**params)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=5, scoring='roc_auc', n_jobs=-1
            )
            
            return cv_scores.mean()
        
        # Criar estudo
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Treinar melhor modelo
        best_params = self.study.best_params
        self.best_model = xgb.XGBClassifier(**best_params)
        self.best_model.fit(X_train, y_train)
        
        return {
            'best_params': best_params,
            'best_score': self.study.best_value,
            'model': self.best_model
        }

class AdvancedModelPipeline:
    """
    Pipeline completo para modelos avançados
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.results = {}
        
        self.dl_models = DeepLearningModels()
        self.ensemble_models = EnsembleModels()
        self.automl = AutoMLOptimizer()
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, 
                    test_size: float = 0.3, balance_method: str = 'smote') -> Tuple:
        """Prepara dados com balanceamento e normalização"""
        
        # Split inicial
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Normalização
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['main'] = scaler
        
        # Balanceamento
        if balance_method == 'smote':
            sampler = SMOTE(random_state=42)
        elif balance_method == 'adasyn':
            sampler = ADASYN(random_state=42)
        elif balance_method == 'smote_tomek':
            sampler = SMOTETomek(random_state=42)
        else:
            sampler = None
        
        if sampler:
            X_train_balanced, y_train_balanced = sampler.fit_resample(X_train_scaled, y_train)
        else:
            X_train_balanced, y_train_balanced = X_train_scaled, y_train
        
        print(f"📊 Dados preparados:")
        print(f"   Treino original: {X_train.shape[0]} amostras")
        print(f"   Treino balanceado: {X_train_balanced.shape[0]} amostras")
        print(f"   Teste: {X_test.shape[0]} amostras")
        
        return X_train_balanced, X_test_scaled, y_train_balanced, y_test, X_train_scaled, y_train
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Treina todos os modelos avançados"""
        
        results = {}
        
        print("🤖 Treinando modelos avançados...")
        
        # 1. Modelos tradicionais otimizados
        print("   1. XGBoost otimizado com AutoML...")
        if OPTUNA_AVAILABLE:
            automl_result = self.automl.optimize_xgboost(X_train, y_train, n_trials=30)
            results['xgb_optimized'] = automl_result
        
        # 2. Ensemble de Voting
        print("   2. Voting Ensemble...")
        voting_model = self.ensemble_models.create_voting_ensemble(X_train, y_train)
        voting_model.fit(X_train, y_train)
        results['voting_ensemble'] = voting_model
        
        # 3. Stacking Ensemble
        print("   3. Stacking Ensemble...")
        stacking_model = self.ensemble_models.create_stacking_ensemble(X_train, y_train)
        results['stacking_ensemble'] = stacking_model
        
        # 4. Deep Learning Models
        if TENSORFLOW_AVAILABLE:
            print("   4. Deep Learning Models...")
            
            # Feedforward NN
            ff_model = self.dl_models.create_feedforward_nn(X_train.shape[1], "feedforward")
            if ff_model:
                ff_history = self.dl_models.train_model("feedforward", X_train, y_train, X_val, y_val)
                results['feedforward_nn'] = {'model': ff_model, 'history': ff_history}
            
            # Autoencoder
            ae_model = self.dl_models.create_autoencoder_anomaly_detector(X_train.shape[1], "autoencoder")
            if ae_model:
                # Treinar autoencoder apenas com dados normais
                normal_data = X_train[y_train == 0]
                ae_history = self.dl_models.train_model("autoencoder", normal_data, normal_data)
                results['autoencoder'] = {'model': ae_model, 'history': ae_history}
        
        self.results = results
        return results
    
    def evaluate_all_models(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """Avalia todos os modelos treinados"""
        
        evaluation_results = []
        
        for model_name, model_data in self.results.items():
            try:
                if model_name == 'xgb_optimized' and 'model' in model_data:
                    model = model_data['model']
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                elif model_name in ['voting_ensemble']:
                    model = model_data
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                elif model_name == 'stacking_ensemble':
                    y_pred_proba = self.ensemble_models.predict_stacking(X_test)
                    
                elif model_name in ['feedforward_nn', 'autoencoder'] and TENSORFLOW_AVAILABLE:
                    model = model_data['model']
                    if model_name == 'autoencoder':
                        # Para autoencoder, usar erro de reconstrução como score
                        reconstructed = model.predict(X_test)
                        mse = np.mean((X_test - reconstructed) ** 2, axis=1)
                        y_pred_proba = (mse - mse.min()) / (mse.max() - mse.min())  # Normalizar
                    else:
                        y_pred_proba = model.predict(X_test).flatten()
                
                else:
                    continue
                
                # Calcular métricas
                auc_score = roc_auc_score(y_test, y_pred_proba)
                ap_score = average_precision_score(y_test, y_pred_proba)
                
                evaluation_results.append({
                    'model': model_name,
                    'auc_score': auc_score,
                    'average_precision': ap_score
                })
                
                print(f"   {model_name}: AUC={auc_score:.4f}, AP={ap_score:.4f}")
                
            except Exception as e:
                print(f"   Erro ao avaliar {model_name}: {e}")
        
        return pd.DataFrame(evaluation_results).sort_values('auc_score', ascending=False)

def train_advanced_models(X: np.ndarray, y: np.ndarray) -> Tuple[AdvancedModelPipeline, pd.DataFrame]:
    """
    Função principal para treinar todos os modelos avançados
    """
    
    print("🚀 Iniciando treinamento de modelos avançados...")
    
    # Inicializar pipeline
    pipeline = AdvancedModelPipeline()
    
    # Preparar dados
    X_train, X_test, y_train, y_test, X_train_orig, y_train_orig = pipeline.prepare_data(X, y)
    
    # Treinar modelos
    results = pipeline.train_all_models(X_train, y_train, X_test, y_test)
    
    # Avaliar modelos
    evaluation_df = pipeline.evaluate_all_models(X_test, y_test)
    
    print("\n🏆 RESULTADOS FINAIS:")
    print("=" * 50)
    print(evaluation_df.to_string(index=False))
    
    return pipeline, evaluation_df
