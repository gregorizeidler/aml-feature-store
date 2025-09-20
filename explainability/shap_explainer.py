"""
Sistema de Explicabilidade Avançada usando SHAP
Fornece explicações interpretáveis para predições de AML
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# SHAP para explicabilidade
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP não disponível. Instale com: pip install shap")

# LIME como alternativa
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("⚠️ LIME não disponível. Instale com: pip install lime")

# Plotly para visualizações interativas
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

class SHAPExplainer:
    """
    Explicador usando SHAP (SHapley Additive exPlanations)
    """
    
    def __init__(self, model, X_train: pd.DataFrame, feature_names: List[str]):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
        if SHAP_AVAILABLE:
            self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Inicializa o explainer SHAP apropriado"""
        
        # Detectar tipo de modelo e usar explainer apropriado
        model_type = type(self.model).__name__
        
        if 'XGB' in model_type or 'LightGBM' in model_type or 'CatBoost' in model_type:
            # Tree explainer para modelos baseados em árvore
            self.explainer = shap.TreeExplainer(self.model)
            print("🌳 Usando TreeExplainer para modelo baseado em árvore")
            
        elif 'RandomForest' in model_type or 'GradientBoosting' in model_type:
            # Tree explainer para sklearn tree models
            self.explainer = shap.TreeExplainer(self.model)
            print("🌳 Usando TreeExplainer para modelo sklearn")
            
        else:
            # Kernel explainer para outros modelos (mais lento mas universal)
            background_sample = shap.sample(self.X_train, 100)  # Amostra para background
            self.explainer = shap.KernelExplainer(self.model.predict_proba, background_sample)
            print("🔧 Usando KernelExplainer (universal)")
    
    def explain_prediction(self, X_instance: pd.DataFrame, 
                          return_dict: bool = True) -> Dict[str, Any]:
        """Explica uma predição específica"""
        
        if not SHAP_AVAILABLE or self.explainer is None:
            return {"error": "SHAP não disponível"}
        
        # Calcular SHAP values
        if hasattr(self.explainer, 'shap_values'):
            shap_values = self.explainer.shap_values(X_instance)
            
            # Para classificação binária, pegar valores da classe positiva
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Classe positiva
        else:
            shap_values = self.explainer(X_instance).values
            if len(shap_values.shape) > 2:
                shap_values = shap_values[:, :, 1]  # Classe positiva
        
        # Preparar explicação
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]  # Primeira instância
        
        # Criar dicionário de explicação
        explanation = {
            'prediction': self.model.predict_proba(X_instance)[0][1],
            'base_value': self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, (list, np.ndarray)) else self.explainer.expected_value,
            'shap_values': dict(zip(self.feature_names, shap_values)),
            'feature_values': dict(zip(self.feature_names, X_instance.iloc[0].values)),
            'top_positive_features': [],
            'top_negative_features': []
        }
        
        # Identificar features mais importantes
        feature_importance = [(name, shap_val, feat_val) 
                            for name, shap_val, feat_val in 
                            zip(self.feature_names, shap_values, X_instance.iloc[0].values)]
        
        # Ordenar por impacto absoluto
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Separar features positivas e negativas
        positive_features = [(name, shap_val, feat_val) for name, shap_val, feat_val in feature_importance if shap_val > 0]
        negative_features = [(name, shap_val, feat_val) for name, shap_val, feat_val in feature_importance if shap_val < 0]
        
        explanation['top_positive_features'] = positive_features[:5]
        explanation['top_negative_features'] = negative_features[:5]
        
        return explanation
    
    def explain_dataset(self, X_dataset: pd.DataFrame, max_samples: int = 1000):
        """Explica um dataset completo"""
        
        if not SHAP_AVAILABLE or self.explainer is None:
            return None
        
        # Limitar amostras para performance
        if len(X_dataset) > max_samples:
            X_sample = X_dataset.sample(n=max_samples, random_state=42)
        else:
            X_sample = X_dataset
        
        print(f"Calculando SHAP values para {len(X_sample)} amostras...")
        
        # Calcular SHAP values
        if hasattr(self.explainer, 'shap_values'):
            shap_values = self.explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Classe positiva
        else:
            shap_values = self.explainer(X_sample).values
            if len(shap_values.shape) > 2:
                shap_values = shap_values[:, :, 1]  # Classe positiva
        
        self.shap_values = shap_values
        self.X_explained = X_sample
        
        return shap_values
    
    def plot_waterfall(self, explanation: Dict[str, Any], 
                      title: str = "Explicação da Predição") -> Optional[Any]:
        """Cria gráfico waterfall da explicação"""
        
        if not SHAP_AVAILABLE:
            return None
        
        # Preparar dados para waterfall
        features = list(explanation['shap_values'].keys())
        shap_vals = list(explanation['shap_values'].values())
        
        # Ordenar por impacto absoluto
        sorted_indices = sorted(range(len(shap_vals)), key=lambda i: abs(shap_vals[i]), reverse=True)
        
        # Top 10 features
        top_features = [features[i] for i in sorted_indices[:10]]
        top_shap_vals = [shap_vals[i] for i in sorted_indices[:10]]
        
        # Criar gráfico
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Base value
        base_value = explanation['base_value']
        prediction = explanation['prediction']
        
        # Calcular posições
        cumulative = base_value
        positions = [base_value]
        
        for shap_val in top_shap_vals:
            cumulative += shap_val
            positions.append(cumulative)
        
        # Cores
        colors = ['red' if val < 0 else 'green' for val in top_shap_vals]
        
        # Barras
        for i, (feature, shap_val, pos) in enumerate(zip(top_features, top_shap_vals, positions[1:])):
            prev_pos = positions[i]
            
            if shap_val > 0:
                ax.bar(i+1, shap_val, bottom=prev_pos, color='green', alpha=0.7)
            else:
                ax.bar(i+1, abs(shap_val), bottom=pos, color='red', alpha=0.7)
            
            # Adicionar valor
            ax.text(i+1, pos + shap_val/2, f'{shap_val:.3f}', 
                   ha='center', va='center', fontweight='bold')
        
        # Base value e predição final
        ax.axhline(y=base_value, color='blue', linestyle='--', alpha=0.7, label=f'Base Value: {base_value:.3f}')
        ax.axhline(y=prediction, color='purple', linestyle='-', alpha=0.7, label=f'Predição: {prediction:.3f}')
        
        # Configurar eixos
        ax.set_xticks(range(len(top_features) + 1))
        ax.set_xticklabels(['Base'] + top_features, rotation=45, ha='right')
        ax.set_ylabel('Contribuição para a Predição')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self) -> Optional[Any]:
        """Plota importância global das features"""
        
        if self.shap_values is None:
            print("Execute explain_dataset() primeiro")
            return None
        
        # Calcular importância média
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        # Criar DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=True)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 12))
        
        bars = ax.barh(importance_df['feature'], importance_df['importance'], 
                      color='skyblue', alpha=0.8)
        
        # Adicionar valores
        for bar, importance in zip(bars, importance_df['importance']):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{importance:.3f}', va='center', ha='left')
        
        ax.set_xlabel('Importância Média (|SHAP|)')
        ax.set_title('Importância Global das Features')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

class LIMEExplainer:
    """
    Explicador usando LIME (Local Interpretable Model-agnostic Explanations)
    """
    
    def __init__(self, model, X_train: pd.DataFrame, feature_names: List[str]):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.explainer = None
        
        if LIME_AVAILABLE:
            self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Inicializa o explainer LIME"""
        
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X_train.values,
            feature_names=self.feature_names,
            class_names=['Normal', 'Suspeita'],
            mode='classification',
            discretize_continuous=True
        )
        
        print("🔍 LIME explainer inicializado")
    
    def explain_prediction(self, X_instance: pd.DataFrame, 
                          num_features: int = 10) -> Dict[str, Any]:
        """Explica uma predição usando LIME"""
        
        if not LIME_AVAILABLE or self.explainer is None:
            return {"error": "LIME não disponível"}
        
        # Explicar instância
        explanation = self.explainer.explain_instance(
            X_instance.iloc[0].values,
            self.model.predict_proba,
            num_features=num_features
        )
        
        # Extrair informações
        lime_explanation = {
            'prediction': self.model.predict_proba(X_instance)[0][1],
            'lime_explanation': explanation.as_list(),
            'intercept': explanation.intercept[1],
            'local_pred': explanation.local_pred[1]
        }
        
        return lime_explanation

class ExplanationGenerator:
    """
    Gerador de explicações em linguagem natural
    """
    
    def __init__(self):
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 1.0
        }
    
    def generate_natural_explanation(self, explanation: Dict[str, Any], 
                                   customer_id: str, amount: float) -> str:
        """Gera explicação em linguagem natural"""
        
        prediction = explanation['prediction']
        
        # Determinar nível de risco
        if prediction <= self.risk_thresholds['low']:
            risk_level = "BAIXO"
            risk_color = "🟢"
        elif prediction <= self.risk_thresholds['medium']:
            risk_level = "MÉDIO"
            risk_color = "🟡"
        else:
            risk_level = "ALTO"
            risk_color = "🔴"
        
        # Início da explicação
        explanation_text = f"""
{risk_color} ANÁLISE DE RISCO - CLIENTE {customer_id}
{'='*50}

📊 RESULTADO DA ANÁLISE:
   • Score de Risco: {prediction:.3f}
   • Nível de Risco: {risk_level}
   • Valor da Transação: R$ {amount:,.2f}

🔍 PRINCIPAIS FATORES QUE INFLUENCIARAM A DECISÃO:
"""
        
        # Fatores positivos (aumentam risco)
        positive_factors = explanation.get('top_positive_features', [])
        if positive_factors:
            explanation_text += "\n   🔺 FATORES DE RISCO (aumentam suspeita):\n"
            
            for i, (feature, shap_val, feat_val) in enumerate(positive_factors[:3], 1):
                feature_explanation = self._explain_feature(feature, feat_val, shap_val)
                explanation_text += f"   {i}. {feature_explanation}\n"
        
        # Fatores negativos (diminuem risco)
        negative_factors = explanation.get('top_negative_features', [])
        if negative_factors:
            explanation_text += "\n   🔻 FATORES PROTETIVOS (diminuem suspeita):\n"
            
            for i, (feature, shap_val, feat_val) in enumerate(negative_factors[:3], 1):
                feature_explanation = self._explain_feature(feature, feat_val, shap_val)
                explanation_text += f"   {i}. {feature_explanation}\n"
        
        # Recomendações
        explanation_text += self._generate_recommendations(prediction, positive_factors)
        
        return explanation_text
    
    def _explain_feature(self, feature_name: str, feature_value: float, 
                        shap_value: float) -> str:
        """Explica uma feature específica em linguagem natural"""
        
        impact = "AUMENTA" if shap_value > 0 else "DIMINUI"
        impact_strength = "fortemente" if abs(shap_value) > 0.1 else "moderadamente"
        
        # Mapeamento de features para explicações
        feature_explanations = {
            'unique_ips_1h': f"Cliente usou {int(feature_value)} IPs diferentes na última hora - {impact} {impact_strength} o risco",
            'velocity_score_1h': f"Velocidade de {feature_value:.1f} transações/hora - {impact} {impact_strength} o risco",
            'txn_amount_sum_5m': f"Volume de R$ {feature_value:,.2f} nos últimos 5 minutos - {impact} {impact_strength} o risco",
            'txn_count_1h': f"Realizou {int(feature_value)} transações na última hora - {impact} {impact_strength} o risco",
            'night_txn_ratio_1h': f"{feature_value*100:.1f}% das transações foram noturnas - {impact} {impact_strength} o risco",
            'amount_deviation_score_1h': f"Desvio de R$ {feature_value:,.2f} no padrão de valores - {impact} {impact_strength} o risco"
        }
        
        # Buscar explicação específica ou usar genérica
        if feature_name in feature_explanations:
            return feature_explanations[feature_name]
        else:
            return f"{feature_name} = {feature_value:.3f} - {impact} {impact_strength} o risco (impacto: {shap_value:.3f})"
    
    def _generate_recommendations(self, prediction: float, 
                                positive_factors: List[Tuple]) -> str:
        """Gera recomendações baseadas na análise"""
        
        recommendations = "\n💡 RECOMENDAÇÕES:\n"
        
        if prediction > 0.8:
            recommendations += "   🚨 AÇÃO IMEDIATA NECESSÁRIA:\n"
            recommendations += "   • Bloquear transação temporariamente\n"
            recommendations += "   • Investigação manual urgente\n"
            recommendations += "   • Contatar cliente para verificação\n"
            
        elif prediction > 0.6:
            recommendations += "   ⚠️ MONITORAMENTO REFORÇADO:\n"
            recommendations += "   • Aprovar com monitoramento\n"
            recommendations += "   • Acompanhar próximas transações\n"
            recommendations += "   • Considerar investigação\n"
            
        elif prediction > 0.3:
            recommendations += "   ℹ️ MONITORAMENTO PADRÃO:\n"
            recommendations += "   • Aprovar transação\n"
            recommendations += "   • Monitoramento de rotina\n"
            
        else:
            recommendations += "   ✅ BAIXO RISCO:\n"
            recommendations += "   • Aprovar transação\n"
            recommendations += "   • Nenhuma ação adicional necessária\n"
        
        # Recomendações específicas baseadas nos fatores
        if positive_factors:
            top_factor = positive_factors[0][0]  # Feature mais importante
            
            if 'unique_ips' in top_factor:
                recommendations += "   • Verificar se IPs são de localizações conhecidas\n"
            elif 'velocity' in top_factor:
                recommendations += "   • Verificar se alta velocidade é justificada\n"
            elif 'night' in top_factor:
                recommendations += "   • Investigar razão para transações noturnas\n"
        
        return recommendations

class ComprehensiveExplainer:
    """
    Sistema completo de explicabilidade combinando SHAP e LIME
    """
    
    def __init__(self, model, X_train: pd.DataFrame, feature_names: List[str]):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        
        # Inicializar explicadores
        self.shap_explainer = SHAPExplainer(model, X_train, feature_names) if SHAP_AVAILABLE else None
        self.lime_explainer = LIMEExplainer(model, X_train, feature_names) if LIME_AVAILABLE else None
        self.explanation_generator = ExplanationGenerator()
    
    def explain_transaction(self, X_instance: pd.DataFrame, 
                          customer_id: str, amount: float,
                          method: str = 'shap') -> Dict[str, Any]:
        """Explica uma transação completamente"""
        
        result = {
            'customer_id': customer_id,
            'amount': amount,
            'timestamp': datetime.now(),
            'prediction': self.model.predict_proba(X_instance)[0][1],
            'explanation_method': method
        }
        
        # Usar SHAP por padrão
        if method == 'shap' and self.shap_explainer:
            shap_explanation = self.shap_explainer.explain_prediction(X_instance)
            result['technical_explanation'] = shap_explanation
            
            # Gerar explicação em linguagem natural
            natural_explanation = self.explanation_generator.generate_natural_explanation(
                shap_explanation, customer_id, amount
            )
            result['natural_explanation'] = natural_explanation
            
        # Usar LIME como alternativa
        elif method == 'lime' and self.lime_explainer:
            lime_explanation = self.lime_explainer.explain_prediction(X_instance)
            result['technical_explanation'] = lime_explanation
            
        else:
            result['error'] = f"Método {method} não disponível"
        
        return result
    
    def generate_explanation_report(self, explanations: List[Dict[str, Any]]) -> str:
        """Gera relatório consolidado de explicações"""
        
        report = f"""
📋 RELATÓRIO DE EXPLICAÇÕES AML
{'='*50}
Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total de transações analisadas: {len(explanations)}

"""
        
        # Estatísticas gerais
        predictions = [exp['prediction'] for exp in explanations]
        high_risk_count = sum(1 for p in predictions if p > 0.6)
        medium_risk_count = sum(1 for p in predictions if 0.3 < p <= 0.6)
        low_risk_count = sum(1 for p in predictions if p <= 0.3)
        
        report += f"""
📊 DISTRIBUIÇÃO DE RISCO:
   🔴 Alto Risco (>0.6):    {high_risk_count:3d} ({high_risk_count/len(explanations)*100:.1f}%)
   🟡 Médio Risco (0.3-0.6): {medium_risk_count:3d} ({medium_risk_count/len(explanations)*100:.1f}%)
   🟢 Baixo Risco (<0.3):    {low_risk_count:3d} ({low_risk_count/len(explanations)*100:.1f}%)

"""
        
        # Transações de alto risco
        high_risk_explanations = [exp for exp in explanations if exp['prediction'] > 0.6]
        
        if high_risk_explanations:
            report += "🚨 TRANSAÇÕES DE ALTO RISCO:\n"
            report += "="*30 + "\n"
            
            for i, exp in enumerate(high_risk_explanations[:5], 1):  # Top 5
                report += f"\n{i}. Cliente: {exp['customer_id']}\n"
                report += f"   Score: {exp['prediction']:.3f}\n"
                report += f"   Valor: R$ {exp['amount']:,.2f}\n"
                
                if 'natural_explanation' in exp:
                    # Extrair principais fatores da explicação
                    tech_exp = exp.get('technical_explanation', {})
                    pos_factors = tech_exp.get('top_positive_features', [])
                    if pos_factors:
                        report += f"   Principal fator: {pos_factors[0][0]}\n"
        
        return report

def main():
    """Função principal para demonstração"""
    
    print("🔍 Demonstração do Sistema de Explicabilidade AML")
    print("=" * 60)
    
    # Simular dados para demonstração
    np.random.seed(42)
    n_samples, n_features = 1000, 20
    
    # Gerar features sintéticas
    X = np.random.randn(n_samples, n_features)
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # Gerar target correlacionado
    y = (X[:, 0] + X[:, 1] * 0.5 + X[:, 2] * 0.3 + np.random.randn(n_samples) * 0.1) > 0
    
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Treinar modelo simples
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_df, y)
    
    print("✅ Modelo treinado para demonstração")
    
    # Inicializar explicador
    explainer = ComprehensiveExplainer(model, X_df, feature_names)
    
    # Explicar algumas transações
    explanations = []
    
    for i in range(5):
        X_instance = X_df.iloc[[i]]
        customer_id = f"CUST_{i:06d}"
        amount = np.random.uniform(100, 5000)
        
        explanation = explainer.explain_transaction(
            X_instance, customer_id, amount, method='shap'
        )
        
        explanations.append(explanation)
        
        print(f"\n📋 Explicação {i+1}:")
        if 'natural_explanation' in explanation:
            print(explanation['natural_explanation'])
        else:
            print(f"Score: {explanation['prediction']:.3f}")
    
    # Gerar relatório
    report = explainer.generate_explanation_report(explanations)
    print("\n" + report)

if __name__ == "__main__":
    main()
