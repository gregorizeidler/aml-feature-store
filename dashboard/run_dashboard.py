"""
Script para executar o dashboard Streamlit
"""

import subprocess
import sys
import os

def run_dashboard():
    """Executa o dashboard Streamlit"""
    
    # Verificar se streamlit está instalado
    try:
        import streamlit
        print("✅ Streamlit encontrado")
    except ImportError:
        print("❌ Streamlit não encontrado. Instalando...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "plotly"])
    
    # Executar dashboard
    dashboard_path = os.path.join(os.path.dirname(__file__), "app.py")
    
    print("🚀 Iniciando dashboard AML...")
    print("📊 Acesse: http://localhost:8501")
    print("⏹️  Para parar: Ctrl+C")
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", dashboard_path,
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])

if __name__ == "__main__":
    run_dashboard()
