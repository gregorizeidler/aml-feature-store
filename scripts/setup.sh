#!/bin/bash

# Script de setup automático para AML Feature Store
# Autor: Seu Nome
# Data: $(date)

set -e  # Parar em caso de erro

echo "🚀 Iniciando setup do AML Feature Store..."
echo "================================================"

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Função para log colorido
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Verificar se Docker está instalado
check_docker() {
    log_info "Verificando Docker..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker não encontrado. Por favor, instale o Docker primeiro."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose não encontrado. Por favor, instale o Docker Compose primeiro."
        exit 1
    fi
    
    log_success "Docker e Docker Compose encontrados"
}

# Verificar se Python está instalado
check_python() {
    log_info "Verificando Python..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 não encontrado. Por favor, instale Python 3.9+ primeiro."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    REQUIRED_VERSION="3.9"
    
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
        log_error "Python 3.9+ é necessário. Versão encontrada: $PYTHON_VERSION"
        exit 1
    fi
    
    log_success "Python $PYTHON_VERSION encontrado"
}

# Criar ambiente virtual
setup_venv() {
    log_info "Configurando ambiente virtual Python..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        log_success "Ambiente virtual criado"
    else
        log_warning "Ambiente virtual já existe"
    fi
    
    # Ativar ambiente virtual
    source venv/bin/activate
    
    # Atualizar pip
    pip install --upgrade pip
    
    log_success "Ambiente virtual configurado"
}

# Instalar dependências Python
install_python_deps() {
    log_info "Instalando dependências Python..."
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        log_success "Dependências Python instaladas"
    else
        log_error "Arquivo requirements.txt não encontrado"
        exit 1
    fi
}

# Iniciar serviços Docker
start_docker_services() {
    log_info "Iniciando serviços Docker..."
    
    # Parar serviços existentes se estiverem rodando
    docker-compose down 2>/dev/null || true
    
    # Iniciar serviços em background
    docker-compose up -d
    
    log_success "Serviços Docker iniciados"
    
    # Aguardar serviços ficarem prontos
    log_info "Aguardando serviços ficarem prontos..."
    sleep 30
    
    # Verificar status dos serviços
    if docker-compose ps | grep -q "Up"; then
        log_success "Serviços estão rodando"
    else
        log_error "Alguns serviços falharam ao iniciar"
        docker-compose ps
        exit 1
    fi
}

# Configurar Feast Feature Store
setup_feast() {
    log_info "Configurando Feast Feature Store..."
    
    cd feature_repo
    
    # Aplicar definições de features
    feast apply
    
    cd ..
    
    log_success "Feast Feature Store configurado"
}

# Gerar dados de exemplo
generate_sample_data() {
    log_info "Gerando dados de exemplo..."
    
    cd offline_data
    python generate_sample_data.py
    cd ..
    
    log_success "Dados de exemplo gerados"
}

# Testar API
test_api() {
    log_info "Testando API (aguarde alguns segundos para inicialização)..."
    
    # Iniciar API em background
    cd api
    nohup uvicorn main:app --host 0.0.0.0 --port 8000 > ../api.log 2>&1 &
    API_PID=$!
    cd ..
    
    # Aguardar API inicializar
    sleep 10
    
    # Testar health check
    if curl -s http://localhost:8000/health > /dev/null; then
        log_success "API está respondendo"
        
        # Executar testes básicos
        cd api
        python test_api.py
        cd ..
        
    else
        log_warning "API não está respondendo ainda. Verifique os logs em api.log"
    fi
    
    # Salvar PID para cleanup posterior
    echo $API_PID > api.pid
}

# Função de cleanup
cleanup() {
    log_info "Limpando processos..."
    
    # Parar API se estiver rodando
    if [ -f "api.pid" ]; then
        API_PID=$(cat api.pid)
        kill $API_PID 2>/dev/null || true
        rm api.pid
    fi
}

# Configurar trap para cleanup
trap cleanup EXIT

# Função principal
main() {
    echo "🏦 AML Feature Store - Setup Automático"
    echo "======================================="
    echo ""
    
    # Verificações iniciais
    check_docker
    check_python
    
    # Setup do ambiente
    setup_venv
    install_python_deps
    
    # Iniciar infraestrutura
    start_docker_services
    
    # Configurar Feature Store
    setup_feast
    
    # Gerar dados
    generate_sample_data
    
    # Testar sistema
    test_api
    
    echo ""
    echo "🎉 Setup concluído com sucesso!"
    echo "================================"
    echo ""
    echo "📋 Próximos passos:"
    echo "  1. Acesse a API: http://localhost:8000/docs"
    echo "  2. Monitore o Flink: http://localhost:8081"
    echo "  3. Execute o notebook: jupyter notebook notebooks/aml_analysis.ipynb"
    echo ""
    echo "🔧 Comandos úteis:"
    echo "  • Parar serviços: docker-compose down"
    echo "  • Ver logs: docker-compose logs -f"
    echo "  • Reiniciar: docker-compose restart"
    echo ""
    echo "📊 Para iniciar o produtor de dados:"
    echo "  cd producer && python transaction_producer.py --rate 5.0"
    echo ""
    echo "🚀 Sistema pronto para uso!"
}

# Executar função principal
main "$@"
