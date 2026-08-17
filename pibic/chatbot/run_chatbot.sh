#!/bin/bash
# Script para iniciar o Chatbot da loja

echo "Iniciando processo de inicialização..."

# Cria o ambiente virtual e instala as dependências se necessário
if [ ! -d "chat_venv" ]; then
    echo "Criando ambiente virtual (venv)..."
    python3 -m venv chat_venv
fi

source chat_venv/bin/activate

echo "Instalando dependências (Streamlit e Ollama)..."
pip install streamlit ollama -q

# Baixa o Qwen se não tiver
echo "Aguarde enquanto carregamos o Cérebro do Chatbot (Qwen 1.5b)..."
ollama pull qwen2.5:1.5b

# Roda o sistema
echo -e "\n=== Tudo pronto! Iniciando a tela do Chat! ==="
echo "Seu navegador deve abrir automaticamente."
echo "Caso não abra, acesse: http://localhost:8501"

streamlit run /home/uchoa/chatbot_qwen_vendas.py
