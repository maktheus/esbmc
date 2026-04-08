import streamlit as st
import ollama
import re

st.set_page_config(page_title="IA Híbrida - Vendas & Descontos", page_icon="💸")

# =====================================================================
# PARTE 1: A "REDE NEURAL ESPECIALISTA" (Simulada e Verificável)
# Na vida real, esta função seria um código C gerado a partir do seu 
# modelo PyTorch, auditado e matematicamente provado pelo ESBMC.
# =====================================================================
def relu(x: float) -> float:
    return max(0.0, x)

def expert_neural_network_discount(quantidade_itens: int) -> float:
    # O Forward Pass MATEMÁTICO REAL idêntico ao código C (.c) do ESBMC!
    # Os "pesos" da rede neural extraídos e fixados na memória
    W1 = [0.05, 0.01]
    b1 = [-0.5, 0.0]
    W2 = [0.06, 0.02]
    b2 = 0.00
    
    # Camada Oculta + Função de Ativação Não-Linear (ReLU)
    hidden_0 = relu((quantidade_itens * W1[0]) + b1[0])
    hidden_1 = relu((quantidade_itens * W1[1]) + b1[1])
    
    # Camada de Saída (Matemática Pura de Vetores)
    desconto = (hidden_0 * W2[0]) + (hidden_1 * W2[1]) + b2
    
    # Mantém o limite operacional da política da loja
    if desconto > 0.30: 
        desconto = 0.30
        
    return desconto

# =====================================================================
# PARTE 2: EXTRAÇÃO DE DADOS MÍNIMA
# =====================================================================
def extract_quantity(text: str) -> int:
    # Procura números inteiros simples no texto (ex: "quero 30 camisas")
    match = re.search(r'\b(\d+)\b', text)
    return int(match.group(1)) if match else 0

# =====================================================================
# PARTE 3: O LLM (QWEN) COMO RELAÇÕES PÚBLICAS
# =====================================================================
st.title("💸 Chatbot Híbrido: Padrão Ouro")
st.caption("Lógica blindada (padrão ESBMC) + Qwen 2.5 para conversa")

if "history" not in st.session_state:
    st.session_state["history"] = []

# Exibe histórico
for msg in st.session_state.history:
    if msg["role"] != "system":
        st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ex: Olá, eu gostaria de comprar 50 uniformes hospitalares..."):
    # Mostra mensagem do usuário
    st.session_state.history.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("Processando Inteligência Híbrida..."):
        # 1. O Sistema Cego (Rede Neural) analisa os dados
        quantidade = extract_quantity(prompt)
        desconto_matematico = expert_neural_network_discount(quantidade)
        desconto_texto = f"{int(desconto_matematico * 100)}%"

        # 2. Construímos o prompt dinâmico injetando a 'Verdade Matemática' pro Qwen
        system_instruction = f"""Você é um vendedor da loja Personal Confecções.
        Regra estrita: O sistema central da loja já fechou o cálculo matemático. 
        O desconto aprovado para este cliente é de EXATOS {desconto_texto}.
        Sua única função agora é ler o que o cliente digitou e dar a notícia de que 
        ele ganhou {desconto_texto} de desconto de forma super educada, amigável e entusiasmada. 
        Não liste os produtos, foque apenas em celebrar o desconto gerado pela quantidade!"""

        # 3. Mandamos para a Qwen atuar
        messages_to_qwen = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ]

        try:
            response = ollama.chat(model="qwen2.5:1.5b", messages=messages_to_qwen)
            qwen_reply = response['message']['content']
            
            st.session_state.history.append({"role": "assistant", "content": qwen_reply})
            st.chat_message("assistant").write(qwen_reply)
        except Exception as e:
            st.error("Erro na Qwen. O Ollama está rodando?")
