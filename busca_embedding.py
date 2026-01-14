import json
import os
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURAÇÕES ---
OLLAMA_API_EMBED = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "mxbai-embed-large"
PASTA_ENTRADA = "processamento/classifica_entidades/prontuarios_classificados"
PASTA_SAIDA = "processamento/busca_embedding/prontuarios_busca"
PASTA_BANCOS = "embedding_cid"

# Cache para evitar ler o mesmo arquivo de capítulo várias vezes no mesmo processo
cache_capitulos = {}

def get_query_embedding(text):
    """Gera o embedding para a busca usando o Ollama."""
    payload = {"model": EMBED_MODEL, "prompt": text}
    try:
        response = requests.post(OLLAMA_API_EMBED, json=payload, timeout=60)
        return response.json().get("embedding")
    except Exception as e:
        print(f" Erro ao gerar embedding de busca: {e}")
        return None

def carregar_banco(capitulo):
    """Carrega o banco do capítulo para a memória (com cache)."""
    if capitulo in cache_capitulos:
        return cache_capitulos[capitulo]
    
    nome_arquivo = f"{capitulo}.jsonl" if os.path.exists(os.path.join(PASTA_BANCOS, f"{capitulo}.jsonl")) else f"{capitulo}.json"
    caminho_banco = os.path.join(PASTA_BANCOS, nome_arquivo)
    
    if not os.path.exists(caminho_banco):
        return None

    dados_capitulo = []
    with open(caminho_banco, 'r', encoding='utf-8') as f:
        for linha in f:
            if linha.strip():
                dados_capitulo.append(json.loads(linha))
    
    cache_capitulos[capitulo] = dados_capitulo
    return dados_capitulo

def buscar_no_banco_memoria(termo, contexto, capitulo, top_k=5):
    """Realiza a busca vetorial usando os dados carregados em memória."""
    banco_dados = carregar_banco(capitulo)
    
    if not banco_dados:
        return []

    # 1. Gerar embedding do termo de busca
    query_text = f"Termo: {termo} | Contexto: {contexto}"
    query_vec = get_query_embedding(query_text)
    if not query_vec: return []
    query_vec = np.array(query_vec).reshape(1, -1)

    candidatos = []

    # 2. Comparar com os itens do banco
    for item in banco_dados:
        item_vec = np.array(item["embedding"]).reshape(1, -1)
        score = cosine_similarity(query_vec, item_vec)[0][0]
        
        candidatos.append({
            item["id"]: {
                "confidence_embedding": round(float(score), 4),
                "text": item["text"]  # TEXTO COMPLETO PRESERVADO AQUI
            }
        })

    # 3. Ordenar e retornar os Top K
    candidatos.sort(key=lambda x: list(x.values())[0]["confidence_embedding"], reverse=True)
    return candidatos[:top_k]

def processar_busca_final():
    if not os.path.exists(PASTA_SAIDA): os.makedirs(PASTA_SAIDA)

    for nome_arquivo in os.listdir(PASTA_ENTRADA):
        if not nome_arquivo.endswith('.json'): continue
        
        # Limpar o cache a cada novo prontuário para não sobrecarregar a RAM 
        # (Opcional, dependendo do tamanho da sua base CID)
        cache_capitulos.clear()
        
        print(f"Processando busca vetorial: {nome_arquivo}")
        with open(os.path.join(PASTA_ENTRADA, nome_arquivo), 'r', encoding='utf-8') as f:
            dados = json.load(f)

        contexto = dados.get("text", "")
        labels = dados.get("labels", {})
        novos_labels = {}

        for termo, info in labels.items():
            cap = info.get("capitulo")
            print(f"   -> Buscando '{termo}' no capítulo {cap}")
            
            opcoes = buscar_no_banco_memoria(termo, contexto, cap)
            
            novos_labels[termo] = {
                "capitulo": cap,
                "opcoes": opcoes
            }

        dados["labels"] = novos_labels
        
        with open(os.path.join(PASTA_SAIDA, nome_arquivo), 'w', encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    processar_busca_final()