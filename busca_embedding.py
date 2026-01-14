import json
import os
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURAÇÕES ---
OLLAMA_API = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "mxbai-embed-large"
PASTA_ENTRADA = "processamento/classifica_entidades/prontuarios_classificados"
PASTA_SAIDA = "processamento/busca_embedding/prontuarios_busca"
PASTA_BANCOS = "embedding_cid" # Onde estão os arquivos de códigos por capítulo

def get_embedding(text):
    """Gera embedding usando o Ollama."""
    payload = {"model": EMBED_MODEL, "prompt": text}
    try:
        response = requests.post(OLLAMA_API, json=payload)
        return response.json()["embedding"]
    except Exception as e:
        print(f"Erro ao gerar embedding: {e}")
        return None

def buscar_no_capitulo(termo, contexto, capitulo, top_k=5):
    """
    Simula a busca no banco vetorial do capítulo específico.
    Aqui você deve integrar com seu banco real (FAISS, Chroma, etc).
    """
    caminho_banco = os.path.join(PASTA_BANCOS, f"{capitulo}.json")
    
    if not os.path.exists(caminho_banco):
        return []

    with open(caminho_banco, 'r', encoding='utf-8') as f:
        banco_dados = json.load(f) # Formato esperado: [{"codigo": "...", "descricao": "...", "embedding": [...]}]

    # Gerar embedding da consulta (Termo + Contexto para desambiguação)
    query_text = f"Termo: {termo}. Contexto: {contexto}"
    query_embed = np.array(get_embedding(query_text)).reshape(1, -1)

    resultados = []
    for item in banco_dados:
        item_embed = np.array(item["embedding"]).reshape(1, -1)
        score = cosine_similarity(query_embed, item_embed)[0][0]
        resultados.append({item["codigo"]: {"confidence_embedding": round(float(score), 4)}})

    # Ordenar por score e pegar os top_k
    resultados.sort(key=lambda x: list(x.values())[0]["confidence_embedding"], reverse=True)
    return resultados[:top_k]

def processar_busca():
    if not os.path.exists(PASTA_SAIDA): os.makedirs(PASTA_SAIDA)

    for nome_arquivo in os.listdir(PASTA_ENTRADA):
        if not nome_arquivo.endswith('.json'): continue
        
        print(f"Buscando códigos para: {nome_arquivo}")
        with open(os.path.join(PASTA_ENTRADA, nome_arquivo), 'r', encoding='utf-8') as f:
            dados = json.load(f)

        contexto = dados.get("text", "")
        labels = dados.get("labels", {})
        novos_labels = {}

        for termo, info in labels.items():
            capitulo = info.get("capitulo")
            print(f"  -> Buscando '{termo}' no banco do Cap {capitulo}")
            
            # Realiza a busca vetorial
            opcoes = buscar_no_capitulo(termo, contexto, capitulo)
            
            novos_labels[termo] = {
                "capitulo": capitulo,
                "opcoes": opcoes
            }

        dados["labels"] = novos_labels
        
        with open(os.path.join(PASTA_SAIDA, nome_arquivo), 'w', encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    processar_busca()