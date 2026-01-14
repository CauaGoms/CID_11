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

def get_query_embedding(text):
    """Gera o embedding para a busca usando o Ollama."""
    payload = {"model": EMBED_MODEL, "prompt": text}
    try:
        response = requests.post(OLLAMA_API_EMBED, json=payload, timeout=60)
        return response.json().get("embedding")
    except Exception as e:
        print(f" Erro ao gerar embedding de busca: {e}")
        return None

def buscar_no_banco_jsonl(termo, contexto, capitulo, top_k=5):
    """Lê o JSONL do capítulo e compara os vetores."""
    # Ajuste a extensão para .json ou .jsonl conforme seus arquivos
    nome_arquivo = f"{capitulo}.jsonl" if os.path.exists(os.path.join(PASTA_BANCOS, f"{capitulo}.jsonl")) else f"{capitulo}.json"
    caminho_banco = os.path.join(PASTA_BANCOS, nome_arquivo)
    
    if not os.path.exists(caminho_banco):
        return []

    # 1. Gerar embedding do termo de busca (Termo + Contexto)
    query_text = f"Termo: {termo} | Contexto: {contexto}"
    query_vec = get_query_embedding(query_text)
    if not query_vec: return []
    query_vec = np.array(query_vec).reshape(1, -1)

    candidatos = []

    # 2. Ler o arquivo JSONL linha por linha
    with open(caminho_banco, 'r', encoding='utf-8') as f:
        for linha in f:
            item = json.loads(linha)
            
            # Converter a lista de embedding do JSON para array numpy
            item_vec = np.array(item["embedding"]).reshape(1, -1)
            
            # Calcular similaridade de cosseno
            score = cosine_similarity(query_vec, item_vec)[0][0]
            
            candidatos.append({
                item["id"]: {
                    "confidence_embedding": round(float(score), 4),
                    "text": item["text"][:100] # Opcional: apenas para debug
                }
            })

    # 3. Ordenar e retornar os Top K
    candidatos.sort(key=lambda x: list(x.values())[0]["confidence_embedding"], reverse=True)
    return candidatos[:top_k]

def processar_busca_final():
    if not os.path.exists(PASTA_SAIDA): os.makedirs(PASTA_SAIDA)

    for nome_arquivo in os.listdir(PASTA_ENTRADA):
        if not nome_arquivo.endswith('.json'): continue
        
        print(f"Processando busca vetorial: {nome_arquivo}")
        with open(os.path.join(PASTA_ENTRADA, nome_arquivo), 'r', encoding='utf-8') as f:
            dados = json.load(f)

        contexto = dados.get("text", "")
        labels = dados.get("labels", {})
        novos_labels = {}

        for termo, info in labels.items():
            cap = info.get("capitulo")
            print(f"   -> Buscando '{termo}' no capítulo {cap}")
            
            opcoes = buscar_no_banco_jsonl(termo, contexto, cap)
            
            novos_labels[termo] = {
                "capitulo": cap,
                "opcoes": opcoes
            }

        dados["labels"] = novos_labels
        
        with open(os.path.join(PASTA_SAIDA, nome_arquivo), 'w', encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    processar_busca_final()