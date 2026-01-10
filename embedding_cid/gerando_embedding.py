import json
import requests

# Configurações
OLLAMA_ENDPOINT = "http://localhost:11434/api/embeddings"
MODEL_NAME = "mxbai-embed-large"
INPUT_FILE = 'codigos_cid/ICD-11-com-descritivo.json'
OUTPUT_FILE = 'cid_embeddings_mxbai.jsonl'

def gerar_embedding_ollama(texto):
    """Faz a chamada ao Ollama para gerar o vetor."""
    payload = {
        "model": MODEL_NAME,
        "prompt": texto
    }
    try:
        response = requests.post(OLLAMA_ENDPOINT, json=payload)
        return response.json().get('embedding')
    except Exception as e:
        print(f"Erro ao gerar embedding: {e}")
        return None

def processar_para_jsonl():
    print(f"Iniciando geração de embeddings com {MODEL_NAME}...")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        dados = json.load(f)

    total = len(dados)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for i, item in enumerate(dados):
            # Combinamos título e descrição para dar contexto ao modelo
            # O mxbai se beneficia de textos bem estruturados
            texto_formatado = f"Representação de doença CID-11: {item['valor']}. Definição: {item['descricao']}"
            
            print(f"[{i+1}/{total}] Vetorizando: {item['identificador']}", end='\r')
            
            vetor = gerar_embedding_ollama(texto_formatado)
            
            if vetor:
                # Criamos a linha no formato JSONL
                linha = {
                    "id": item['identificador'],
                    "text": texto_formatado,
                    "embedding": vetor,
                    "metadata": {
                        "nome_original": item['valor'],
                        "codigo": item['identificador']
                    }
                }
                f_out.write(json.dumps(linha, ensure_ascii=False) + '\n')

    print(f"\nConcluído! Arquivo '{OUTPUT_FILE}' gerado com sucesso.")

if __name__ == "__main__":
    processar_para_jsonl()