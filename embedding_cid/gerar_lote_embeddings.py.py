import json
import requests
import os

# --- CONFIGURAÇÕES ---
OLLAMA_ENDPOINT = "http://localhost:11434/api/embeddings"
MODEL_NAME = "mxbai-embed-large"
PASTA_ENTRADA = "codigos_cid/codigos_cid_jsons"
PASTA_SAIDA = "embedding_cid"

# Garante que a pasta de saída existe
os.makedirs(PASTA_SAIDA, exist_ok=True)

def gerar_embedding_ollama(texto):
    """Faz a chamada ao Ollama para gerar o vetor."""
    payload = {
        "model": MODEL_NAME,
        "prompt": texto
    }
    try:
        response = requests.post(OLLAMA_ENDPOINT, json=payload)
        response.raise_for_status()
        return response.json().get('embedding')
    except Exception as e:
        print(f"\nErro ao gerar embedding: {e}")
        return None

def processar_arquivos():
    # Lista todos os arquivos .json na pasta de entrada
    arquivos = [f for f in os.listdir(PASTA_ENTRADA) if f.endswith('.json')]
    
    if not arquivos:
        print(f"Nenhum arquivo JSON encontrado em {PASTA_ENTRADA}")
        return

    print(f"Iniciando processamento de {len(arquivos)} arquivos com {MODEL_NAME}...")

    for nome_arquivo in arquivos:
        caminho_entrada = os.path.join(PASTA_ENTRADA, nome_arquivo)
        # Define o nome de saída trocando .json por .jsonl
        nome_saida = nome_arquivo.replace('.json', '.jsonl')
        caminho_saida = os.path.join(PASTA_SAIDA, nome_saida)

        print(f"\nProcessando arquivo: {nome_arquivo}")
        
        try:
            with open(caminho_entrada, 'r', encoding='utf-8') as f:
                dados = json.load(f)

            total_itens = len(dados)
            
            with open(caminho_saida, 'w', encoding='utf-8') as f_out:
                for i, item in enumerate(dados):
                    identificador = item.get('identificador', 'N/A')
                    valor = item.get('valor', '')
                    # Se a descrição estiver vazia, usamos uma string padrão para não quebrar o modelo
                    descricao = item.get('descricao', '')
                    
                    # Texto estruturado para o mxbai-embed-large
                    texto_formatado = f"Representação de doença CID-11: {valor}. Definição: {descricao}"
                    
                    print(f" > [{i+1}/{total_itens}] Vetorizando: {identificador}", end='\r')
                    
                    vetor = gerar_embedding_ollama(texto_formatado)
                    
                    if vetor:
                        linha = {
                            "id": identificador,
                            "text": texto_formatado,
                            "embedding": vetor,
                            "metadata": {
                                "nome_original": valor,
                                "codigo": identificador
                            }
                        }
                        f_out.write(json.dumps(linha, ensure_ascii=False) + '\n')
            
            print(f"\nArquivo '{nome_saida}' concluído!")

        except Exception as e:
            print(f"Erro ao processar o arquivo {nome_arquivo}: {e}")

    print("\n" + "="*50)
    print(f"PROCESSO FINALIZADO! Verifique a pasta '{PASTA_SAIDA}'.")

if __name__ == "__main__":
    processar_arquivos()