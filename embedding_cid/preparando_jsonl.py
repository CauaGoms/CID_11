import json

def criar_jsonl_para_embedding():
    with open('ICD-11-com-descritivo.json', 'r', encoding='utf-8') as f:
        dados = json.load(f)

    with open('dataset_cid_para_embedding.jsonl', 'w', encoding='utf-8') as f:
        for item in dados:
            # Criamos um texto rico combinando Título e Descrição
            texto_para_vetorizar = f"Doença: {item['valor']}. Definição: {item['descricao']}"
            
            # Estrutura JSONL
            linha = {
                "id": item['identificador'],
                "text": texto_para_vetorizar,
                "metadata": {
                    "nome": item['valor']
                }
            }
            f.write(json.dumps(linha, ensure_ascii=False) + '\n')

    print("Arquivo JSONL pronto para virar embeddings!")

criar_jsonl_para_embedding()