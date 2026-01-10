import json
import requests

# Configurações da API Local (Docker)
BASE_URL = "http://localhost:8080/icd/release/11/2025-01/mms"
HEADERS = {
    'Accept': 'application/json',
    'Accept-Language': 'pt',
    'API-Version': 'v2'
}

def obter_descricao(codigo):
    """Consulta a API local para obter a definição da doença."""
    try:
        # 1. Busca o CodeInfo para obter a URI da entidade
        lookup_url = f"{BASE_URL}/codeinfo/{codigo}"
        res = requests.get(lookup_url, headers=HEADERS, timeout=5)
        
        if res.status_code == 200:
            entity_uri = res.json().get('stemId')
            # Ajusta a URL para apontar para o localhost
            local_url = entity_uri.replace("https://id.who.int", "http://localhost:8080")
            
            # 2. Busca os detalhes da entidade
            detalhes_res = requests.get(local_url, headers=HEADERS, timeout=5)
            if detalhes_res.status_code == 200:
                dados = detalhes_res.json()
                # Tenta pegar a definição, se não existir, retorna aviso
                return dados.get('definition', {}).get('@value', "Descrição não disponível nesta versão.")
        
        return "Código não encontrado ou sem detalhes."
    except Exception as e:
        return f"Erro na consulta: {str(e)}"

def processar_arquivo():
    print("Iniciando o processamento... Isso pode levar alguns minutos.")
    
    # Carrega o seu arquivo original
    with open('ICD-11-pt-clean.json', 'r', encoding='utf-8') as f:
        dados_originais = json.load(f)

    novo_resultado = []
    total = len(dados_originais)

    for i, item in enumerate(dados_originais):
        codigo = item['identificador']
        print(f"[{i+1}/{total}] Processando: {codigo} - {item['valor'][:30]}...", end='\r')
        
        # Obtém a descrição via API local
        descricao = obter_descricao(codigo)
        
        # Adiciona o novo campo ao dicionário
        item['descricao'] = descricao
        novo_resultado.append(item)

    # Salva o novo arquivo enriquecido
    with open('ICD-11-com-descritivo.json', 'w', encoding='utf-8') as f:
        json.dump(novo_resultado, f, ensure_ascii=False, indent=2)

    print(f"\n\nConcluído! Arquivo 'ICD-11-com-descritivo.json' gerado com sucesso.")

if __name__ == "__main__":
    processar_arquivo()