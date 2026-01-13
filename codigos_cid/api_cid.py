import json
import requests

# Configurações conforme seu ambiente Docker
PORTA = "8080"
VERSAO = "2025-01"
BASE_URL = f"http://localhost:{PORTA}/icd/release/11/{VERSAO}/mms"

HEADERS = {
    'Accept': 'application/json',
    'Accept-Language': 'pt',
    'API-Version': 'v2'
}

def obter_descricao_docker(codigo):
    """
    Busca a descrição no Docker local usando a lógica de 
    forçar o localhost para evitar erro 401.
    """
    try:
        # 1. Faz o lookup do identificador para pegar o ID interno
        lookup_url = f"{BASE_URL}/codeinfo/{codigo}"
        res_lookup = requests.get(lookup_url, headers=HEADERS, timeout=5)
        
        if res_lookup.status_code == 200:
            data = res_lookup.json()
            # Extrai o ID numérico final da URI (ex: 257068234)
            entity_id = data.get('stemId').split('/')[-1]
            
            # 2. Busca detalhes forçando o localhost
            local_uri = f"{BASE_URL}/{entity_id}"
            res_detalhes = requests.get(local_uri, headers=HEADERS, timeout=5)
            
            if res_detalhes.status_code == 200:
                detalhes = res_detalhes.json()
                # Pega a definição. Se não existir, retorna string vazia para tratar depois.
                return detalhes.get('definition', {}).get('@value', "")
        
        return ""
    except Exception:
        return ""

def processar():
    arquivo_entrada = 'ICD-11-pt-clean.json'
    arquivo_saida = 'ICD-11-com-descricoes.json'

    print(f"Lendo {arquivo_entrada}...")
    with open(arquivo_entrada, 'r', encoding='utf-8') as f:
        dados = json.load(f)

    total = len(dados)
    resultado_final = []

    print(f"Iniciando enriquecimento de {total} itens. Aguarde...")

    for i, item in enumerate(dados):
        codigo = item['identificador']
        valor_original = item['valor']
        
        # Busca a descrição no Docker
        descricao = obter_descricao_docker(codigo)
        
        # Cria o novo objeto mantendo os dados originais e somando a descrição
        novo_item = {
            "identificador": codigo,
            "valor": valor_original,
            "descricao": descricao
        }
        
        resultado_final.append(novo_item)
        
        # Feedback de progresso no terminal
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"Progresso: [{i+1}/{total}] - Último: {codigo}", end='\r')

    # Salva o novo JSON
    with open(arquivo_saida, 'w', encoding='utf-8') as f:
        json.dump(resultado_final, f, ensure_ascii=False, indent=2)

    print(f"\n\nSucesso! Arquivo '{arquivo_saida}' gerado com as descrições.")

if __name__ == "__main__":
    processar()