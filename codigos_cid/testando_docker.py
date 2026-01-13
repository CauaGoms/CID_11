import json
import requests

PORTA = "8080"
VERSAO = "2025-01"
BASE_URL = f"http://localhost:{PORTA}/icd/release/11/{VERSAO}/mms"

HEADERS = {
    'Accept': 'application/json',
    'Accept-Language': 'pt',
    'API-Version': 'v2'
}

def teste_unitario_corrigido(codigo):
    print(f"--- Iniciando Teste Corrigido para: {codigo} ---")
    
    try:
        # 1. Fazendo lookup no seu Docker
        lookup_url = f"{BASE_URL}/codeinfo/{codigo}"
        print(f"1. Consultando Local: {lookup_url}")
        res_lookup = requests.get(lookup_url, headers=HEADERS, timeout=10)
        
        if res_lookup.status_code == 200:
            data = res_lookup.json()
            # A API retorna o link da internet, nós vamos ignorar o domínio e pegar só o final
            entity_id = data.get('stemId').split('/')[-1] 
            
            # Montamos a URL FORÇANDO o localhost
            local_uri = f"{BASE_URL}/{entity_id}"
            print(f"2. Buscando detalhes FORÇADOS no Docker: {local_uri}")
            
            res_detalhes = requests.get(local_uri, headers=HEADERS, timeout=10)

            if res_detalhes.status_code == 200:
                detalhes = res_detalhes.json()
                titulo = detalhes.get('title', {}).get('@value', 'Sem título')
                definicao = detalhes.get('definition', {}).get('@value', 'Sem definição detalhada.')
                
                print(f"\n--- SUCESSO! ---")
                print(f"TÍTULO: {titulo}")
                print(f"DESCRIÇÃO: {definicao[:150]}...")
            else:
                print(f"ERRO nos Detalhes: Status {res_detalhes.status_code}")
                print("Se persistir o 401, o container precisa ser reiniciado sem autenticação.")
        else:
            print(f"ERRO no Lookup: Status {res_lookup.status_code}")

    except Exception as e:
        print(f"ERRO: {e}")

if __name__ == "__main__":
    teste_unitario_corrigido('1A00')