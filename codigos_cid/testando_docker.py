import json
import requests

# Configurações de acordo com seu 'docker ps'
PORTA = "8080"
VERSAO = "2025-01"
BASE_URL = f"http://localhost:{PORTA}/icd/release/11/{VERSAO}/mms"

HEADERS = {
    'Accept': 'application/json',
    'Accept-Language': 'pt',
    'API-Version': 'v2'
}

def teste_unitario(codigo):
    print(f"--- Iniciando Teste para o código: {codigo} ---")
    
    try:
        # Passo 1: Lookup do Código
        lookup_url = f"{BASE_URL}/codeinfo/{codigo}"
        print(f"1. Fazendo lookup em: {lookup_url}")
        res_lookup = requests.get(lookup_url, headers=HEADERS, timeout=10)
        
        if res_lookup.status_code != 200:
            print(f"ERRO no Lookup: Status {res_lookup.status_code}")
            return

        data = res_lookup.json()
        entity_uri = data.get('stemId')
        print(f"   Sucesso! URI da entidade encontrada: {entity_uri}")

        # Passo 2: Buscar detalhes na URI local
        local_uri = entity_uri.replace("https://id.who.int", f"http://localhost:{PORTA}")
        print(f"2. Buscando detalhes em: {local_uri}")
        res_detalhes = requests.get(local_uri, headers=HEADERS, timeout=10)

        if res_detalhes.status_code == 200:
            detalhes = res_detalhes.json()
            titulo = detalhes.get('title', {}).get('@value', 'Sem título')
            # Verifica se existe definição, senão avisa
            definicao = detalhes.get('definition', {}).get('@value', 'AVISO: Este código não possui definição detalhada em PT.')
            
            print(f"\n--- RESULTADO DO TESTE ---")
            print(f"CÓDIGO: {codigo}")
            print(f"TÍTULO: {titulo}")
            print(f"DESCRIÇÃO: {definicao[:200]}...")
            print(f"--------------------------")
        else:
            print(f"ERRO nos Detalhes: Status {res_detalhes.status_code}")

    except Exception as e:
        print(f"ERRO DE CONEXÃO: {e}")

if __name__ == "__main__":
    # Testamos com 1A00 (Cólera) que é um código padrão e costuma ter descrição
    teste_unitario('1A00')