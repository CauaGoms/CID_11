import json
import requests

PORTA = "8080" 
# Verifique se a versão no seu Docker é 2025-01 ou 2024-01
VERSAO = "2025-01" 
BASE_URL = f"http://localhost:{PORTA}/icd/release/11/{VERSAO}/mms"

HEADERS = {
    'Accept': 'application/json',
    'Accept-Language': 'pt',
    'API-Version': 'v2'
}

def obter_detalhes_completo(codigo):
    try:
        # 1. Busca o CodeInfo
        lookup_url = f"{BASE_URL}/codeinfo/{codigo}"
        res = requests.get(lookup_url, headers=HEADERS, timeout=10)
        
        if res.status_code == 200:
            data = res.json()
            entity_uri = data.get('stemId')
            
            # Ajusta URI para o seu localhost
            local_url = entity_uri.replace("https://id.who.int", f"http://localhost:{PORTA}")
            
            # 2. Busca Detalhes da Entidade
            detalhes_res = requests.get(local_url, headers=HEADERS, timeout=10)
            if detalhes_res.status_code == 200:
                detalhes = detalhes_res.json()
                
                # Vamos tentar capturar o Título e a Definição com segurança
                titulo = detalhes.get('title', {}).get('@value', 'Sem título')
                # Algumas entidades usam 'definition', outras apenas o título
                definicao = detalhes.get('definition', {}).get('@value', 'Nenhuma definição detalhada encontrada para este código.')
                
                return f"Título: {titulo} | Definição: {definicao}"
        
        return f"Erro: Status {res.status_code} no lookup do código {codigo}"
    except Exception as e:
        return f"Erro de conexão: {str(e)}"

if __name__ == "__main__":
    # Testando apenas com Cólera (1A00) para ver a estrutura
    print(f"--- Iniciando Diagnóstico no Docker (Porta {PORTA}) ---")
    resultado = obter_detalhes_completo('1A00')
    print(f"\nResultado para 1A00:\n{resultado}")