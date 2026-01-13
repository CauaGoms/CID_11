import json
import requests

# Ajuste a porta se você mudou no comando 'docker run' (ex: 8080 ou 80)
PORTA = "8080" 
BASE_URL = f"http://localhost:{PORTA}/icd/release/11/2025-01/mms"

HEADERS = {
    'Accept': 'application/json',
    'Accept-Language': 'pt',
    'API-Version': 'v2'
}

def testar_conexao():
    print(f"--- Testando conexão com Docker em localhost:{PORTA} ---")
    try:
        # Testa se o servidor base está respondendo
        res = requests.get(f"http://localhost:{PORTA}/", timeout=5)
        print(f"Status do Servidor: {res.status_code} (OK)")
        return True
    except Exception as e:
        print(f"ERRO: Não consegui conectar no Docker. O container está rodando?")
        print(f"Detalhe: {e}")
        return False

def obter_descricao_local(codigo):
    try:
        # 1. Busca o CodeInfo
        lookup_url = f"{BASE_URL}/codeinfo/{codigo}"
        res = requests.get(lookup_url, headers=HEADERS, timeout=10)
        
        if res.status_code == 200:
            data = res.json()
            entity_uri = data.get('stemId')
            
            # Ajusta URI para o seu localhost
            local_url = entity_uri.replace("https://id.who.int", f"http://localhost:{PORTA}")
            
            # 2. Busca Detalhes
            detalhes_res = requests.get(local_url, headers=HEADERS, timeout=10)
            if detalhes_res.status_code == 200:
                detalhes = detalhes_res.json()
                # Pega a definição
                return detalhes.get('definition', {}).get('@value', "Sem descrição disponível.")
        
        return f"Erro na API: Status {res.status_code}"
    except Exception as e:
        return f"Erro: {str(e)}"

if __name__ == "__main__":
    if testar_conexao():
        # Carrega o JSON original
        with open('ICD-11-pt-clean.json', 'r', encoding='utf-8') as f:
            dados = json.load(f)

        print("\n--- Iniciando Teste com os 5 primeiros códigos ---")
        teste_reduzido = dados[:5]
        
        for item in teste_reduzido:
            cod = item['identificador']
            print(f"Consultando {cod} ({item['valor']})...")
            item['descricao'] = obter_descricao_local(cod)
            print(f"Resultado: {item['descricao'][:100]}...\n")

        # Salva o teste
        with open('teste_docker_cid.json', 'w', encoding='utf-8') as f:
            json.dump(teste_reduzido, f, ensure_ascii=False, indent=2)
        
        print("Arquivo 'teste_docker_cid.json' gerado!")