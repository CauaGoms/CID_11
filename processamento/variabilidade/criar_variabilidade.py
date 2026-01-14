import json
import os

# --- CONFIGURAÇÕES ---
PASTA_AUDITADA = "processamento/seleciona_labels/prontuarios_auditados"
ARQUIVO_VARIABILIDADE = "variabilidade.json"

def processar_variabilidade():
    # Estrutura para armazenar os dados consolidados
    # { "codigo": { "official_name": "...", "variations": { "termo": {"freq": X, "ids": []} } } }
    mapa_variabilidade = {}

    if not os.path.exists(PASTA_AUDITADA):
        print(f"Erro: A pasta {PASTA_AUDITADA} não existe.")
        return

    arquivos = [f for f in os.listdir(PASTA_AUDITADA) if f.endswith('.json')]
    
    print(f"Iniciando análise de variabilidade em {len(arquivos)} arquivos...")

    for nome_arquivo in arquivos:
        caminho_completo = os.path.join(PASTA_AUDITADA, nome_arquivo)
        
        try:
            with open(caminho_completo, 'r', encoding='utf-8') as f:
                dados = json.load(f)

            prontuario_id = dados.get("prontuario_id", "N/A")
            labels = dados.get("labels", {})

            for codigo, info in labels.items():
                # REGRA: Apenas se a decisão da auditoria foi MANTER
                if info.get("decisao") == "MANTER":
                    termo = info.get("term_original", "").lower().strip()
                    descricao_oficial = info.get("descricao_cid", "")
                    
                    # Limpeza simples da descrição para pegar apenas o nome (antes da "Definição:")
                    nome_oficial = descricao_oficial.split("Definição:")[0].replace("Representação de doença CID-11:", "").strip()
                    if not nome_oficial:
                        nome_oficial = "Descrição não disponível"

                    # Inicializa o código no mapa se não existir
                    if codigo not in mapa_variabilidade:
                        mapa_variabilidade[codigo] = {
                            "official_name": nome_oficial,
                            "variations": {}
                        }

                    # Inicializa ou atualiza o termo específico
                    if termo not in mapa_variabilidade[codigo]["variations"]:
                        mapa_variabilidade[codigo]["variations"][termo] = {
                            "frequency": 0,
                            "id_prontuarios": set() # Usando set para evitar duplicar ID no mesmo termo
                        }
                    
                    mapa_variabilidade[codigo]["variations"][termo]["frequency"] += 1
                    mapa_variabilidade[codigo]["variations"][termo]["id_prontuarios"].add(prontuario_id)

        except Exception as e:
            print(f"Erro ao processar arquivo {nome_arquivo}: {e}")

    # --- FORMATAÇÃO FINAL ---
    # Converte os sets em listas e organiza a estrutura conforme o exemplo solicitado
    estrutura_final = {}
    for codigo, conteudo in mapa_variabilidade.items():
        estrutura_final[codigo] = {
            "official_name": conteudo["official_name"],
            "variations": []
        }
        
        for termo, stats in conteudo["variations"].items():
            estrutura_final[codigo]["variations"].append({
                "term": termo,
                "frequency": stats["frequency"],
                "id_prontuarios": sorted(list(stats["id_prontuarios"]))
            })

    # Salva o resultado final no JSON
    with open(ARQUIVO_VARIABILIDADE, 'w', encoding='utf-8') as f:
        json.dump(estrutura_final, f, ensure_ascii=False, indent=2)

    print(f"Sucesso! Relatório de variabilidade gerado em: {ARQUIVO_VARIABILIDADE}")

if __name__ == "__main__":
    processar_variabilidade()