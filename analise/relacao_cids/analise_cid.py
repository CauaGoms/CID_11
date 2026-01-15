import json
import os
import pandas as pd

def gerar_tabela_correlacao_com_significado(caminho_prontuarios, caminho_dicionario, arquivo_saida="tabela_correlacao_completa.csv"):
    # 1. Carregar o Dicionário de Significados
    with open(caminho_dicionario, 'r', encoding='utf-8') as f:
        lista_dic = json.load(f)
    
    # Criar um mapa rápido { "1A00": {"valor": "Cólera", "descricao": "..."} }
    dic_significados = {item['identificador']: item for item in lista_dic}

    correlacoes = {} # {CID_A: {CID_B: freq}}
    contagem_total = {}

    # 2. Processar os Prontuários
    arquivos = [f for f in os.listdir(caminho_prontuarios) if f.endswith('.json')]
    
    for nome_arquivo in arquivos:
        try:
            with open(os.path.join(caminho_prontuarios, nome_arquivo), 'r', encoding='utf-8') as f:
                dados = json.load(f)
                cids_mantidos = [cid for cid, info in dados.get("labels", {}).items() if info.get("decisao") == "MANTER"]

                for cid in cids_mantidos:
                    contagem_total[cid] = contagem_total.get(cid, 0) + 1
                    if cid not in correlacoes:
                        correlacoes[cid] = {}
                    
                    for outro_cid in cids_mantidos:
                        if cid != outro_cid:
                            correlacoes[cid][outro_cid] = correlacoes[cid].get(outro_cid, 0) + 1
        except:
            continue

    # 3. Construir a Tabela Final
    dados_tabela = []
    for cid_principal, relacionados in correlacoes.items():
        # Busca informações no dicionário
        info_principal = dic_significados.get(cid_principal, {"valor": "Não encontrado", "descricao": ""})
        
        # Pega os 3 CIDs que mais aparecem com ele
        top_relacionados = sorted(relacionados.items(), key=lambda x: x[1], reverse=True)[:3]
        
        lista_str_relacionados = []
        for c, f in top_relacionados:
            info_rel = dic_significados.get(c, {"valor": "N/A"})
            lista_str_relacionados.append(f"{c} - {info_rel['valor']} ({f}x)")
        
        dados_tabela.append({
            "CID": cid_principal,
            "Nome do CID": info_principal['valor'],
            "Frequência": contagem_total[cid_principal],
            "Significado Completo": info_principal['descricao'],
            "Comorbidades Relacionadas": " | ".join(lista_str_relacionados)
        })

    # 4. Salvar e Ordenar
    df = pd.DataFrame(dados_tabela).sort_values(by="Frequência", ascending=False)
    df.to_csv(arquivo_saida, index=False, sep=';', encoding='utf-8-sig')
    print(f"Sucesso! Tabela gerada com {len(df)} linhas em: {arquivo_saida}")

# --- EXECUÇÃO ---
# caminho_prontuarios: pasta onde estão os arquivos 8907.json, etc.
# caminho_dicionario: o arquivo JSON que você me enviou com os significados.
gerar_tabela_correlacao_com_significado("analise/prontuarios_auditados/", "codigos_cid/ICD-11-com-descricoes.json")