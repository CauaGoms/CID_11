import json
import os
from collections import Counter

def analisar_cids_extensivo(caminho_pasta, arquivo_saida="relatorio_final_cid.txt"):
    # Estrutura: {cid: {'count': 0, 'desc': '', 'termos': Counter(), 'co_cids': Counter()}}
    stats = {}

    if not os.path.exists(caminho_pasta):
        print("Erro: Pasta não encontrada.")
        return

    arquivos = [f for f in os.listdir(caminho_pasta) if f.endswith('.json')]
    
    for nome_arquivo in arquivos:
        try:
            with open(os.path.join(caminho_pasta, nome_arquivo), 'r', encoding='utf-8') as f:
                dados = json.load(f)
                labels = dados.get("labels", {})
                entidades_dict = dados.get("entities", {})
                
                # 1. Coletar todos os termos únicos de todas as entidades deste prontuário
                todos_termos_prontuario = []
                for lista_termos in entidades_dict.values():
                    todos_termos_prontuario.extend(lista_termos)
                
                # 2. Identificar CIDs mantidos neste prontuário
                cids_mantidos = [cid for cid, info in labels.items() if info.get("decisao") == "MANTER"]

                for cid in cids_mantidos:
                    if cid not in stats:
                        stats[cid] = {
                            'count': 0, 
                            'desc': labels[cid].get("descricao_cid", "N/A"),
                            'termos': Counter(), 
                            'co_cids': Counter()
                        }
                    
                    stats[cid]['count'] += 1
                    
                    # Contabilizar termos de entidades (exceto o próprio termo do CID para evitar redundância óbvia)
                    termo_proprio = labels[cid].get("term_original", "").upper()
                    for t in todos_termos_prontuario:
                        if t.upper() != termo_proprio:
                            stats[cid]['termos'][t.upper()] += 1
                    
                    # Contabilizar outros CIDs no mesmo prontuário
                    for outro_cid in cids_mantidos:
                        if outro_cid != cid:
                            stats[cid]['co_cids'][outro_cid] += 1
                            
        except Exception:
            continue

    # Ordenação por frequência do CID
    cids_ordenados = sorted(stats.items(), key=lambda x: x[1]['count'], reverse=True)

    with open(arquivo_saida, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("ANÁLISE DE CORRELAÇÃO: CID-11 vs TERMOS DE ENTIDADES vs OUTROS CIDs\n")
        f.write("="*100 + "\n\n")

        for cid, data in cids_ordenados:
            f.write(f"CID: {cid}\n")
            f.write(f"DESCRIÇÃO: {data['desc'][:120]}...\n")
            f.write(f"FREQUÊNCIA TOTAL DO CID: {data['count']}\n\n")

            # Top 4 Termos de Entidades
            f.write(f"TOP 4 TERMOS DE ENTIDADES QUE MAIS APARECEM COM ESTE CID:\n")
            top_termos = data['termos'].most_common(4)
            if top_termos:
                for termo, freq in top_termos:
                    f.write(f"   - {termo}: {freq}x\n")
            else:
                f.write("   - Nenhum termo de entidade encontrado.\n")

            # Top CIDs Co-ocorrentes
            f.write(f"\nOUTROS CIDs MAIS COMUNS NO MESMO PRONTUÁRIO:\n")
            top_co = data['co_cids'].most_common(4)
            if top_co:
                for co_cid, freq in top_co:
                    f.write(f"   - {co_cid}: {freq}x\n")
            else:
                f.write("   - Este CID geralmente aparece sozinho.\n")

            f.write("\n" + "-"*60 + "\n\n")

    print(f"Relatório detalhado gerado: {arquivo_saida}")

# Execução
caminho = "./prontuarios_auditados/"
analisar_cids_extensivo(caminho)