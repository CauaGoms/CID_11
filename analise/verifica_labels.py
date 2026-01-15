import json
import os

def gerar_relatorio_simples(caminho_pasta, arquivo_txt="arquivos_vazios.txt"):
    arquivos_com_labels_vazias = []

    # Verifica se a pasta existe
    if not os.path.exists(caminho_pasta):
        print(f"Erro: A pasta '{caminho_pasta}' não foi encontrada.")
        return

    # Lista arquivos .json na pasta
    arquivos = [f for f in os.listdir(caminho_pasta) if f.endswith('.json')]

    for nome_arquivo in arquivos:
        caminho_completo = os.path.join(caminho_pasta, nome_arquivo)
        
        try:
            with open(caminho_completo, 'r', encoding='utf-8') as f:
                dados = json.load(f)
                labels = dados.get("labels")

                # Verifica se a chave 'labels' existe e se é um dicionário vazio {}
                if labels == {}:
                    arquivos_com_labels_vazias.append(nome_arquivo)
                    
        except json.JSONDecodeError:
            # Arquivos corrompidos que não puderam ser lidos
            continue

    # Ordena os nomes em ordem crescente
    arquivos_com_labels_vazias.sort()

    # Grava apenas os nomes no arquivo TXT
    with open(arquivo_txt, 'w', encoding='utf-8') as f_out:
        for nome in arquivos_com_labels_vazias:
            f_out.write(nome + "\n")

    print(f"Relatório gerado com sucesso!")
    print(f"Total de arquivos com labels vazias: {len(arquivos_com_labels_vazias)}")

# --- CONFIGURAÇÃO ---
# Informe o caminho da sua pasta aqui
caminho_da_pasta = "./prontuarios_auditados/"

gerar_relatorio_simples(caminho_da_pasta)