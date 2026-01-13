#Script para analisar códigos CID-11 na pasta codigos_cid_jsonsimport json
import json  # Certifique-se de que esta linha está no topo
import os

def analisar_arquivos_json(diretorio_entrada, arquivo_relatorio):
    if not os.path.exists(diretorio_entrada):
        print(f"Erro: O diretório '{diretorio_entrada}' não foi encontrado.")
        return

    arquivos = [f for f in os.listdir(diretorio_entrada) if f.endswith('.json')]
    
    if not arquivos:
        print("Nenhum arquivo JSON encontrado na pasta.")
        return

    print(f"Analisando {len(arquivos)} arquivos em '{diretorio_entrada}'...")

    total_geral_ids = 0
    total_geral_com_descricao = 0

    with open(arquivo_relatorio, 'w', encoding='utf-8') as f_relatorio:
        f_relatorio.write("RELATÓRIO DE ANÁLISE DE COBERTURA - CID-11\n")
        f_relatorio.write("="*60 + "\n\n")

        for nome_arquivo in sorted(arquivos):
            caminho_completo = os.path.join(diretorio_entrada, nome_arquivo)
            
            try:
                with open(caminho_completo, 'r', encoding='utf-8') as f_json:
                    # Carrega os dados garantindo que o módulo json está disponível
                    conteudo = f_json.read()
                    if not conteudo.strip():
                        continue
                    
                    dados = json.loads(conteudo)

                if isinstance(dados, list):
                    qtd_total = len(dados)
                    # Verifica se a chave 'descricao' existe, não é None e não é vazia
                    qtd_com_desc = sum(1 for item in dados if item.get('descricao') and str(item['descricao']).strip())
                    
                    porcentagem = (qtd_com_desc / qtd_total * 100) if qtd_total > 0 else 0
                    
                    f_relatorio.write(f"Arquivo: {nome_arquivo}\n")
                    f_relatorio.write(f" - Total de Identificadores: {qtd_total}\n")
                    f_relatorio.write(f" - Com Descrição preenchida: {qtd_com_desc}\n")
                    f_relatorio.write(f" - Cobertura: {porcentagem:.2f}%\n")
                    f_relatorio.write("-" * 40 + "\n")

                    total_geral_ids += qtd_total
                    total_geral_com_descricao += qtd_com_desc
            
            except Exception as e:
                f_relatorio.write(f"Arquivo: {nome_arquivo} -> ERRO AO PROCESSAR: {str(e)}\n")
                f_relatorio.write("-" * 40 + "\n")

        # Resumo Final
        f_relatorio.write("\n" + "="*60 + "\n")
        f_relatorio.write("RESUMO GERAL DO PROJETO\n")
        f_relatorio.write(f"Total de Identificadores analisados: {total_geral_ids}\n")
        f_relatorio.write(f"Total com Descrição preenchida: {total_geral_com_descricao}\n")
        
        cobertura_total = (total_geral_com_descricao / total_geral_ids * 100) if total_geral_ids > 0 else 0
        f_relatorio.write(f"Cobertura Global de Dados: {cobertura_total:.2f}%\n")
        f_relatorio.write("="*60 + "\n")

    print(f"Pronto! Relatório gerado em: {arquivo_relatorio}")

if __name__ == "__main__":
    # Ajuste aqui os nomes das pastas se necessário
    PASTA = "codigos_cid_jsons"
    RELATORIO = "analise_cobertura_cid.txt"
    
    analisar_arquivos_json(PASTA, RELATORIO)