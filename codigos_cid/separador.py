import json
import os

# Definição dos capítulos e seus intervalos (baseado na sua tabela)
# A estrutura é: (Nome do Arquivo, Prefixo/Intervalo de Início)
capitulos_config = [
    ("01_doencas_infecciosas.json", "1"),
    ("02_neoplasias.json", "2"),
    ("03_doencas_sangue.json", "3"),
    ("04_sistema_imune.json", "4"),
    ("05_endocrinas_metabolicas.json", "5"),
    ("06_transtornos_mentais.json", "6"),
    ("07_sono_vigilia.json", "7"),
    ("08_sistema_nervoso.json", "8"),
    ("09_sistema_visual.json", "9"),
    ("10_orelha_mastoide.json", "A"),
    ("11_sistema_circulatorio.json", "B"),
    ("12_sistema_respiratorio.json", "C"),
    ("13_sistema_digestivo.json", "D"),
    ("14_pele.json", "E"),
    ("15_musculoesqueletico.json", "F"),
    ("16_geniturinario.json", "G"),
    ("17_saude_sexual.json", "H"),
    ("18_gravidez_parto.json", "J"),
    ("19_perinatal.json", "K"),
    ("20_anomalias_desenvolvimento.json", "L"),
    ("21_sintomas_achados.json", "M"),
    ("22_lesoes_causas_externas.json", "N"),
    ("23_causas_externas_morbidade.json", "P"),
    ("24_fatores_estado_saude.json", "Q"),
    ("25_propositos_especiais.json", "R"),
    ("26_medicina_tradicional.json", "S"),
    ("V_avaliacao_funcionalidade.json", "V"),
    ("X_codigos_extensao.json", "X")
]

def dividir_cid(arquivo_origem):
    # Criar pasta de destino se não existir
    pasta_destino = 'codigos_cid_jsons'
    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)

    # Carregar o arquivo JSON original
    with open(arquivo_origem, 'r', encoding='utf-8') as f:
        dados = json.load(f)

    # Dicionário para armazenar os dados de cada capítulo
    resultado_capitulos = {nome: [] for nome, _ in capitulos_config}

    # Distribuir os itens
    for item in dados:
        identificador = item.get('identificador', '')
        if not identificador:
            continue
        
        encontrado = False
        # Verifica em qual capítulo o código se encaixa pelo primeiro caractere
        for nome_arquivo, prefixo in capitulos_config:
            if identificador.startswith(prefixo):
                resultado_capitulos[nome_arquivo].append(item)
                encontrado = True
                break
        
        if not encontrado:
            print(f"Aviso: Código {identificador} não se encaixou em nenhum capítulo.")

    # Salvar os 28 arquivos
    for nome_arquivo, conteudo in resultado_capitulos.items():
        caminho_completo = os.path.join(pasta_destino, nome_arquivo)
        with open(caminho_completo, 'w', encoding='utf-8') as f_out:
            json.dump(conteudo, f_out, indent=2, ensure_ascii=False)
        print(f"Arquivo {nome_arquivo} criado com {len(conteudo)} itens.")

# Execução do script
if __name__ == "__main__":
    arquivo_input = 'ICD-11-com-descricoes.json' 
    if os.path.exists(arquivo_input):
        dividir_cid(arquivo_input)
    else:
        print(f"Erro: O arquivo {arquivo_input} não foi encontrado no diretório atual.")