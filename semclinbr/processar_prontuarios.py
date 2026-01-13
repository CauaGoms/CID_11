import xml.etree.ElementTree as ET
import json
import os
import re

def carregar_traducoes(caminho_json):
    with open(caminho_json, 'r', encoding='utf-8') as f:
        return json.load(f)

def processar_xmls(pasta_origem, pasta_destino, mapa_traducoes):
    # Cria a pasta de destino se não existir
    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)

    # Lista todos os arquivos XML na pasta de origem
    arquivos_xml = [f for f in os.listdir(pasta_origem) if f.endswith('.xml')]

    for idx, nome_arquivo in enumerate(arquivos_xml, start=1):
        caminho_xml = os.path.join(pasta_origem, nome_arquivo)
        
        try:
            tree = ET.parse(caminho_xml)
            root = tree.getroot()

            # Extração do Texto Original
            texto_prontuario = root.find('TEXT').text if root.find('TEXT') is not None else ""

            # Extração e Tradução das Entidades
            entities = {}
            tags_node = root.find('TAGS')
            if tags_node is not None:
                for ann in tags_node.findall('annotation'):
                    tag_crua = ann.get('tag', '')
                    texto_entidade = ann.get('text', '')

                    # Lida com tags compostas (ex: "Temporal Concept|Abbreviation")
                    partes_tag = tag_crua.split('|')
                    
                    for parte in partes_tag:
                        # Traduz usando o mapa, ou mantém o original se não encontrar
                        tag_traduzida = mapa_traducoes.get(parte.strip(), parte.strip())

                        if tag_traduzida not in entities:
                            entities[tag_traduzida] = []
                        
                        if texto_entidade not in entities[tag_traduzida]:
                            entities[tag_traduzida].append(texto_entidade)

            # Montagem da Estrutura Final
            dados_processados = {
                "prontuario_id": idx,
                "text": texto_prontuario,
                "entities": entities,
                "labels": {} # Campo vazio conforme solicitado
            }

            # Salvamento do arquivo JSON
            nome_saida = os.path.splitext(nome_arquivo)[0] + ".json"
            caminho_saida = os.path.join(pasta_destino, nome_saida)

            with open(caminho_saida, 'w', encoding='utf-8') as f_json:
                json.dump(dados_processados, f_json, ensure_ascii=False, indent=2)

            print(f"Sucesso: {nome_arquivo} -> {nome_saida}")

        except Exception as e:
            print(f"Erro ao processar {nome_arquivo}: {e}")

if __name__ == "__main__":
    # Configurações de caminhos
    PASTA_ENTRADA = "semclinbr/prontuarios_xml" # Coloque seus XMLs aqui
    PASTA_SAIDA = "processamento/prontuarios_processados"
    ARQUIVO_TRADUCAO = "processamento/traducao_entidades.json"

    # Execução
    traducoes = carregar_traducoes(ARQUIVO_TRADUCAO)
    processar_xmls(PASTA_ENTRADA, PASTA_SAIDA, traducoes)