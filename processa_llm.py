import json
import os
import requests

# --- CONFIGURAÇÕES ---
OLLAMA_API = "http://localhost:11434/api/generate"
MODELO = "llama3.1:8b"
PASTA_ENTRADA = "processamento/pront_teste"
PASTA_SAIDA = "processamento/prontuarios_classificados"
TAMANHO_CHUNK = 2000 

DESC_CAPITULOS = """
01: Algumas doenças infecciosas ou parasitárias
02: Neoplasias
03: Doenças do sangue ou dos órgãos formadores do sangue
04: Doenças do sistema imune
05: Doenças endócrinas, nutricionais ou metabólicas
06: Transtornos mentais, comportamentais ou do neurodesenvolvimento
07: Transtornos do sono-vigília
08: Doenças do sistema nervoso
09: Doenças do sistema visual
10: Doenças da orelha ou do processo mastoide
11: Doenças do sistema circulatório
12: Doenças do sistema respiratório
13: Doenças do sistema digestivo
14: Doenças da pele
15: Doenças do sistema musculoesquelético ou do tecido conjuntivo
16: Doenças do sistema geniturinário
17: Condições relacionadas à saúde sexual
18: Gravidez, parto ou puerpério
19: Algumas afecções originadas no período perinatal
20: Anomalias do desenvolvimento
21: Sintomas, sinais ou achados clínicos, não classificados em outra parte
22: Lesões, envenenamentos ou algumas outras consequências de causas externas
23: Causas externas de morbidade ou mortalidade
24: Fatores que influenciam o estado de saúde ou o contato com serviços de saúde
25: Códigos para propósitos especiais
26: Capítulo Suplementar – Condições da Medicina Tradicional
V: Seção suplementar para avaliação de funcionalidade
X: Códigos de extensão"""

FOCO_CLINICO = [
    "Anormalidade Adquirida", "Anormalidade Anatômica", "Anormalidade Congênita",
    "Achado", "Doença ou Síndrome", "Disfunção Celular ou Molecular",
    "Disfunção Mental ou Comportamental", "Fenômeno ou Processo",
    "Função Patológica", "Lesão ou Envenenamento", "Processo Neoplásico",
    "Sinal ou Sintoma"
]

def dividir_texto(texto, limite):
    return [texto[i:i+limite] for i in range(0, len(texto), limite)]

def chamar_llm(prompt):
    payload = {"model": MODELO, "prompt": prompt, "stream": False, "format": "json"}
    try:
        response = requests.post(OLLAMA_API, json=payload, timeout=120)
        return json.loads(response.json()['response'])
    except:
        return {}

def construir_prompt(chunk, referencias_validas):
    return f"""
Você é um Especialista em Codificação Médica da OMS, focado exclusivamente em extração NER para CID-11.
Sua missão é converter o texto médico em um JSON estruturado de alta precisão.

### TEXTO PARA ANÁLISE ###
"{chunk}"

### LISTA DE ENTIDADES JÁ CONHECIDAS (REFERÊNCIAS) ###
{json.dumps(referencias_validas, ensure_ascii=False)}

### GUIA DE CAPÍTULOS CID-11 ###
{DESC_CAPITULOS}

### REGRAS DE OURO (NÃO VIOLAR) ###
1. IDENTIFICAÇÃO: Extraia apenas patologias reais, sintomas ou lesões. 
   - IGNORE siglas de enfermagem, dispositivos ou locais (AVP, SNG, SVD, TOT, CC, D, MIE, MSD, SNE, RHA+, MV).
   - NÃO invente diagnósticos. Se o texto diz "borra de café", não escreva "Hemorragia Digestiva" a menos que o termo apareça ou esteja nas REFERÊNCIAS.

2. HIERARQUIA DE CLASSIFICAÇÃO:
   - INFECÇÃO sempre terá prioridade no Capítulo 01 (mesmo que seja no pulmão ou sangue).
   - NEOPLASIA/TUMOR sempre terá prioridade no Capítulo 02 (mesmo que seja no cérebro ou osso).
   - SINTOMAS INESPECÍFICOS (dor, febre, soluço, edema) vão para o Capítulo 21.
   - LESÕES POR CAUSA EXTERNA (fraturas, quedas, cortes) vão para o Capítulo 22.

3. ESTRUTURA DOS CAMPOS:
   - "term_original": Copie o termo exatamente como está no TEXTO.
   - "capitulo": Apenas o número ou letra (ex: "01", "11", "V").
   - "is_inferred": FALSE se o termo extraído for IGUAL a algum item da lista de REFERÊNCIAS. TRUE se o termo foi pescado do TEXTO mas não estava nas REFERÊNCIAS.
   - "confidence_embedding": Sempre 0.0.
   - "classification_reasoning": Sempre "".

4. FORMATO DE SAÍDA:
   - A chave principal deve ser o NÚMERO do capítulo. 
   - Se houver mais de uma doença para o mesmo capítulo, a LLM deve retornar chaves distintas (ex: "01", "01_bis"). O script tratará a unicidade depois.

### EXEMPLO DE SAÍDA ESPERADA ###
{{
  "01": {{
    "term_original": "SEPTICEMIA",
    "capitulo": "01",
    "confidence_embedding": 0.0,
    "is_inferred": false,
    "classification_reasoning": ""
  }},
  "11": {{
    "term_original": "HIPERTENSÃO",
    "capitulo": "11",
    "confidence_embedding": 0.0,
    "is_inferred": true,
    "classification_reasoning": ""
  }}
}}

Retorne APENAS o JSON.
"""

def processar():
    if not os.path.exists(PASTA_SAIDA): os.makedirs(PASTA_SAIDA)
    arquivos = [f for f in os.listdir(PASTA_ENTRADA) if f.endswith('.json')]

    for nome_arquivo in arquivos:
        print(f"Processando: {nome_arquivo}")
        with open(os.path.join(PASTA_ENTRADA, nome_arquivo), 'r', encoding='utf-8') as f:
            dados = json.load(f)

        referencias = [t for cat, termos in dados.get('entities', {}).items() if cat in FOCO_CLINICO for t in termos]
        chunks = dividir_texto(dados.get('text', ""), TAMANHO_CHUNK)
        
        labels_totais = {}
        contador_global = 1

        for pedaco in chunks:
            resultado_chunk = chamar_llm(construir_prompt(pedaco, referencias))
            
            # Validação: Garante que resultado_chunk é um dicionário
            if not isinstance(resultado_chunk, dict):
                continue

            for cap, info in resultado_chunk.items():
                # Validação: Garante que 'info' é um dicionário antes de atribuir
                if isinstance(info, dict):
                    cap_limpo = str(cap).replace("_", "").strip()
                    chave_final = f"{cap_limpo}_{contador_global}"
                    
                    info['capitulo'] = cap_limpo
                    labels_totais[chave_final] = info
                    contador_global += 1

        dados['labels'] = labels_totais
        with open(os.path.join(PASTA_SAIDA, nome_arquivo), 'w', encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    processar()