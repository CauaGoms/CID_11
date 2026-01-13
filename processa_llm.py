import json
import os
import requests

# --- CONFIGURAÇÕES ---
OLLAMA_API = "http://localhost:11434/api/generate"
MODELO = "llama3.1:8b"
PASTA_ENTRADA = "processamento/pront_teste"
PASTA_SAIDA = "processamento/prontuarios_classificados"
TAMANHO_CHUNK = 2000 

# Descrição simplificada conforme solicitado (apenas o número)
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
Atue como um codificador médico especialista em CID-11. 
Analise este PEDAÇO de prontuário e extraia condições clínicas.

TEXTO: "{chunk}"
REFERÊNCIAS: {json.dumps(referencias_validas, ensure_ascii=False)}

CAPÍTULOS (Use apenas o número/letra antes dos dois pontos):
{DESC_CAPITULOS}

REGRAS:
1. Extraia APENAS doenças, sintomas ou lesões. 
2. IGNORE siglas (AVP, SNG, SVD, TOT, CC, D, MIE, MSD).
3. "is_inferred": FALSE se o termo estiver nas 'REFERÊNCIAS', TRUE caso contrário.
4. "confidence_embedding": 0.0 constante.
5. "classification_reasoning": "" constante.
6. A CHAVE do JSON deve ser apenas o número do capítulo (ex: "01", "11", "V").

Retorne EXCLUSIVAMENTE JSON:
{{ "NUMERO": {{ "term_original": "...", "capitulo": "apenas o numero", "confidence_embedding": 0.0, "is_inferred": bool, "classification_reasoning": "" }} }}
"""

def processar():
    if not os.path.exists(PASTA_SAIDA): os.makedirs(PASTA_SAIDA)
    arquivos = [f for f in os.listdir(PASTA_ENTRADA) if f.endswith('.json')]

    for nome_arquivo in arquivos:
        print(f"Processando: {nome_arquivo}")
        with open(os.path.join(PASTA_ENTRADA, nome_arquivo), 'r', encoding='utf-8') as f:
            dados = json.load(f)

        referencias = [t for cat, termos in dados['entities'].items() if cat in FOCO_CLINICO for t in termos]
        chunks = dividir_texto(dados['text'], TAMANHO_CHUNK)
        
        labels_totais = {}
        contador_global = 1

        for pedaco in chunks:
            resultado_chunk = chamar_llm(construir_prompt(pedaco, referencias))
            
            for cap, info in resultado_chunk.items():
                # Limpa o capítulo de qualquer "_" ou caracteres extras caso a LLM ignore a regra
                cap_limpo = cap.replace("_", "").strip()
                # Cria chave única para não sobrescrever (ex: 11_1, 11_2)
                chave_final = f"{cap_limpo}_{contador_global}"
                
                # Garante que o campo 'capitulo' interno também esteja limpo
                info['capitulo'] = cap_limpo
                
                labels_totais[chave_final] = info
                contador_global += 1

        dados['labels'] = labels_totais
        with open(os.path.join(PASTA_SAIDA, nome_arquivo), 'w', encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    processar()