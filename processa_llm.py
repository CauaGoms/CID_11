import json
import os
import requests

# --- CONFIGURAÇÕES ---
OLLAMA_API = "http://localhost:11434/api/generate"
MODELO = "llama3.1:8b"
PASTA_ENTRADA = "processamento/pront_teste"
PASTA_SAIDA = "processamento/prontuarios_classificados"

DESC_CAPITULOS = """
01_ Algumas doenças infecciosas ou parasitárias: Doenças causadas por agentes infecciosos como bactérias, vírus, parasitas e fungos.
02_ Neoplasias: Crescimento celular anormal e descontrolado, não compatível com as necessidades normais do organismo.
03_ Doenças do sangue ou dos órgãos formadores do sangue: Condições que afetam o sangue e os órgãos responsáveis por sua produção.
04_ Doenças do sistema imune: não há descrição complementar.
05_ Doenças endócrinas, nutricionais ou metabólicas: Distúrbios hormonais, nutricionais e metabólicos que afetam o funcionamento do organismo.
06_ Transtornos mentais, comportamentais ou do neurodesenvolvimento: Alterações clinicamente significativas da cognição, emoção ou comportamento, com impacto funcional.
07_ Transtornos de sono-vigília: Distúrbios relacionados à iniciação, manutenção ou regulação do sono e do ciclo sono-vigília.
08_ Doenças do sistema nervoso: Condições que afetam ou estão associadas ao sistema nervoso central ou periférico.
09_ Doenças do sistema visual: Doenças que acometem os olhos, vias visuais e áreas cerebrais relacionadas à visão.
10_ Doenças da orelha ou do processo mastoide: Condições que afetam a audição, o equilíbrio e estruturas associadas.
11_ Doenças do sistema circulatório: Doenças que afetam o coração, os vasos sanguíneos e o transporte de substâncias no organismo.
12_ Doenças do sistema respiratório: não há descrição complementar.
13_ Doenças do sistema digestivo: não há descrição complementar.
14_ Doenças da pele: Condições que afetam a pele, seus anexos, mucosas associadas e tecidos subjacentes.
15_ Doenças do sistema musculoesquelético ou do tecido conjuntivo: Doenças que afetam músculos, ossos, articulações e tecidos de sustentação.
16_ Doenças do sistema geniturinário: Condições que afetam os sistemas urinário e genital.
17_ Condições relacionadas à saúde sexual: não há descrição complementar.
18_ Gravidez, parto ou puerpério: Condições associadas à gestação, ao parto e ao período pós-parto.
19_ Algumas afecções originadas no período perinatal: Condições que têm origem no período perinatal, mesmo quando manifestadas posteriormente.
20_ Anomalias do desenvolvimento: Alterações estruturais ou funcionais decorrentes de falhas no desenvolvimento pré-natal.
21_ Sintomas, sinais ou achados clínicos, não classificados em outra parte: Sinais, sintomas e achados inespecíficos usados quando não há diagnóstico definitivo.
22_ Lesões, envenenamentos ou algumas outras consequências de causas externas: Danos corporais resultantes de agentes físicos, químicos ou da falta de elementos vitais.
23_ Causas externas de morbidade ou mortalidade: Classificação das circunstâncias e intenções que levam a lesões ou morte.
24_ Fatores que influenciam o estado de saúde ou o contato com serviços de saúde: Situações ou condições que afetam a saúde sem constituírem doença ou lesão.
25_ Códigos para propósitos especiais: não há descrição complementar.
26_ Capítulo Suplementar – Condições da Medicina Tradicional: não há descrição complementar.
V _ Seção suplementar para avaliação de funcionalidade: Instrumentos e categorias para avaliar e quantificar a funcionalidade e a incapacidade.
X _ Códigos de extensão: Códigos suplementares usados para detalhar ou complementar outras classificações, não utilizados como codificação primária."""

def chamar_llm(prompt):
    payload = {
        "model": MODELO, "prompt": prompt, "stream": False, "format": "json"
    }
    try:
        response = requests.post(OLLAMA_API, json=payload, timeout=120)
        return json.loads(response.json()['response'])
    except:
        return {}

def construir_prompt(texto, entidades):
    # Consolida entidades para referência
    lista_referencia = []
    for cat in entidades.values():
        lista_referencia.extend(cat)

    return f"""
Analise o prontuário médico abaixo para realizar a extração de entidades (NER) e classificação por capítulo CID-11.

TEXTO DO PRONTUÁRIO: "{texto}"
ANOTAÇÕES PRÉVIAS (SemClinBR): {json.dumps(lista_referencia, ensure_ascii=False)}

CAPÍTULOS CID-11:
{DESC_CAPITULOS}

INSTRUÇÕES:
1. Identifique TODAS as entidades clínicas (doenças, sintomas, lesões ou motivos de consulta). Pode haver múltiplas entidades em um único prontuário.
2. Para cada entidade, determine o capítulo CID-11 correspondente.
3. Defina "is_inferred": 
   - FALSE: se o termo utilizado for exatamente um dos presentes em 'ANOTAÇÕES PRÉVIAS'.
   - TRUE: se o termo foi extraído por você diretamente do texto original e não estava nas anotações.
4. ATENÇÃO: Os campos "confidence_embedding" deve ser 0.0 e "classification_reasoning" deve ser uma string vazia "".
5. Use o nome da entidade ou uma string vazia como chave do objeto principal, mas garanta que cada entidade seja um objeto distinto.

Retorne EXCLUSIVAMENTE um JSON onde cada chave é uma entidade identificada:
{{
  "NOME_DA_ENTIDADE_1": {{
    "term_original": "termo identificado",
    "capitulo": "número do capítulo",
    "confidence_embedding": 0.0,
    "is_inferred": boolean,
    "classification_reasoning": ""
  }},
  "NOME_DA_ENTIDADE_2": {{
    "term_original": "termo identificado",
    "capitulo": "número do capítulo",
    "confidence_embedding": 0.0,
    "is_inferred": boolean,
    "classification_reasoning": ""
  }}
}}
"""

def processar():
    if not os.path.exists(PASTA_SAIDA):
        os.makedirs(PASTA_SAIDA)

    arquivos = [f for f in os.listdir(PASTA_ENTRADA) if f.endswith('.json')]
    print(f"Iniciando extração NER em {len(arquivos)} arquivos...")

    for nome_arquivo in arquivos:
        print(f"Processando: {nome_arquivo}...")
        with open(os.path.join(PASTA_ENTRADA, nome_arquivo), 'r', encoding='utf-8') as f:
            dados = json.load(f)

        prompt = construir_prompt(dados['text'], dados['entities'])
        resultado = chamar_llm(prompt)
        
        # Inserimos o resultado no campo labels
        dados['labels'] = resultado

        with open(os.path.join(PASTA_SAIDA, nome_arquivo), 'w', encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)

    print("\nProcessamento concluído!")

if __name__ == "__main__":
    processar()