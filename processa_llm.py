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
        response = requests.post(OLLAMA_API, json=payload, timeout=90)
        return json.loads(response.json()['response'])
    except:
        return {}

def construir_prompt(texto, entidades):
    # Criamos uma lista simples das entidades já anotadas para a LLM comparar
    lista_entidades_existentes = []
    for categoria in entidades.values():
        lista_entidades_existentes.extend(categoria)

    return f"""
Aja como um médico auditor. Analise o prontuário e, caso possua, classifique a condição principal.
TEXTO DO PRONTUÁRIO: "{texto}"
ANOTAÇÕES EXISTENTES (SemClinBR): {json.dumps(lista_entidades_existentes, ensure_ascii=False)}

CAPÍTULOS CID-11:
{DESC_CAPITULOS}

INSTRUÇÕES DE PREENCHIMENTO:
1. "term_original": O termo clínico identificado.
2. "is_inferred": 
   - FALSE: Se o termo que você escolheu está na lista de 'ANOTAÇÕES EXISTENTES'.
   - TRUE: Se você extraiu esse termo diretamente do 'TEXTO DO PRONTUÁRIO' mas ele não estava nas anotações.
3. "is_diagnosis": 
   - TRUE: Se você concluiu um diagnóstico que NÃO está escrito explicitamente (ex: o texto diz 'dor no peito e troponina alta' e você previu 'Infarto').
   - FALSE: Se o diagnóstico já está escrito claramente no texto.

Retorne EXCLUSIVAMENTE este JSON:
{{
  "": {{
    "term_original": "termo identificado",
    "capitulo": "número do capítulo",
    "confidence_embedding": 0.0,
    "is_inferred": boolean,
    "is_diagnosis": boolean,
    "classification_reasoning": "justificativa médica"
  }}
}}
"""

def processar():
    if not os.path.exists(PASTA_SAIDA):
        os.makedirs(PASTA_SAIDA)

    arquivos = [f for f in os.listdir(PASTA_ENTRADA) if f.endswith('.json')]
    print(f"Processando {len(arquivos)} arquivos...")

    for nome_arquivo in arquivos:
        print(f"Auditando: {nome_arquivo}...", end='\r')
        with open(os.path.join(PASTA_ENTRADA, nome_arquivo), 'r', encoding='utf-8') as f:
            dados = json.load(f)

        prompt = construir_prompt(dados['text'], dados['entities'])
        dados['labels'] = chamar_llm(prompt)

        with open(os.path.join(PASTA_SAIDA, nome_arquivo), 'w', encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)

    print("\nConcluído!")

if __name__ == "__main__":
    processar()