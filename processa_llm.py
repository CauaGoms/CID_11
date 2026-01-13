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

# Lista de categorias SemClinBR para filtrar o que é relevante para CID-11
FOCO_CLINICO = [
    "Anormalidade Adquirida", "Anormalidade Anatômica", "Anormalidade Congênita",
    "Achado", "Doença ou Síndrome", "Disfunção Celular ou Molecular",
    "Disfunção Mental ou Comportamental", "Fenômeno ou Processo",
    "Função Patológica", "Lesão ou Envenenamento", "Processo Neoplásico",
    "Sinal ou Sintoma"
]

def chamar_llm(prompt):
    payload = {
        "model": MODELO,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    try:
        response = requests.post(OLLAMA_API, json=payload, timeout=120)
        response.raise_for_status()
        return json.loads(response.json()['response'])
    except Exception as e:
        print(f"Erro na API Ollama: {e}")
        return {}

def construir_prompt(texto, entidades):
    # Filtragem das referências para focar apenas no que é clínico
    referencias_validas = []
    for categoria, termos in entidades.items():
        if categoria in FOCO_CLINICO:
            referencias_validas.extend(termos)

    return f"""
Atue como um codificador médico especialista em CID-11. Sua tarefa é extrair entidades clínicas e classificá-las estritamente pelos capítulos da CID-11.

TEXTO PARA ANÁLISE: "{texto}"
ENTIDADES DE REFERÊNCIA (SemClinBR): {json.dumps(referencias_validas, ensure_ascii=False)}

CAPÍTULOS CID-11 DISPONÍVEIS:
{DESC_CAPITULOS}

REGRAS DE EXTRAÇÃO E FORMATAÇÃO:
1. Extraia APENAS condições clínicas reais (doenças, sintomas ou lesões). 
3. Se houver mais de uma entidade para o mesmo capítulo, use chaves numeradas (ex: "11_1", "11_2").
4. "is_inferred": FALSE se o termo estiver nas 'ENTIDADES DE REFERÊNCIA'. TRUE se extraído apenas do texto bruto.
5. "confidence_embedding": DEVE ser sempre 0.0.
6. "classification_reasoning": DEVE ser sempre uma string vazia "".
7. NÃO invente doenças não mencionadas no texto.

Retorne EXCLUSIVAMENTE um JSON onde a CHAVE PRINCIPAL é o NÚMERO/LETRA DO CAPÍTULO:
{{
  "NUMERO_CAPITULO": {{
    "term_original": "termo exato como aparece no texto",
    "capitulo": "mesmo número da chave",
    "confidence_embedding": 0.0,
    "is_inferred": boolean,
    "classification_reasoning": ""
  }}
}}
"""

def processar_arquivos():
    if not os.path.exists(PASTA_SAIDA):
        os.makedirs(PASTA_SAIDA)

    arquivos = [f for f in os.listdir(PASTA_ENTRADA) if f.endswith('.json')]
    print(f"Iniciando classificação de {len(arquivos)} prontuários...")

    for nome_arquivo in arquivos:
        print(f"Processando: {nome_arquivo}")
        
        caminho_in = os.path.join(PASTA_ENTRADA, nome_arquivo)
        with open(caminho_in, 'r', encoding='utf-8') as f:
            dados = json.load(f)

        prompt = construir_prompt(dados['text'], dados['entities'])
        resultado_llm = chamar_llm(prompt)

        # Atualiza o campo labels com a extração da LLM
        dados['labels'] = resultado_llm

        caminho_out = os.path.join(PASTA_SAIDA, nome_arquivo)
        with open(caminho_out, 'w', encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)

    print("\nProcesso finalizado com sucesso!")

if __name__ == "__main__":
    processar_arquivos()