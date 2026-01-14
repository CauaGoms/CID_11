import json
import os
import requests

# --- CONFIGURAÇÕES ---
OLLAMA_API = "http://localhost:11434/api/generate"
MODELO = "llama3.1:8b"
PASTA_ENTRADA = "processamento/pront_teste"
PASTA_SAIDA = "processamento/prontuarios_classificados"

# Categorias do SemClinBR que realmente importam para a CID-11
CATEGORIAS_CLINICAS = [
    "Doença ou Síndrome", 
    "Sinal ou Sintoma", 
    "Lesão ou Envenenamento", 
    "Achado", 
    "Processo Neoplásico", 
    "Disfunção Mental ou Comportamental", 
    "Anormalidade Congênita",
    "Bactéria",
    "Vírus",
    "Fungo",
    "Anormalidade Anatômica",
    "Anormalidade Adquirida",
    "Função Patológica",
    "Disfunção Celular ou Molecular"
]

CAPITULOS_CID = {
    "01": "Algumas doenças infecciosas ou parasitárias: Doenças causadas por agentes infecciosos como bactérias, vírus, parasitas e fungos, transmitidas por contato direto, vetores, alimentos, água ou outras vias.",
    "02": "Neoplasias: Proliferação celular anormal e descontrolada, benigna ou maligna, que pode invadir tecidos adjacentes ou produzir metástases.",
    "03": "Doenças do sangue ou dos órgãos formadores do sangue: Condições que afetam o sangue, a coagulação e os órgãos hematopoéticos, como medula óssea e baço.",
    "04": "Doenças do sistema imune: Distúrbios do sistema imunológico, incluindo imunodeficiências, doenças autoimunes, inflamatórias e reações de hipersensibilidade.",
    "05": "Doenças endócrinas, nutricionais ou metabólicas: Distúrbios hormonais, nutricionais e metabólicos que afetam o crescimento, o metabolismo energético e a homeostase do organismo.",
    "06": "Transtornos mentais, comportamentais ou do neurodesenvolvimento: Alterações clinicamente significativas da cognição, regulação emocional ou comportamento, com impacto funcional pessoal, social ou ocupacional.",
    "07": "Transtornos de sono-vigília: Distúrbios relacionados à iniciação, manutenção ou regulação do sono, incluindo insônia, hipersonolência, parassonias e alterações do ritmo circadiano.",
    "08": "Doenças do sistema nervoso: Condições que afetam o sistema nervoso central, periférico ou autonômico, incluindo doenças neurológicas estruturais, degenerativas ou funcionais.",
    "09": "Doenças do sistema visual: Doenças que acometem os olhos, seus anexos, as vias visuais e áreas cerebrais responsáveis pela percepção visual.",
    "10": "Doenças da orelha ou do processo mastoide: Condições que afetam a audição, o equilíbrio e as estruturas do ouvido externo, médio, interno e mastoide.",
    "11": "Doenças do sistema circulatório: Doenças que afetam o coração, os vasos sanguíneos e a circulação sanguínea, comprometendo o transporte de oxigênio e nutrientes.",
    "12": "Doenças do sistema respiratório: Condições que afetam as vias aéreas, pulmões e músculos respiratórios, interferindo na ventilação e nas trocas gasosas.",
    "13": "Doenças do sistema digestivo: Doenças que afetam o trato gastrointestinal, fígado, vesícula biliar, pâncreas e estruturas associadas à digestão e absorção.",
    "14": "Doenças da pele: Condições que afetam a pele, seus anexos (cabelos, unhas e glândulas), mucosas associadas e tecidos subjacentes.",
    "15": "Doenças do sistema musculoesquelético ou do tecido conjuntivo: Doenças que afetam músculos, ossos, articulações, ligamentos, tendões e tecidos de sustentação.",
    "16": "Doenças do sistema geniturinário: Condições que afetam os sistemas urinário e genital, incluindo rins, vias urinárias e órgãos reprodutivos.",
    "17": "Condições relacionadas à saúde sexual: Condições associadas à função sexual, reprodução, identidade sexual e saúde sexual em geral, não classificadas em outros capítulos.",
    "18": "Gravidez, parto ou puerpério: Condições associadas à gestação, ao trabalho de parto, ao parto e ao período pós-parto imediato.",
    "19": "Algumas afecções originadas no período perinatal: Condições que têm origem no período perinatal, mesmo quando a morbidade ou mortalidade ocorre posteriormente.",
    "20": "Anomalias do desenvolvimento: Alterações estruturais ou funcionais decorrentes de falhas no desenvolvimento pré-natal de órgãos ou sistemas.",
    "21": "Sintomas, sinais ou achados clínicos, não classificados em outra parte: Sinais, sintomas e achados clínicos ou laboratoriais inespecíficos usados quando não há diagnóstico definitivo.",
    "22": "Lesões, envenenamentos ou algumas outras consequências de causas externas: Danos corporais decorrentes de agentes físicos, químicos ou da privação de elementos vitais, com início geralmente agudo.",
    "23": "Causas externas de morbidade ou mortalidade: Classificação das circunstâncias, eventos e intenções que resultam em lesões, envenenamentos ou morte.",
    "24": "Fatores que influenciam o estado de saúde ou o contato com serviços de saúde: Situações, condições ou motivos de contato com serviços de saúde que não constituem doença ou lesão.",
    "25": "Códigos para propósitos especiais: Códigos reservados para finalidades específicas, como vigilância epidemiológica, emergências de saúde pública ou usos administrativos.",
    "26": "Capítulo Suplementar – Condições da Medicina Tradicional: Condições, padrões diagnósticos e conceitos utilizados em sistemas de medicina tradicional reconhecidos pela OMS.",
    "V": "Seção suplementar para avaliação de funcionalidade: Instrumentos e categorias para descrever, medir e comparar níveis de funcionalidade e incapacidade.",
    "X": "Códigos de extensão: Códigos suplementares usados para detalhar características adicionais, contexto ou atributos de outras categorias, não utilizados como codificação primária."
}


def chamar_llm(prompt):
    payload = {"model": MODELO, "prompt": prompt, "stream": False, "format": "json"}
    try:
        response = requests.post(OLLAMA_API, json=payload, timeout=120)
        return json.loads(response.json()['response'])
    except:
        return None

# ETAPA 1: EXTRAÇÃO FILTRADA
def extrair_entidades_capitulo(texto, num_cap, desc_cap, referencias_filtradas):
    # O prompt agora recebe as referências já limpas para guiar a extração
    prompt = f"""
    Atue como médico codificador CID-11.
    Analise o TEXTO e extraia apenas condições clínicas (doenças, lesões ou sintomas) que pertençam ao Capítulo {num_cap} ({desc_cap}).
    
    TEXTO: "{texto}"
    DICAS DE REFERÊNCIA: {referencias_filtradas}
    
    REGRAS CRÍTICAS:
    1. IGNORE siglas de dispositivos, procedimentos ou locais (AVP, SNG, CC, SVD, VM, TOT).
    2. NÃO invente doenças não descritas.
    3. Foque em termos que combinem com a descrição do Capítulo {num_cap}.
    
    Retorne JSON: {{ "termos": ["TERMO1", "TERMO2"] }}
    Se nada for encontrado, retorne: {{ "termos": [] }}
    """
    res = chamar_llm(prompt)
    return res.get("termos", []) if res else []

# ETAPA 2: VALIDAÇÃO DE INFERÊNCIA
def verificar_inferencia(lista_termos, todas_referencias):
    if not lista_termos: return []
    prompt = f"""
    Dada a lista de termos extraídos: {lista_termos}
    E a lista de referências original: {todas_referencias}
    
    Para cada termo, determine:
    - is_inferred: false (se o termo ou sinônimo exato está nas referências)
    - is_inferred: true (se você o extraiu diretamente do texto bruto)
    
    Retorne JSON: {{ "validacao": [ {{ "termo": "...", "is_inferred": bool }}, ... ] }}
    """
    res = chamar_llm(prompt)
    return res.get("validacao", []) if res else []

def formatar_resultado_final(dados_validados, num_cap):
    resultado_parcial = {}
    for item in dados_validados:
        termo = item.get("termo", "").upper()
        if termo:
            resultado_parcial[termo] = {
                "capitulo": num_cap,
                "is_inferred": item.get("is_inferred", True)
            }
    return resultado_parcial

def processar():
    if not os.path.exists(PASTA_SAIDA): os.makedirs(PASTA_SAIDA)
    
    for nome_arquivo in os.listdir(PASTA_ENTRADA):
        if not nome_arquivo.endswith('.json'): continue
        print(f"-> Analisando: {nome_arquivo}")
        
        with open(os.path.join(PASTA_ENTRADA, nome_arquivo), 'r', encoding='utf-8') as f:
            dados = json.load(f)
        
        texto = dados.get('text', "")
        
        # --- FILTRO DE CATEGORIAS ---
        # Pegamos apenas o que é clinicamente relevante para evitar ruído de siglas
        referencias_limpas = []
        todas_referencias = []
        for cat, termos in dados.get('entities', {}).items():
            todas_referencias.extend(termos) # Para o is_inferred
            if cat in CATEGORIAS_CLINICAS:
                referencias_limpas.extend(termos)
        
        labels_completas = {}

        for num, desc in CAPITULOS_CID.items():
            print(f"   Verificando Cap {num}...")
            
            # Passamos as referências filtradas para o Especialista não se perder
            termos_brutos = extrair_entidades_capitulo(texto, num, desc, referencias_limpas)
            
            if termos_brutos:
                termos_validados = verificar_inferencia(termos_brutos, todas_referencias)
                labels_capitulo = formatar_resultado_final(termos_validados, num)
                labels_completas.update(labels_capitulo)

        dados['labels'] = labels_completas
        
        with open(os.path.join(PASTA_SAIDA, nome_arquivo), 'w', encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    processar()