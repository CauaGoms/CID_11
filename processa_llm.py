import json
import os
import requests

# --- CONFIGURAÇÕES ---
OLLAMA_API = "http://localhost:11434/api/generate"
MODELO = "llama3.1:8b"
PASTA_ENTRADA = "processamento/pront_teste"
PASTA_SAIDA = "processamento/prontuarios_classificados"

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

def chamar_llm(prompt):
    payload = {
        "model": MODELO,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0  # Determinismo total para evitar alucinações
        }
    }
    try:
        response = requests.post(OLLAMA_API, json=payload, timeout=120)
        return json.loads(response.json()['response'])
    except:
        return None

def extrair_entidades_capitulo(texto, num_cap, desc_cap, entidades_candidatas):
    # Prompt focado em validação e não em invenção
    prompt = f"""
Atue como um perito em codificação CID-11.
OBJETIVO: Identificar quais entidades da lista 'CANDIDATOS' pertencem ao Capítulo {num_cap} ({desc_cap}).

TEXTO DE CONTEXTO: "{texto}"
CANDIDATOS: {entidades_candidatas}

REGRAS:
1. Analise cada termo em CANDIDATOS. 
2. Se o termo pertencer inequivocamente ao Capítulo {num_cap}, inclua-o na saída.
3. Se o termo NÃO pertencer ao Capítulo {num_cap}, ignore-o.
4. NUNCA invente termos que não estão na lista de CANDIDATOS.
5. Se nenhum termo servir, retorne uma lista vazia.

Retorne EXCLUSIVAMENTE um JSON: {{ "termos": ["TERMO1", "TERMO2"] }}
"""
    res = chamar_llm(prompt)
    return res.get("termos", []) if res else []

def processar():
    if not os.path.exists(PASTA_SAIDA): os.makedirs(PASTA_SAIDA)
    
    for nome_arquivo in os.listdir(PASTA_ENTRADA):
        if not nome_arquivo.endswith('.json'): continue
        print(f"-> Processando: {nome_arquivo}")
        
        with open(os.path.join(PASTA_ENTRADA, nome_arquivo), 'r', encoding='utf-8') as f:
            dados = json.load(f)
        
        texto = dados.get('text', "")
        
        # Filtro de categorias (Reduz o ruído antes de enviar)
        candidatos = []
        referencias_totais = []
        for cat, termos in dados.get('entities', {}).items():
            referencias_totais.extend([t.upper() for t in termos])
            if cat in CATEGORIAS_CLINICAS:
                candidatos.extend([t.upper() for t in termos])
        
        candidatos = list(set(candidatos)) # Remove duplicatas
        labels_finais = {}

        for num, desc in CAPITULOS_CID.items():
            if not candidatos: break
            
            print(f"   Checando Cap {num}...")
            encontrados = extrair_entidades_capitulo(texto, num, desc, candidatos)
            
            for termo in encontrados:
                termo_up = termo.upper()
                # A lógica de inferência agora é feita pelo Python (mais confiável)
                # is_inferred é falso se o termo exato foi extraído pelo SemClinBR
                is_inferred = termo_up not in referencias_totais
                
                labels_finais[termo_up] = {
                    "capitulo": num,
                    "is_inferred": is_inferred
                }

        dados['labels'] = labels_finais
        with open(os.path.join(PASTA_SAIDA, nome_arquivo), 'w', encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    processar()