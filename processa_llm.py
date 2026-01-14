import json
import os
import requests

# --- CONFIGURAÇÕES ---
OLLAMA_API = "http://localhost:11434/api/generate"
MODELO = "llama3.1:8b"
PASTA_ENTRADA = "processamento/pront_teste"
PASTA_SAIDA = "processamento/prontuarios_classificados"

# Categorias clínicas para filtrar ruído (Abreviações, Dispositivos, etc. ficam de fora)
CATEGORIAS_CLINICAS = [
    "Doença ou Síndrome", "Sinal ou Sintoma", "Lesão ou Envenenamento", 
    "Achado", "Processo Neoplásico", "Disfunção Mental ou Comportamental", 
    "Anormalidade Congênita", "Anormalidade Anatômica", "Anormalidade Adquirida",
    "Bactéria", "Vírus", "Fungo", "Função Patológica", "Disfunção Celular ou Molecular"
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
    payload = {
        "model": MODELO,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0}
    }
    try:
        response = requests.post(OLLAMA_API, json=payload, timeout=120)
        return json.loads(response.json()['response'])
    except:
        return {}

def classificar_entidades(texto, candidatos):
    if not candidatos: return {}
    
    prompt = f"""
Você é um Especialista em CID-11. Atribua o capítulo correto para cada entidade da lista abaixo.
Contexto do Prontuário: "{texto}"

LISTA DE ENTIDADES PARA CLASSIFICAR:
{candidatos}

CAPÍTULOS CID-11 DISPONÍVEIS:
{CAPITULOS_CID}

REGRAS:
1. Responda apenas para as entidades da LISTA.
2. Use apenas o código do capítulo (ex: "01", "11", "V", "X").
3. Se a entidade não for uma doença/lesão/sintoma real, use "IGNORAR".

Retorne EXCLUSIVAMENTE um JSON:
{{ "NOME_DA_ENTIDADE": "CODIGO_DO_CAPITULO" }}
"""
    return chamar_llm(prompt)

def processar():
    if not os.path.exists(PASTA_SAIDA): os.makedirs(PASTA_SAIDA)
    
    for nome_arquivo in os.listdir(PASTA_ENTRADA):
        if not nome_arquivo.endswith('.json'): continue
        print(f"Lendo {nome_arquivo}...")
        
        with open(os.path.join(PASTA_ENTRADA, nome_arquivo), 'r', encoding='utf-8') as f:
            dados = json.load(f)
        
        texto = dados.get('text', "")
        
        # 1. Filtro de Candidatos (SemClinBR) e Referências Totais
        candidatos = []
        referencias_totais = []
        for cat, termos in dados.get('entities', {}).items():
            # Todas as entidades originais entram aqui para checar is_inferred depois
            referencias_totais.extend([t.upper() for t in termos])
            # Apenas as clínicas vão para a LLM classificar
            if cat in CATEGORIAS_CLINICAS:
                candidatos.extend([t.upper() for t in termos])
        
        candidatos = list(set(candidatos)) # Remover duplicatas
        
        # 2. Classificação em Bloco (1 chamada por prontuário)
        print(f"Classificando {len(candidatos)} entidades...")
        classificacoes = classificar_entidades(texto, candidatos)
        
        # 3. Formatação Final
        labels_finais = {}
        if isinstance(classificacoes, dict):
            for termo, cap in classificacoes.items():
                termo_up = termo.upper()
                if cap == "IGNORAR": continue
                
                # O is_inferred é FALSE se o termo exato foi extraído pelo SemClinBR
                is_inferred = termo_up not in referencias_totais
                
                labels_finais[termo_up] = {
                    "capitulo": str(cap),
                    "is_inferred": is_inferred
                }

        dados['labels'] = labels_finais
        
        with open(os.path.join(PASTA_SAIDA, nome_arquivo), 'w', encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)
        print(f"Salvo: {nome_arquivo}\n")

if __name__ == "__main__":
    processar()