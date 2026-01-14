import json
import os
import requests

# --- CONFIGURAÇÕES ---
OLLAMA_API = "http://localhost:11434/api/generate"
MODELO = "llama3.1:8b"
PASTA_ENTRADA = "processamento/pront_teste"
PASTA_SAIDA = "processamento/prontuarios_classificados"

# Lista de capítulos para o loop (Estratégia de Especialistas Individuais)
# Você pode adicionar todos os 26 aqui seguindo este padrão
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
        "format": "json" # Força a saída JSON do Ollama
    }
    try:
        response = requests.post(OLLAMA_API, json=payload, timeout=120)
        return json.loads(response.json()['response'])
    except Exception as e:
        return None

# ETAPA 1: EXTRAÇÃO (Um prompt por capítulo)
def extrair_entidades_capitulo(texto, num_cap, desc_cap):
    prompt = f"""
    Atue como médico codificador. Extraia do texto apenas termos que pertençam ao Capítulo {num_cap} ({desc_cap}) da CID-11.
    TEXTO: "{texto}"
    Retorne no formato JSON: {{ "termos": ["termo1", "termo2"] }}
    Se não houver nada, retorne: {{ "termos": [] }}
    """
    res = chamar_llm(prompt)
    return res.get("termos", []) if res else []

# ETAPA 2: VALIDAÇÃO DE INFERÊNCIA (Compara com SemClinBR)
def verificar_inferencia(lista_termos, referencias_semclinbr):
    if not lista_termos: return []
    prompt = f"""
    Compare estes termos extraídos: {lista_termos}
    Com as referências do SemClinBR: {referencias_semclinbr}
    
    Para cada termo, defina 'is_inferred' como false se ele estiver nas referências, ou true se não estiver.
    Retorne no formato JSON: {{ "validacao": [ {{ "termo": "...", "is_inferred": bool }}, ... ] }}
    """
    res = chamar_llm(prompt)
    return res.get("validacao", []) if res else []

# ETAPA 3: FORMATAÇÃO FINAL (Gera a estrutura desejada)
def formatar_resultado_final(dados_validados, num_cap):
    # Esta função organiza os dados no formato final: "Termo": { "capitulo": "X", "is_inferred": bool }
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
        print(f"-> Analisando arquivo: {nome_arquivo}")
        
        with open(os.path.join(PASTA_ENTRADA, nome_arquivo), 'r', encoding='utf-8') as f:
            dados = json.load(f)
        
        texto = dados.get('text', "")
        # Consolida todas as entidades de referência em uma lista simples
        referencias = [t for lista in dados.get('entities', {}).values() for t in lista]
        
        labels_completas = {}

        # Loop de Especialistas (Vários prompts independentes)
        for num, desc in CAPITULOS_CID.items():
            print(f"   Analista Cap {num} verificando...")
            
            # 1. Extrai
            termos_brutos = extrair_entidades_capitulo(texto, num, desc)
            
            if termos_brutos:
                # 2. Valida Inferencia
                termos_validados = verificar_inferencia(termos_brutos, referencias)
                
                # 3. Formata e Acumula
                labels_capitulo = formatar_resultado_final(termos_validados, num)
                labels_completas.update(labels_capitulo)

        dados['labels'] = labels_completas
        
        with open(os.path.join(PASTA_SAIDA, nome_arquivo), 'w', encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)
        print(f"Finalizado: {nome_arquivo}\n")

if __name__ == "__main__":
    processar()