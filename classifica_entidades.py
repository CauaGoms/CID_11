import json
import os
import torch
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- CONFIGURAÇÕES ---
MODELO_MEDGEMMA = "google/medgemma-1.5-4b-it"
OLLAMA_API = "http://localhost:11434/api/generate"
MODELO_LLAMA = "llama3.1:8b"

PASTA_ENTRADA = "semclinbr/prontuarios"
PASTA_SAIDA = "processamento_medgemma/classifica_entidades/prontuarios_classificados"

# Lista completa e detalhada para o MedGemma ter contexto total
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

# --- CARREGAMENTO DO MODELO ---
tokenizer = AutoTokenizer.from_pretrained(MODELO_MEDGEMMA)
model_med = AutoModelForCausalLM.from_pretrained(
    MODELO_MEDGEMMA,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

def chamar_medgemma_especialista(texto_completo, candidatos):
    """
    Usa o contexto total do prontuário para classificar os termos.
    """
    # Monta a lista de referência completa dos capítulos
    referencia_cid = "\n".join([f"Capítulo {k}: {v}" for k, v in CAPITULOS_CID.items()])
    
    # Prompt sem restrições severas de tamanho
    prompt = f"""Você é um médico especialista em codificação diagnóstica.
Analise o PRONTUÁRIO CLÍNICO abaixo para entender o contexto de cada termo mencionado.

--- PRONTUÁRIO ---
{texto_completo}
--- FIM DO PRONTUÁRIO ---

LISTA DE TERMOS PARA CLASSIFICAÇÃO:
{", ".join(candidatos)}

REFERÊNCIA DE CAPÍTULOS CID-11:
{referencia_cid}

INSTRUÇÃO: Caso um termo da lista acima seja uma entidade que se enquadre em um dos capítulos listados, identifique o código do Capítulo CID-11. Caso contrário, ignore o termo. Responda estritamente no formato: termo -> código
"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model_med.generate(
            **inputs, 
            max_new_tokens=1024, # Aumentado para suportar listas longas de termos
            temperature=0.1,
            do_sample=False
        )
    
    resposta_completa = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extrai apenas a parte gerada após o prompt
    return resposta_completa[len(prompt):].strip()

def chamar_llama_formatador(analise_medica):
    """
    Llama via Ollama para limpeza e extração de JSON.
    """
    prompt = f"""Extraia as relações 'termo -> código' da análise abaixo e formate como um JSON plano.
    
ANÁLISE:
{analise_medica}

REGRAS:
- Chave: o termo original.
- Valor: o código (ex: "05", "11", "V").
- Retorne APENAS o JSON puro. No markdown, sem explicações.
- Não invente códigos ou termos que não estejam na análise.
- Se não houver termos válidos, retorne um JSON vazio: {{}}"""

    payload = {
        "model": MODELO_LLAMA,
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

def processar():
    if not os.path.exists(PASTA_SAIDA): os.makedirs(PASTA_SAIDA)
    
    arquivos = [f for f in os.listdir(PASTA_ENTRADA) if f.endswith('.json')]
    
    for nome_arquivo in arquivos:
        print(f"-> Processando {nome_arquivo}...")
        
        with open(os.path.join(PASTA_ENTRADA, nome_arquivo), 'r', encoding='utf-8') as f:
            dados = json.load(f)
        
        texto_completo = dados.get('text', "")
        # Coleta todos os termos de entidades identificadas
        candidatos = list(set([t.lower() for lista in dados.get('entities', {}).values() for t in lista]))
        
        if candidatos:
            # 1. MedGemma analisa com contexto total
            analise = chamar_medgemma_especialista(texto_completo, candidatos)
            
            # 2. Llama formata o resultado
            json_classificado = chamar_llama_formatador(analise)
            
            # 3. Integração
            labels = {}
            for termo, cap in json_classificado.items():
                labels[termo.lower()] = {"capitulo": str(cap).upper()}
            
            dados['labels'] = labels
            
        with open(os.path.join(PASTA_SAIDA, nome_arquivo), 'w', encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    processar()