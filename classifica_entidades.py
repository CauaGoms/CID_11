import json
import os
import torch
import re
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

# --- CONFIGURAÇÕES ---
MODEL_ID = "google/medgemma-1.5-4b-it" # Recomendado usar a versão IT de texto se não for usar imagens
PASTA_ENTRADA = "semclinbr/prontuarios"
PASTA_SAIDA = "processamento_medgemma/classifica_entidades/prontuarios_classificados"

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

# --- CARREGAMENTO ---
print("Carregando MedGemma...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = PaliGemmaForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
)

def chamar_medgemma(prompt):
    try:
        inputs = processor(text=prompt, return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        
        # Pega apenas os tokens novos gerados
        generation = output[0][input_len:]
        response_text = processor.decode(generation, skip_special_tokens=True)
        
        # Busca o JSON dentro da string de resposta
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            return json.loads(match.group().replace("'", '"')) # Garante aspas duplas
        return {}
    except Exception as e:
        print(f"   ⚠ Erro na inferência: {e}")
        return {}

def classificar_entidades(texto, candidatos):
    if not candidatos: return {}
    
    # Exemplo real no prompt para o modelo não se perder
    exemplo_json = '{"pneumonia": "01", "febre": "21", "normal": "IGNORAR"}'
    
    prompt = f"""<bos>Você é um Codificador Médico Especialista. Classifique as ENTIDADES nos capítulos da CID-11 baseando-se no TEXTO.

TEXTO: "{texto}"
ENTIDADES: {json.dumps(candidatos, ensure_ascii=False)}
CAPÍTULOS: {json.dumps(CAPITULOS_CID, ensure_ascii=False)}

REGRAS:
1. Retorne APENAS um JSON.
2. Use "IGNORAR" para termos normais ou ausência de doença.
3. Formato esperado: {exemplo_json}

JSON: """

    return chamar_medgemma(prompt)

def processar():
    if not os.path.exists(PASTA_SAIDA): os.makedirs(PASTA_SAIDA)
    arquivos = [f for f in os.listdir(PASTA_ENTRADA) if f.endswith('.json')]
    
    for i, nome_arquivo in enumerate(arquivos, 1):
        print(f"[{i}/{len(arquivos)}] Processando: {nome_arquivo}")
        
        with open(os.path.join(PASTA_ENTRADA, nome_arquivo), 'r', encoding='utf-8') as f:
            dados = json.load(f)
        
        # Filtra candidatos
        candidatos = list(set([t.lower() for cat, termos in dados.get('entities', {}).items() 
                              if cat in CATEGORIAS_CLINICAS for t in termos]))
        
        if not candidatos:
            dados['labels'] = {}
        else:
            classificacoes = classificar_entidades(dados.get('text', ""), candidatos)
            
            labels_finais = {}
            for termo in candidatos:
                # Busca o termo na resposta (tratando case insensitive)
                cap = classificacoes.get(termo) or classificacoes.get(termo.capitalize())
                if cap and cap != "IGNORAR":
                    labels_finais[termo] = {"capitulo": str(cap)}
            
            dados['labels'] = labels_finais
            print(f"   ✓ {len(labels_finais)} labels geradas.")

        with open(os.path.join(PASTA_SAIDA, nome_arquivo), 'w', encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    processar()