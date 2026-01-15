import json
import os
import torch
import requests
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

# --- CONFIGURAÇÕES ---
# MedGemma local para o raciocínio médico
MODELO_MEDGEMMA = "google/medgemma-1.5-4b-it" 
# Llama via Ollama para a formatação JSON
OLLAMA_API = "http://localhost:11434/api/generate"
MODELO_LLAMA = "llama3.1:8b"

PASTA_ENTRADA = "semclinbr/prontuarios"
PASTA_SAIDA = "processamento_medgemma/classifica_entidades/prontuarios_classificados"

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

# --- CARREGAMENTO DO MEDGEMMA ---
print("Carregando MedGemma como especialista clínico...")
processor = AutoProcessor.from_pretrained(MODELO_MEDGEMMA)
model_med = PaliGemmaForConditionalGeneration.from_pretrained(
    MODELO_MEDGEMMA, torch_dtype=torch.bfloat16, device_map="auto"
)

def chamar_medgemma_especialista(texto, candidatos):
    prompt = f"""<bos>[INST] Você é um Auditor Médico Especialista em CID-11.
Sua missão é validar se os termos extraídos são achados clínicos codificáveis e atribuir o capítulo correto da CID-11.

TEXTO DO PRONTUÁRIO:
"{texto}"

ENTIDADES AVALIADAS:
{", ".join(candidatos)}

### TAREFA:
1. Avalie se a entidade representa uma patologia, sintoma ou achado clínico real no contexto deste prontuário.
2. Se for um CID válido, identifique o capítulo (01 a 26, V ou X).

### CRITÉRIOS DE CAPÍTULO:
{CAPITULOS_CID}

### FORMATO DE RESPOSTA:
Entidade: [Justificativa clínica] -> Decisão: [CÓDIGO DO CAPÍTULO ou IGNORAR]

RESPOSTA DE AUDITORIA: [/INST]"""
    
    inputs = processor(text=prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        output = model_med.generate(**inputs, max_new_tokens=512, do_sample=False)
    
    return processor.decode(output[0][input_len:], skip_special_tokens=True)

def chamar_llama_formatador(analise_medica, candidatos):
    prompt = f"""
Atue como um formatador de JSON técnico. Sua tarefa é extrair as classificações CID-11 da análise médica abaixo.

ANÁLISE MÉDICA DO ESPECIALISTA:
"{analise_medica}"

LISTA DE TERMOS ORIGINAIS:
{candidatos}

REGRAS OBRIGATÓRIAS:
1. Retorne um JSON PLANO onde a CHAVE é o termo e o VALOR é apenas o código do capítulo (ex: "01", "14", "21").
2. Se o médico indicou "IGNORAR" ou se o código não foi mencionado, NÃO inclua o termo no JSON.
3. Não crie listas, não crie sub-objetos. Apenas {{"termo": "codigo"}}.
4. Exemplo de saída: {{"diabetes": "05", "dor de cabeça": "21"}}

RETORNE APENAS O JSON:"""
    
    payload = {
        "model": MODELO_LLAMA,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0}
    }
    try:
        response = requests.post(OLLAMA_API, json=payload, timeout=120)
        # O Ollama com format:json garante que a string seja um JSON válido
        return json.loads(response.json()['response'])
    except Exception as e:
        print(f"      ⚠ Erro ao formatar com Llama: {e}")
        return {}

def processar():
    if not os.path.exists(PASTA_SAIDA): os.makedirs(PASTA_SAIDA)
    arquivos = [f for f in os.listdir(PASTA_ENTRADA) if f.endswith('.json')]
    total_arquivos = len(arquivos)
    
    if total_arquivos == 0: return print("Nenhum arquivo encontrado.")

    for i, nome_arquivo in enumerate(arquivos, 1):
        print(f"[{i}/{total_arquivos}] -> {nome_arquivo}")
        
        with open(os.path.join(PASTA_ENTRADA, nome_arquivo), 'r', encoding='utf-8') as f:
            dados = json.load(f)
        
        texto = dados.get('text', "")
        candidatos = list(set([t.lower() for termos in dados.get('entities', {}).values() for t in termos]))
        
        if candidatos:
            # 1. MedGemma (Cérebro): Decisão clínica
            print(f"   → MedGemma decidindo capítulos...")
            analise_texto = chamar_medgemma_especialista(texto, candidatos)
            
            # 2. Llama 3.1 (Formatador): Organização técnica
            print(f"   → Llama 3.1 formatando JSON...")
            classificacoes = chamar_llama_formatador(analise_texto, candidatos)
            
            labels_finais = {}
            if isinstance(classificacoes, dict):
                for termo, cap in classificacoes.items():
                    # Limpeza extra: só aceita se cap for código ou letra (V, X)
                    cap_str = str(cap).strip().upper()
                    if cap_str and cap_str != "IGNORAR" and cap_str != "NONE":
                        labels_finais[termo.lower()] = {"capitulo": cap_str}
            
            dados['labels'] = labels_finais
            print(f"   ✓ {len(labels_finais)} termos processados.")
        else:
            dados['labels'] = {}

        with open(os.path.join(PASTA_SAIDA, nome_arquivo), 'w', encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    processar()