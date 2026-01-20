import json
import os
import torch
import requests
from tqdm import tqdm
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
    prompt = f"""Você é um médico especialista em codificação diagnóstica e terminologia médica (CID-11).
Sua tarefa é cruzar os termos extraídos com o contexto clínico do prontuário para atribuir o capítulo correto.

### CONTEXTO DO PRONTUÁRIO
{texto_completo}

### LISTA DE TERMOS PARA ANÁLISE
{", ".join(candidatos)}

### REFERÊNCIA DE CAPÍTULOS CID-11
{referencia_cid}

### DIRETRIZES DE CLASSIFICAÇÃO:
1. **Prioridade Diagnóstica**: Classifique apenas termos que representem entidades que se enquadrem em um capítulo específico do CID-11.
2. **Diferenciação por Contexto**: Se um termo puder pertencer a dois capítulos, use o contexto do prontuário. (Ex: "Diabetes" em gestante pode ser Capítulo 18 em vez de 05).

### FORMATO DE SAÍDA:
Responda EXCLUSIVAMENTE com a lista no formato 'termo -> código'. Não escreva introduções ou conclusões.
Se nenhum termo for classificável, retorne apenas: "Nenhum termo enquadrável".

Resposta:
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
    prompt = f"""Atue como um conversor de texto médico para JSON estruturado.
Sua única tarefa é ler a ANÁLISE MÉDICA e extrair as associações de CID-11.

### ANÁLISE MÉDICA PARA PROCESSAR:
{analise_medica}

### REGRAS TÉCNICAS OBRIGATÓRIAS:
1. **Formato de Saída**: Retorne EXCLUSIVAMENTE um objeto JSON plano.
2. **Estrutura**: {{"termo": "código"}}.
3. **Limpeza de Chaves**: O "termo" deve ser exatamente como aparece na análise, mas em letras minúsculas.
4. **Limpeza de Valores**: O "código" deve conter apenas os caracteres do capítulo (ex: "01", "11", "V", "X"). Remova pontos ou espaços extras.
5. **Zero Alucinação**: Se um termo na análise não tiver um código associado ou se a análise disser "Nenhum termo enquadrável", ignore-o.
6. **Integridade**: Não adicione explicações, introduções ou blocos de código markdown (```json). Retorne apenas o texto do JSON.

### EXEMPLO DE SAÍDA ESPERADA:
{{"cefaleia": "08", "epilepsia": "08", "gastrite": "13"}}

Se não houver dados válidos, retorne apenas: {{}}

JSON:"""

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
    if not os.path.exists(PASTA_SAIDA): 
        os.makedirs(PASTA_SAIDA)
    
    arquivos = [f for f in os.listdir(PASTA_ENTRADA) if f.endswith('.json')]
    
    # Criamos a barra de progresso principal para os arquivos
    pbar = tqdm(arquivos, desc="Processando Prontuários", unit="arq")
    
    for nome_arquivo in pbar:
        # Atualiza a descrição da barra com o nome do arquivo atual
        pbar.set_postfix_str(f"Arquivo: {nome_arquivo}")
        
        caminho_entrada = os.path.join(PASTA_ENTRADA, nome_arquivo)
        with open(caminho_entrada, 'r', encoding='utf-8') as f:
            dados = json.load(f)
        
        texto_completo = dados.get('text', "")
        candidatos = list(set([t.lower() for lista in dados.get('entities', {}).values() for t in lista]))
        
        labels_finais = {}
        
        if candidatos:
            # 1. MedGemma analisa
            analise = chamar_medgemma_especialista(texto_completo, candidatos)
            
            # 2. Llama formata
            json_classificado = chamar_llama_formatador(analise)
            
            # 3. Integração e contagem
            if isinstance(json_classificado, dict):
                for termo, cap in json_classificado.items():
                    term_key = termo.lower()
                    # Garante que só incluímos o que o MedGemma realmente classificou
                    if term_key in candidatos:
                        labels_finais[term_key] = {"capitulo": str(cap).upper()}
            
            # Log no console acima da barra para não quebrá-la
            tqdm.write(f"  [√] {nome_arquivo}: {len(labels_finais)}/{len(candidatos)} entidades classificadas.")
        
        dados['labels'] = labels_finais

        # Salva o resultado
        caminho_saida = os.path.join(PASTA_SAIDA, nome_arquivo)
        with open(caminho_saida, 'w', encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)

    print("\n✓ Processamento concluído com sucesso!")

if __name__ == "__main__":
    processar()