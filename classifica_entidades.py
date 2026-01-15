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
    "01": "Algumas doenças infecciosas ou parasitárias",
    "02": "Neoplasias",
    "03": "Doenças do sangue ou dos órgãos formadores do sangue",
    "04": "Doenças do sistema imune",
    "05": "Doenças endócrinas, nutricionais ou metabólicas",
    "06": "Transtornos mentais, comportamentais ou do neurodesenvolvimento",
    "07": "Transtornos de sono-vigília",
    "08": "Doenças do sistema nervoso",
    "09": "Doenças do sistema visual",
    "10": "Doenças da orelha ou do processo mastoide",
    "11": "Doenças do sistema circulatório",
    "12": "Doenças do sistema respiratório",
    "13": "Doenças do sistema digestivo",
    "14": "Doenças da pele",
    "15": "Doenças do sistema musculoesquelético ou do tecido conjuntivo",
    "16": "Doenças do sistema geniturinário",
    "17": "Condições relacionadas à saúde sexual",
    "18": "Gravidez, parto ou puerpério",
    "19": "Algumas afecções originadas no período perinatal",
    "20": "Anomalias do desenvolvimento",
    "21": "Sintomas, sinais ou achados clínicos, não classificados em outra parte",
    "22": "Lesões, envenenamentos ou algumas outras consequências de causas externas",
    "23": "Causas externas de morbidade ou mortalidade",
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
    prompt = f"""<bos>[INST] Você é um Auditor Médico. Sua tarefa é filtrar a lista de termos e manter APENAS aqueles que são CIDs (Classificação Estatística Internacional de Doenças e Problemas Relacionados à Saúde). Seu objetivo é classificar doenças, lesões, sintomas e causas de morte válidos conforme o texto e atribuir os capítulos dos quais pertencem.

TEXTO: "{texto}"
TERMOS: {", ".join(candidatos)}

REGRAS:
1. Para cada termo, verifique se ele de fato pode corresponder a uma CID com base no texto fornecido.
2. Se o termo corresponder a uma CID, atribua o capítulo correto (ex: "01", "14", "21") conforme a lista de capítulos fornecida a seguir:

CAPÍTULOS CID: {CAPITULOS_CID}

FORMATO DE SAÍDA:
(termo) -> (capítulo_CID)
[/INST]"""
    
    inputs = processor(text=prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        output = model_med.generate(
            **inputs, 
            max_new_tokens=512, 
            do_sample=True,
            temperature=0.2, 
            top_p=0.9,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    
    resposta = processor.decode(output[0], skip_special_tokens=True)
    # Remove o prompt da resposta para o Llama não se confundir
    return resposta.split("[/INST]")[-1].strip()

def chamar_llama_formatador(analise_medica, candidatos):
    prompt = f"""
Atue como um formatador de JSON técnico. Sua tarefa é extrair as classificações CID-11 da análise médica abaixo.

ANÁLISE MÉDICA DO ESPECIALISTA:
"{analise_medica}"

REGRAS OBRIGATÓRIAS:
1. Retorne um JSON PLANO onde a CHAVE é o termo e o VALOR é apenas o código do capítulo (ex: "01", "14", "21").
2. Se o código não foi mencionado, NÃO inclua o termo no JSON.
3. Não crie listas, não crie sub-objetos. Apenas {{"termo": "codigo"}}.
4. Não inclua termos que não estejam na lista de candidatos fornecida.
5. Exemplo de saída: {{"diabetes": "05", "dor de cabeça": "21"}}

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
            print(analise_texto)
            
            # 2. Llama 3.1 (Formatador): Organização técnica
            print(f"   → Llama 3.1 formatando JSON...")
            classificacoes = chamar_llama_formatador(analise_texto, candidatos)
            print(classificacoes)
            
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