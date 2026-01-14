import json
import os
import requests

# --- CONFIGURAÇÕES ---
OLLAMA_API_CHAT = "http://localhost:11434/api/chat"
LLM_MODEL = "llama3" 
PASTA_ENTRADA = "processamento/escolha_cid/prontuarios_cid"
PASTA_AUDITADA = "processamento/seleciona_labels/prontuarios_auditados"

# Lista de motivos para a LLM escolher em caso de REMOVER
LISTA_MOTIVOS_REPROVACAO = [
    "false_positive_entity", "missing_entity", "overly_generic_entity", 
    "wrong_entity_type", "abbreviation_misclassification", "negated_finding",
    "hypothetical_finding", "family_history_misread", "temporal_misinterpretation",
    "anatomical_mismatch", "symptom_vs_diagnosis_confusion", "procedure_vs_diagnosis_confusion",
    "medication_vs_substance_confusion", "semantic_drift_normalization", "wrong_cid_granularity",
    "cid_overgeneralization", "cid_over_specificity", "context_ignorance",
    "section_misinterpretation", "duplicated_entity", "hallucinated_linking", "confidence_overestimation"
]

def validar_vinculo_clinico(contexto, termo, codigo, descricao_cid, reasoning_original):
    """
    Solicita à LLM que valide o vínculo e, se inválido, classifique o erro.
    """
    
    prompt = f"""
    Você é um auditor médico especialista em CID-11. Sua tarefa é validar se a codificação de um termo extraído de um prontuário está correta.

    CONTEÚDO DO PRONTUÁRIO: "{contexto}"
    TERMO EXTRAÍDO: "{termo}"
    CÓDIGO CID-11 ATRIBUÍDO: "{codigo}"
    DESCRIÇÃO DO CÓDIGO: "{descricao_cid}"
    JUSTIFICATIVA DO SISTEMA: "{reasoning_original}"

    CRITÉRIOS DE VALIDAÇÃO:
    1. O código realmente descreve o termo no contexto clínico?
    2. A associação é forçada, alucinada ou ignora negações/hipóteses?

    Se a sua decisão for REMOVER (valido: false), você DEVE escolher EXATAMENTE um motivo desta lista:
    {LISTA_MOTIVOS_REPROVACAO}

    Responda EXCLUSIVAMENTE em formato JSON:
    {{
        "valido": true ou false,
        "motivo_tecnico": "O nome do item da lista acima se for falso, ou 'n/a' se for verdadeiro",
        "analise_critica": "explicação detalhada da sua decisão"
    }}
    """

    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json"
    }

    try:
        response = requests.post(OLLAMA_API_CHAT, json=payload, timeout=90)
        content = response.json().get("message", {}).get("content")
        return json.loads(content)
    except Exception as e:
        print(f"      Erro na auditoria do termo '{termo}': {e}")
        return None

def processar_auditoria():
    if not os.path.exists(PASTA_AUDITADA): os.makedirs(PASTA_AUDITADA)
    
    arquivos = [f for f in os.listdir(PASTA_ENTRADA) if f.endswith('.json')]
    total = len(arquivos)

    for i, nome_arquivo in enumerate(arquivos, 1):
        print(f"[{i}/{total}] Auditando clinicamente: {nome_arquivo}...")
        
        with open(os.path.join(PASTA_ENTRADA, nome_arquivo), 'r', encoding='utf-8') as f:
            dados = json.load(f)

        contexto = dados.get("text", "")
        labels_atuais = dados.get("labels", {})
        novos_labels = {}

        for codigo, info in labels_atuais.items():
            termo = info.get("term_original")
            descricao = info.get("descricao_cid")
            reasoning_original = info.get("classification_reasoning")

            print(f"   -> Auditando: {termo} ({codigo})")
            resultado = validar_vinculo_clinico(contexto, termo, codigo, descricao, reasoning_original)

            if resultado:
                is_valido = resultado.get("valido")
                decisao = "MANTER" if is_valido else "REMOVER"
                
                # Adicionamos os campos solicitados
                info["decisao"] = decisao
                info["motivo_decisao"] = resultado.get("analise_critica")
                
                # Se for removido, grava a categoria técnica do erro
                if not is_valido:
                    info["tipo_erro_auditoria"] = resultado.get("motivo_tecnico")
                else:
                    info["tipo_erro_auditoria"] = None

                novos_labels[codigo] = info
                print(f"      [{decisao}] Motivo técnico: {resultado.get('motivo_tecnico')}")
            else:
                info["decisao"] = "ERRO_PROCESSAMENTO"
                info["tipo_erro_auditoria"] = "api_error"
                novos_labels[codigo] = info

        dados["labels"] = novos_labels
        
        with open(os.path.join(PASTA_AUDITADA, nome_arquivo), 'w', encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)

        percentual_falta = 100 - ((i / total) * 100)
        print(f"   Concluído. Falta: {percentual_falta:.1f}%\n")

if __name__ == "__main__":
    processar_auditoria()