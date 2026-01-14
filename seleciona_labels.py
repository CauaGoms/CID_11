import json
import os
import requests

# --- CONFIGURAÇÕES ---
OLLAMA_API_CHAT = "http://localhost:11434/api/chat"
LLM_MODEL = "llama3" 
PASTA_ENTRADA = "processamento/escolha_cid/prontuarios_cid"
PASTA_AUDITADA = "processamento/seleciona_labels/prontuarios_auditados"

LISTA_MOTIVOS_REPROVACAO = [
    "false_positive_entity", "missing_entity", "overly_generic_entity", 
    "wrong_entity_type", "abbreviation_misclassification", "negated_finding",
    "hypothetical_finding", "family_history_misread", "temporal_misinterpretation",
    "anatomical_mismatch", "symptom_vs_diagnosis_confusion", "procedure_vs_diagnosis_confusion",
    "medication_vs_substance_confusion", "semantic_drift_normalization", "wrong_cid_granularity",
    "cid_overgeneralization", "cid_over_specificity", "context_ignorance",
    "section_misinterpretation", "duplicated_entity", "hallucinated_linking", "confidence_overestimation"
]

def validar_vinculo_clinico_rigoroso(contexto, termo, codigo, descricao_cid, reasoning_original):
    # O prompt agora é focado em ENCONTRAR ERROS
    prompt = f"""
    Você é um AUDITOR MÉDICO CÉPTICO E RIGOROSO. Seu trabalho é encontrar falhas na codificação CID-11.
    Muitas classificações anteriores estão ERRADAS ou ALUCINADAS. Seja extremamente crítico.

    PRONTUÁRIO REAL: "{contexto}"
    TERMO QUE FOI EXTRAÍDO: "{termo}"
    CÓDIGO CID-11 QUE FOI ATRIBUÍDO: "{codigo}"
    DESCRIÇÃO OFICIAL DO CÓDIGO: "{descricao_cid}"
    JUSTIFICATIVA DO SISTEMA ANTERIOR: "{reasoning_original}"

    INSTRUÇÕES DE AUDITORIA:
    1. Se o termo no prontuário for uma NEGAÇÃO (ex: "sem febre") e o CID for a doença (ex: "Febre"), marque como 'negated_finding' e remova.
    2. Se o sistema anterior forçou a barra (ex: associou "chorosa" com "cocaína" sem prova direta), marque como 'hallucinated_linking' e remova.
    3. Se o código CID for específico demais para um termo genérico, remova.
    4. Analise se a anatomia mencionada no código condiz com a anatomia do prontuário.

    Sua decisão deve ser 'valido: false' a menos que a evidência seja INQUESTIONÁVEL.

    LISTA DE MOTIVOS TÉCNICOS (Escolha um se for falso):
    {LISTA_MOTIVOS_REPROVACAO}

    Responda EXCLUSIVAMENTE em formato JSON:
    {{
        "valido": true ou false,
        "motivo_tecnico": "nome_do_motivo",
        "analise_critica": "Desmonte a justificativa anterior se houver erro."
    }}
    """

    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0  # Remove a criatividade, foca na precisão
        }
    }

    try:
        response = requests.post(OLLAMA_API_CHAT, json=payload, timeout=90)
        content = response.json().get("message", {}).get("content")
        return json.loads(content)
    except Exception as e:
        return None

def processar_auditoria():
    if not os.path.exists(PASTA_AUDITADA): os.makedirs(PASTA_AUDITADA)
    arquivos = [f for f in os.listdir(PASTA_ENTRADA) if f.endswith('.json')]
    total = len(arquivos)

    for i, nome_arquivo in enumerate(arquivos, 1):
        with open(os.path.join(PASTA_ENTRADA, nome_arquivo), 'r', encoding='utf-8') as f:
            dados = json.load(f)

        contexto = dados.get("text", "")
        labels_atuais = dados.get("labels", {})
        novos_labels = {}

        print(f"[{i}/{total}] Auditando com RIGOR: {nome_arquivo}")

        for codigo, info in labels_atuais.items():
            resultado = validar_vinculo_clinico_rigoroso(
                contexto, info['term_original'], codigo, info['descricao_cid'], info['classification_reasoning']
            )

            if resultado:
                is_valido = resultado.get("valido")
                info["decisao"] = "MANTER" if is_valido else "REMOVER"
                info["motivo_decisao"] = resultado.get("analise_critica")
                info["tipo_erro_auditoria"] = resultado.get("motivo_tecnico") if not is_valido else None
                
                # Opcional: Remover do JSON final se for inválido
                # if is_valido: novos_labels[codigo] = info
                novos_labels[codigo] = info # Mantendo todos com a tag da decisão
            
        dados["labels"] = novos_labels
        with open(os.path.join(PASTA_AUDITADA, nome_arquivo), 'w', encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)

        print(f"   Falta carregar: {100 - (i/total*100):.1f}%")

if __name__ == "__main__":
    processar_auditoria()