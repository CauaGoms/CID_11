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
Você é um AUDITOR MÉDICO EXTREMAMENTE CÉTICO. 
Assuma que a maioria das codificações CID-11 está incorreta até que se prove o contrário.

PRONTUÁRIO REAL:
\"\"\"{contexto}\"\"\"

TERMO EXTRAÍDO:
\"{termo}\"

CÓDIGO CID-11 ATRIBUÍDO:
\"{codigo}\"

DESCRIÇÃO OFICIAL DO CÓDIGO CID-11:
\"\"\"{descricao_cid}\"\"\"

JUSTIFICATIVA DO SISTEMA ANTERIOR:
\"\"\"{reasoning_original}\"\"\"

MISSÃO:
Determine se EXISTE PROVA TÉCNICA SUFICIENTE para manter esse código CID-11.
Se não conseguir demonstrar claramente a adequação do código à definição CID-11,
o código DEVE ser removido.

REGRAS DE INVALIDAÇÃO (qualquer uma é suficiente):
1. O termo é genérico, administrativo ou descritivo e não representa entidade CID.
2. O código exige critérios (anatômicos, temporais, etiológicos ou psicogênicos)
   que NÃO estão presentes explicitamente no prontuário.
3. O vínculo entre termo e código depende de inferência clínica não demonstrada.
4. A anatomia do código não corresponde claramente à anatomia mencionada.
5. O termo representa sintoma, estado, achado ou contexto — não diagnóstico CID.

REGRAS DE VALIDAÇÃO (pelo menos UMA deve ser provada explicitamente):
A. O termo é a própria denominação canônica do código CID-11.
B. A definição CID-11 é diretamente satisfeita pelo texto do prontuário.
C. O código é o mapeamento padrão inequívoco usado na prática clínica para esse termo.

IMPORTANTE:
- Diagnósticos escritos pelo médico NÃO são automaticamente válidos.
- Na dúvida, REMOVA.
- Nunca presuma condições não descritas.

Se INVALIDAR, escolha exatamente UM motivo técnico da lista:
{LISTA_MOTIVOS_REPROVACAO}

RESPONDA EXCLUSIVAMENTE EM JSON:
{{
  "valido": true ou false,
  "motivo_tecnico": "nome_do_motivo_ou_null",
  "prova_ou_falha": "Explique qual regra de validação foi provada OU por que nenhuma foi satisfeita."
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