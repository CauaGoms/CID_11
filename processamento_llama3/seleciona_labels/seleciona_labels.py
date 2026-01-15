import json
import os
import requests

# --- CONFIGURAÇÕES ---
OLLAMA_API_CHAT = "http://localhost:11434/api/chat"
LLM_MODEL = "llama3" 
PASTA_ENTRADA = "processamento/escolha_cid/prontuarios_vazios_cid"
PASTA_AUDITADA = "processamento/seleciona_labels/prontuarios_vazios_auditados"

LISTA_MOTIVOS_REPROVACAO = [
  "tipo_entidade_incorreto",
  "entidade_generica_demais",
  "inferencia_sem_suporte",
  "incompatibilidade_anatomica",
  "criterios_obrigatorios_ausentes",
  "achado_negado",
  "termo_nao_diagnostico",
  "contexto_alucinado"
]

def validar_vinculo_clinico_equilibrado(contexto, termo, codigo, descricao_cid, reasoning_original):
    # Prompt recalibrado para ser justo, mas atento a erros reais
    prompt = f"""
    Você é um Auditor Médico Especialista. Sua missão é garantir que a codificação CID-11 seja CLINICAMENTE COERENTE com o prontuário.

    CONTEÚDO DO PRONTUÁRIO: "{contexto}"
    TERMO EXTRAÍDO: "{termo}"
    CÓDIGO CID-11 ATRIBUÍDO: "{codigo}"
    DESCRIÇÃO DO CÓDIGO: "{descricao_cid}"
    JUSTIFICATIVA DO SISTEMA: "{reasoning_original}"

    DIRETRIZES DE AUDITORIA:
    1. MANTER se o código CID representa fielmente o termo ou uma condição clínica diretamente relacionada descrita no prontuário.
    2. REMOVER se houver contradição óbvia (ex: termo negado "sem dor" mapeado para "dor").
    3. REMOVER se a associação for puramente imaginária ou sem qualquer base no texto (alucinação).
    4. Seja justo: se o termo for uma abreviação médica comum ou sinônimo que faz sentido no contexto, aceite a classificação.
    5. Não proponha, sugira ou infira nenhum novo código CID. Avalie exclusivamente se o código atribuído é clinicamente válido ou inválido para o termo e o prontuário apresentados.

    Se decidir por REMOVER, use um destes motivos:
    {LISTA_MOTIVOS_REPROVACAO}

    Responda em JSON:
    {{
        "valido": true ou false,
        "motivo_tecnico": "nome_do_motivo_se_falso",
        "analise_critica": "Explique brevemente sua lógica clínica para manter ou remover."
    }}
    """

    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.2  # Calibração para bom senso clínico
        }
    }

    try:
        response = requests.post(OLLAMA_API_CHAT, json=payload, timeout=90)
        content = response.json().get("message", {}).get("content")
        return json.loads(content)
    except Exception:
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

        print(f"[{i}/{total}] Auditando (Calibrado): {nome_arquivo}")

        for codigo, info in labels_atuais.items():
            resultado = validar_vinculo_clinico_equilibrado(
                contexto, info['term_original'], codigo, info['descricao_cid'], info['classification_reasoning']
            )

            if resultado:
                is_valido = resultado.get("valido")
                info["decisao"] = "MANTER" if is_valido else "REMOVER"
                info["motivo_decisao"] = resultado.get("analise_critica")
                info["tipo_erro_auditoria"] = resultado.get("motivo_tecnico") if not is_valido else None
                novos_labels[codigo] = info
            
        dados["labels"] = novos_labels
        with open(os.path.join(PASTA_AUDITADA, nome_arquivo), 'w', encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)

        print(f"   Progresso: {(i/total*100):.1f}%")

if __name__ == "__main__":
    processar_auditoria()