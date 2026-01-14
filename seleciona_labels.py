import json
import os
import requests

# --- CONFIGURAÇÕES ---
OLLAMA_API_CHAT = "http://localhost:11434/api/chat"
LLM_MODEL = "llama3" 
PASTA_ENTRADA = "processamento/escolha_cid/prontuarios_cid"
PASTA_AUDITADA = "processamento/seleciona_labels/prontuarios_auditados"

def validar_vinculo_clinico(contexto, termo, codigo, descricao_cid, reasoning_original):
    """
    Solicita à LLM que valide se o código CID realmente faz sentido para o termo
    dentro daquele prontuário específico.
    """
    
    prompt = f"""
    Você é um auditor médico especialista em CID-11. Sua tarefa é validar se a codificação atribuída a um termo extraído de um prontuário está correta ou se é um erro/alucinação.

    CONTEÚDO DO PRONTUÁRIO: "{contexto}"
    TERMO EXTRAÍDO: "{termo}"
    CÓDIGO CID-11 ATRIBUÍDO: "{codigo}"
    DESCRIÇÃO DO CÓDIGO: "{descricao_cid}"
    JUSTIFICATIVA DO SISTEMA: "{reasoning_original}"

    CRITÉRIOS DE VALIDAÇÃO:
    1. O código CID-11 realmente descreve o termo no contexto clínico apresentado?
    2. A associação é forçada ou absurda?
    3. Se o termo for genérico demais e o CID específico demais sem base, deve-se remover.

    Responda EXCLUSIVAMENTE em formato JSON:
    {{
        "valido": true ou false,
        "analise_critica": "explicação detalhada da sua decisão de manter ou remover"
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
                # Se for válido, marcamos como MANTER, caso contrário REMOVER
                decisao = "MANTER" if resultado.get("valido") else "REMOVER"
                motivo = resultado.get("analise_critica")
                
                # Adicionamos os novos campos ao dicionário da label
                info["decisao"] = decisao
                info["motivo_decisao"] = motivo
                
                # Opcional: Se quiser que o JSON final contenha APENAS os mantidos, 
                # descomente o 'if' abaixo. Caso queira todos para conferência, deixe assim.
                # if decisao == "MANTER":
                novos_labels[codigo] = info
                
                print(f"      [{decisao}] {motivo[:60]}...")
            else:
                # Caso a API falhe, marcamos como erro para revisão manual
                info["decisao"] = "ERRO_PROCESSAMENTO"
                info["motivo_decisao"] = "Não foi possível auditar via LLM."
                novos_labels[codigo] = info

        # Atualiza o campo labels com as novas informações de auditoria
        dados["labels"] = novos_labels
        
        # Salva o arquivo com os novos campos
        with open(os.path.join(PASTA_AUDITADA, nome_arquivo), 'w', encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)

        percentual_falta = 100 - ((i / total) * 100)
        print(f"   Concluído. Falta: {percentual_falta:.1f}%\n")

if __name__ == "__main__":
    processar_auditoria()