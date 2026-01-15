import json
import os
import requests

# --- CONFIGURAÇÕES ---
OLLAMA_API_CHAT = "http://localhost:11434/api/chat"
LLM_MODEL = "llama3" 
PASTA_BUSCA = "processamento/busca_embedding/prontuarios_busca"
PASTA_FINAL = "processamento/escolha_cid/prontuarios_cid"

def refinar_com_llm(contexto, termo, capitulo, opcoes):
    """
    Envia as opções para a LLM escolher a melhor baseada no contexto clínico.
    """
    lista_opcoes_texto = ""
    for opcao in opcoes:
        for codigo, detalhes in opcao.items():
            lista_opcoes_texto += f"- Código: {codigo} | Descrição: {detalhes['text']}\n"

    prompt = f"""
    Você é um especialista em codificação médica CID-11.
    CONTEXTO DO PRONTUÁRIO: "{contexto}"
    TERMO EXTRAÍDO: "{termo}"
    CAPÍTULO CID: {capitulo}

    Abaixo estão as opções encontradas pela busca vetorial. Escolha a MAIS PRECISA para o termo dentro do contexto clínico.
    Ignore o score de confiança anterior.

    OPÇÕES:
    {lista_opcoes_texto}

    Responda EXCLUSIVAMENTE em formato JSON seguindo este modelo:
    {{
        "codigo": "O código escolhido",
        "reasoning": "Breve explicação clínica da escolha"
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
        resultado = response.json().get("message", {}).get("content")
        return json.loads(resultado)
    except Exception as e:
        print(f"      Erro na LLM para o termo '{termo}': {e}")
        return None

def processar_refinamento_final():
    if not os.path.exists(PASTA_FINAL): os.makedirs(PASTA_FINAL)
    
    arquivos = [f for f in os.listdir(PASTA_BUSCA) if f.endswith('.json')]
    total = len(arquivos)

    for i, nome_arquivo in enumerate(arquivos, 1):
        print(f"[{i}/{total}] Refinando: {nome_arquivo}...")
        
        with open(os.path.join(PASTA_BUSCA, nome_arquivo), 'r', encoding='utf-8') as f:
            dados = json.load(f)

        contexto = dados.get("text", "")
        labels_antigos = dados.get("labels", {})
        novos_labels = {}

        for termo, info in labels_antigos.items():
            opcoes = info.get("opcoes", [])
            cap = info.get("capitulo")
            
            if not opcoes: continue

            print(f"   -> Refinando escolha para: {termo}")
            decisao = refinar_com_llm(contexto, termo, cap, opcoes)

            if decisao:
                codigo_eleito = decisao.get("codigo")
                
                # Variáveis para armazenar os dados capturados da lista original de opções
                conf_original = 0.0
                descricao_encontrada = ""

                # Busca os dados originais (score e texto) para o código que a LLM escolheu
                for opt in opcoes:
                    if codigo_eleito in opt:
                        conf_original = opt[codigo_eleito].get("confidence_embedding")
                        descricao_encontrada = opt[codigo_eleito].get("text")
                        break

                # Monta a estrutura final conforme solicitado
                novos_labels[codigo_eleito] = {
                    "term_original": termo,
                    "capitulo": cap,
                    "confidence_embedding": conf_original,
                    "descricao_cid": descricao_encontrada,
                    "classification_reasoning": decisao.get("reasoning")
                }

        # Atualiza a estrutura do JSON
        dados["labels"] = novos_labels
        
        # Salva o resultado final
        with open(os.path.join(PASTA_FINAL, nome_arquivo), 'w', encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)

        percentual_falta = 100 - ((i / total) * 100)
        print(f"   Falta: {percentual_falta:.1f}% para concluir todos os arquivos.\n")

if __name__ == "__main__":
    processar_refinamento_final()