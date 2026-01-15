import json
import os
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

# --- CONFIGURAÇÕES ---
MODEL_ID = "google/medgemma-4b-it"
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

# --- CARREGAMENTO DO MODELO (UMA VEZ) ---
print("Carregando MedGemma localmente...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = PaliGemmaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print(f"✓ Modelo carregado na GPU: {torch.cuda.is_available()}")

def validar_vinculo_clinico_equilibrado(contexto, termo, codigo, descricao_cid, reasoning_original):
    """
    Usa MedGemma para validar codificação CID-11 clinicamente.
    """
    prompt = f"""Você é um Auditor Médico Especialista. Sua missão é garantir que a codificação CID-11 seja CLINICAMENTE COERENTE com o prontuário.

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
5. Não proponha, sugira ou infira nenhum novo código CID. Avalie exclusivamente se o código atribuído é clinicamente válido ou inválido.

Se decidir por REMOVER, use um destes motivos: {', '.join(LISTA_MOTIVOS_REPROVACAO)}

Responda APENAS em JSON válido (sem explicações adicionais):
{{
    "valido": true ou false,
    "motivo_tecnico": "nome_do_motivo_se_falso ou null",
    "analise_critica": "Explique brevemente sua lógica clínica para manter ou remover."
}}"""

    try:
        # Prepara input
        inputs = processor(text=prompt, return_tensors="pt")
        
        # Move para GPU se disponível
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Gera resposta
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.2,  # Baixa temperatura para respostas mais consistentes
                do_sample=False
            )
        
        # Decodifica
        response_text = processor.decode(output_ids[0], skip_special_tokens=True)
        
        # Extrai JSON da resposta
        # MedGemma pode incluir o prompt na resposta, então procura por "{"
        json_start = response_text.rfind('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            return json.loads(json_str)
        else:
            print(f"⚠ Resposta não contém JSON válido: {response_text[:100]}")
            return None
            
    except json.JSONDecodeError as e:
        print(f"⚠ Erro ao decodificar JSON: {e}")
        return None
    except Exception as e:
        print(f"⚠ Erro na inferência: {e}")
        return None

def processar_auditoria():
    """
    Processa todos os arquivos JSON e audita com MedGemma.
    """
    if not os.path.exists(PASTA_AUDITADA):
        os.makedirs(PASTA_AUDITADA)
    
    arquivos = [f for f in os.listdir(PASTA_ENTRADA) if f.endswith('.json')]
    total = len(arquivos)
    
    print(f"\n{'='*60}")
    print(f"Iniciando auditoria de {total} arquivos com MedGemma")
    print(f"{'='*60}\n")
    
    for i, nome_arquivo in enumerate(arquivos, 1):
        caminho_entrada = os.path.join(PASTA_ENTRADA, nome_arquivo)
        
        with open(caminho_entrada, 'r', encoding='utf-8') as f:
            dados = json.load(f)
        
        contexto = dados.get("text", "")
        labels_atuais = dados.get("labels", {})
        novos_labels = {}
        
        print(f"[{i}/{total}] Auditando: {nome_arquivo}")
        
        for codigo, info in labels_atuais.items():
            resultado = validar_vinculo_clinico_equilibrado(
                contexto,
                info['term_original'],
                codigo,
                info['descricao_cid'],
                info['classification_reasoning']
            )
            
            if resultado:
                is_valido = resultado.get("valido")
                info["decisao"] = "MANTER" if is_valido else "REMOVER"
                info["motivo_decisao"] = resultado.get("analise_critica")
                info["tipo_erro_auditoria"] = resultado.get("motivo_tecnico") if not is_valido else None
                
                # Debug: mostra decisão
                status = "✓ MANTER" if is_valido else "✗ REMOVER"
                print(f"   {status} | {codigo}: {info['term_original']}")
            else:
                info["decisao"] = "INDETERMINADO"
                info["motivo_decisao"] = "Falha na inferência"
                info["tipo_erro_auditoria"] = None
                print(f"   ⚠ INDETERMINADO | {codigo}: {info['term_original']}")
            
            novos_labels[codigo] = info
        
        dados["labels"] = novos_labels
        
        # Salva resultado auditado
        caminho_saida = os.path.join(PASTA_AUDITADA, nome_arquivo)
        with open(caminho_saida, 'w', encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)
        
        progress = (i / total * 100)
        print(f"   Progresso: {progress:.1f}% | Salvo em: {caminho_saida}\n")
    
    print(f"\n{'='*60}")
    print(f"✓ Auditoria completa! Arquivos salvos em: {PASTA_AUDITADA}")
    print(f"{'='*60}")

if __name__ == "__main__":
    processar_auditoria()
