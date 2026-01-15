import json
import os
from collections import Counter
from fpdf import FPDF
from fpdf.enums import XPos, YPos

class ClinicalPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(44, 62, 80)
        self.cell(0, 10, "Relatorio BI Clinico - Analise de Patologias", 
                  border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Pagina {self.page_no()}", 
                  border=0, new_x=XPos.RIGHT, new_y=YPos.TOP, align="C")

def limpar_texto(texto):
    """Trata caracteres especiais para evitar erro de codificação no PDF"""
    if not texto: return ""
    # Substitui traços longos e garante latin-1
    t = str(texto).replace('\u2013', '-').replace('\u2014', '-')
    return t.encode('latin-1', 'replace').decode('latin-1')

def gerar_dashboard_bi_final(caminho_prontuarios, caminho_dicionario, base_nome="dashboard_clinico"):
    # --- 1. CARREGAR DICIONÁRIO ---
    try:
        with open(caminho_dicionario, 'r', encoding='utf-8') as f:
            lista_dic = json.load(f)
        dic_nomes = {item['identificador']: item['valor'] for item in lista_dic}
    except Exception as e:
        print(f"Erro ao carregar dicionario: {e}")
        return

    # --- 2. PROCESSAR DADOS ---
    stats = {}
    arquivos = [f for f in os.listdir(caminho_prontuarios) if f.endswith('.json')]
    
    for nome_arquivo in arquivos:
        try:
            with open(os.path.join(caminho_prontuarios, nome_arquivo), 'r', encoding='utf-8') as f:
                dados = json.load(f)
                labels = dados.get("labels", {})
                entidades_dict = dados.get("entities", {})
                cids_mantidos = [cid for cid, info in labels.items() if info.get("decisao") == "MANTER"]

                for cid in cids_mantidos:
                    if cid not in stats:
                        stats[cid] = {'count': 0, 'rel_cids': Counter(), 'entidades_por_tipo': {}}
                    stats[cid]['count'] += 1
                    for outro in cids_mantidos:
                        if cid != outro: 
                            stats[cid]['rel_cids'][outro] += 1
                    
                    for tipo_entidade, lista_termos in entidades_dict.items():
                        if tipo_entidade not in stats[cid]['entidades_por_tipo']:
                            stats[cid]['entidades_por_tipo'][tipo_entidade] = Counter()
                        for termo in lista_termos:
                            stats[cid]['entidades_por_tipo'][tipo_entidade][termo.upper()] += 1
        except: continue

    # --- 3. GERAR PDF ---
    pdf = ClinicalPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Ordenar por CIDs mais frequentes
    for cid, info in sorted(stats.items(), key=lambda x: x[1]['count'], reverse=True):
        nome_cid = limpar_texto(dic_nomes.get(cid, "Descricao nao disponivel"))
        cid_limpo = limpar_texto(cid)
        
        # Bloco de Título (CID Principal)
        pdf.set_fill_color(232, 246, 255)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(150, 10, f" CID: {cid_limpo} - {nome_cid[:55]}", border=1, fill=True)
        
        pdf.set_fill_color(52, 152, 219)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(40, 10, f"{info['count']} Casos", border=1, fill=True, 
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        pdf.set_text_color(0, 0, 0)

        # SEÇÃO: COMORBIDADES (CIDs CORRELATOS COM CONTAGEM)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 7, "COMORBIDADES FREQUENTES (CID + QTD CO-OCORRENCIA):", 
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 8)
        
        # Aqui montamos a string com a quantidade (f) de vezes que apareceu
        rel_data = []
        for c, f in info['rel_cids'].most_common(6):
            nome_rel = limpar_texto(dic_nomes.get(c, c))
            rel_data.append(f"{nome_rel} ({c}) [{f}x]")
        
        rel_text = ", ".join(rel_data)
        pdf.multi_cell(0, 5, rel_text if rel_text else "Nenhuma co-ocorrencia registrada.", 
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        # SEÇÃO: EVIDÊNCIAS CLÍNICAS
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 7, "EVIDENCIAS CLINICAS EXTRAIDAS:", 
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(44, 62, 80)
        
        entidades_resumo = ""
        for tipo, termos in info['entidades_por_tipo'].items():
            top_termos = [f"{limpar_texto(t)}({f}x)" for t, f in termos.most_common(5)]
            if top_termos:
                entidades_resumo += f"[{tipo}]: {', '.join(top_termos)}\n"
        
        pdf.set_font("Helvetica", "", 8)
        pdf.multi_cell(0, 4, entidades_resumo if entidades_resumo else "Sem evidencias.", 
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(5)

    pdf.output(f"{base_nome}.pdf")
    print(f"✅ PDF Atualizado com Sucesso: {base_nome}.pdf")

# Execução
gerar_dashboard_bi_final("analise/prontuarios_auditados/", "codigos_cid/ICD-11-com-descricoes.json")