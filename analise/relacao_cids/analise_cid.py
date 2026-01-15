import json
import os
from collections import Counter

def gerar_dashboard_bi_final(caminho_prontuarios, caminho_dicionario, arquivo_saida="dashboard_clinico_v3.html"):
    # 1. Carregar Dicionário de CIDs
    try:
        with open(caminho_dicionario, 'r', encoding='utf-8') as f:
            lista_dic = json.load(f)
        dic_nomes = {item['identificador']: item['valor'] for item in lista_dic}
    except Exception as e:
        print(f"Erro ao carregar dicionário: {e}")
        return

    # 2. Processar Dados
    stats = {}
    tipos_encontrados = set()
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
                    
                    # Coocorrência de CIDs com contagem
                    for outro in cids_mantidos:
                        if cid != outro: 
                            stats[cid]['rel_cids'][outro] += 1
                    
                    # Entidades por Tipo
                    for tipo_entidade, lista_termos in entidades_dict.items():
                        tipos_encontrados.add(tipo_entidade)
                        if tipo_entidade not in stats[cid]['entidades_por_tipo']:
                            stats[cid]['entidades_por_tipo'][tipo_entidade] = Counter()
                        for termo in lista_termos:
                            stats[cid]['entidades_por_tipo'][tipo_entidade][termo.upper()] += 1
        except: continue

    # 3. Gerar HTML
    html_template = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <title>BI Clínico Avançado</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background: #f0f2f5; margin: 0; display: flex; height: 100vh; }}
            .sidebar {{ width: 320px; background: #2c3e50; color: white; padding: 25px; overflow-y: auto; box-shadow: 2px 0 5px rgba(0,0,0,0.1); }}
            .main {{ flex: 1; padding: 30px; overflow-y: auto; scroll-behavior: smooth; }}
            .card {{ background: white; padding: 25px; border-radius: 12px; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 6px solid #3498db; transition: 0.3s; }}
            .card:hover {{ transform: translateY(-3px); box-shadow: 0 6px 12px rgba(0,0,0,0.1); }}
            .filter-group {{ margin-bottom: 25px; background: #34495e; padding: 15px; border-radius: 8px; }}
            .btn-group {{ display: flex; gap: 10px; margin-bottom: 15px; }}
            button {{ flex: 1; padding: 8px; border-radius: 4px; border: none; cursor: pointer; font-size: 0.8em; font-weight: bold; text-transform: uppercase; transition: 0.2s; }}
            .btn-all {{ background: #27ae60; color: white; }}
            .btn-none {{ background: #c0392b; color: white; }}
            button:hover {{ opacity: 0.8; }}
            .tag {{ display: inline-block; padding: 4px 10px; border-radius: 20px; font-size: 0.8em; margin: 3px; border: 1px solid; font-weight: 500; }}
            .tag-CID {{ background: #e8f6ff; border-color: #3498db; color: #2980b9; }}
            .ent-tag {{ background: #f8f9f9; border-color: #d5dbdb; color: #2c3e50; }}
            .hidden {{ display: none !important; }}
            .search-bar {{ width: 100%; padding: 12px; border-radius: 6px; border: none; margin-bottom: 20px; font-size: 14px; }}
            .section-label {{ font-size: 0.75em; text-transform: uppercase; color: #95a5a6; margin-top: 15px; display: block; letter-spacing: 1px; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="sidebar">
            <h2 style="margin-top:0;">🔎 Filtros BI</h2>
            <input type="text" id="txtSearch" class="search-bar" placeholder="Buscar Doença ou CID...">
            
            <div class="filter-group">
                <span style="display:block; margin-bottom:10px; font-weight:bold;">Categorias de Entidades</span>
                <div class="btn-group">
                    <button class="btn-all" onclick="toggleAll(true)">Marcar Todos</button>
                    <button class="btn-none" onclick="toggleAll(false)">Limpar</button>
                </div>
                <div id="typeFilters">
                    {"".join([f'<label style="display:block; margin:8px 0; font-size:0.9em;"><input type="checkbox" class="type-checkbox" value="{t}" checked onclick="applyFilters()"> {t}</label>' for t in sorted(tipos_encontrados)])}
                </div>
            </div>
        </div>

        <div class="main">
            <div id="content">
    """

    for cid, info in sorted(stats.items(), key=lambda x: x[1]['count'], reverse=True):
        nome_cid = dic_nomes.get(cid, "Descrição não disponível no dicionário")
        
        # Tags de CIDs Co-ocorrentes com (x)
        rel_cids_html = "".join([f'<span class="tag tag-CID">{dic_nomes.get(c, c)} ({f}x)</span>' for c, f in info['rel_cids'].most_common(8)])
        
        # Tags de Entidades por Categoria
        entidades_html = ""
        for tipo, termos in info['entidades_por_tipo'].items():
            for termo, freq in termos.most_common(15):
                entidades_html += f'<span class="tag ent-tag" data-type="{tipo}">{termo} <small style="color:#7f8c8d">({freq}x)</small></span>'

        html_template += f"""
                <div class="card" data-name="{nome_cid.upper()} {cid}">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:1.4em; font-weight:bold; color:#2c3e50;">{cid}</span>
                        <span style="background:#3498db; color:white; padding:4px 12px; border-radius:20px; font-size:0.85em;">{info['count']} Ocorrências</span>
                    </div>
                    <div style="font-size: 1.2em; color:#34495e; margin: 10px 0;">{nome_cid}</div>
                    
                    <span class="section-label">Comorbidades Frequentes (Co-ocorrência)</span>
                    <div style="margin-bottom:15px;">{rel_cids_html if rel_cids_html else "<i>Sem co-ocorrência</i>"}</div>

                    <span class="section-label">Evidências Clínicas (Filtráveis)</span>
                    <div class="ent-container">{entidades_html if entidades_html else "<i>Nenhuma evidência extraída</i>"}</div>
                </div>
        """

    html_template += """
            </div>
        </div>

        <script>
            function toggleAll(status) {
                document.querySelectorAll('.type-checkbox').forEach(cb => cb.checked = status);
                applyFilters();
            }

            function applyFilters() {
                const checkboxes = document.querySelectorAll('.type-checkbox');
                const activeTypes = Array.from(checkboxes).filter(i => i.checked).map(i => i.value);
                const searchTerm = document.getElementById('txtSearch').value.toUpperCase();

                // Filtrar Entidades
                document.querySelectorAll('.ent-tag').forEach(tag => {
                    const type = tag.getAttribute('data-type');
                    tag.classList.toggle('hidden', !activeTypes.includes(type));
                });

                // Filtrar Cards
                document.querySelectorAll('.card').forEach(card => {
                    const name = card.getAttribute('data-name');
                    card.style.display = name.includes(searchTerm) ? "" : "none";
                });
            }

            document.getElementById('txtSearch').addEventListener('keyup', applyFilters);
        </script>
    </body>
    </html>
    """

    with open(arquivo_saida, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"✅ Dashboard BI Finalizado: {arquivo_saida}")

# Execução
gerar_dashboard_bi_final("analise/prontuarios_auditados/", "codigos_cid/ICD-11-com-descricoes.json")