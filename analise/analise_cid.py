import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# --- Configurações de Estilo para Artigos Científicos ---
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")

class ClinicalAuditAnalyzer:
    def __init__(self, input_path, output_dir='outputs'):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.tables_dir = self.output_dir / "tables"
        self.figures_dir = self.output_dir / "figures"
        
        # Criar estrutura de pastas
        for folder in [self.tables_dir, self.figures_dir]:
            folder.mkdir(parents=True, exist_ok=True)
            
        self.df_main = pd.DataFrame()

    def load_data(self):
        """1. Carrega e normaliza os arquivos JSON em um DataFrame tabular."""
        all_records = []
        
        json_files = list(self.input_path.glob('*.json'))
        if not json_files:
            print(f"Erro: Nenhum arquivo JSON encontrado em {self.input_path}")
            return

        for file_path in json_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                prontuario_id = data.get('prontuario_id')
                entities = data.get('entities', {})
                labels = data.get('labels', {})
                
                # Lista de todos os tipos de entidades presentes neste prontuário
                entity_types_in_doc = list(entities.keys())
                
                for cid_code, details in labels.items():
                    record = {
                        'prontuario_id': prontuario_id,
                        'cid_code': cid_code,
                        'term_original': details.get('term_original'),
                        'capitulo': details.get('capitulo'),
                        'confidence_embedding': details.get('confidence_embedding'),
                        'decisao': details.get('decisao'),
                        'tipo_erro_auditoria': details.get('tipo_erro_auditoria'),
                        'entity_types': entity_types_in_doc # Lista de tipos associados ao doc
                    }
                    all_records.append(record)
                    
        self.df_main = pd.DataFrame(all_records)
        # Tratamento de nulos
        self.df_main['tipo_erro_auditoria'] = self.df_main['tipo_erro_auditoria'].fillna('N/A')
        self.df_main['confidence_embedding'] = pd.to_numeric(self.df_main['confidence_embedding'], errors='coerce')
        
        print(f"Sucesso: {len(self.df_main)} registros processados de {len(json_files)} arquivos.")

    def analyze_cid_entities(self):
        """2. Análise de tipos de entidades associadas a cada código CID."""
        # Explodir a lista de entidades para que cada uma ganhe uma linha
        df_exploded = self.df_main.explode('entity_types')
        
        analysis = df_exploded.groupby(['cid_code', 'entity_types']).size().reset_index(name='frequencia')
        total_per_cid = analysis.groupby('cid_code')['frequencia'].transform('sum')
        analysis['proporcao'] = analysis['frequencia'] / total_per_cid
        
        analysis.to_csv(self.tables_dir / "cid_entity_relation.csv", index=False)
        return analysis

    def analyze_clinical_decisions(self):
        """3. Calcula métricas de decisão (Manter/Remover) por tipo de entidade."""
        df_exploded = self.df_main.explode('entity_types')
        
        metrics = df_exploded.groupby('entity_types').agg(
            total_codigos=('decisao', 'count'),
            manter=('decisao', lambda x: (x == 'MANTER').sum()),
            remover=('decisao', lambda x: (x == 'REMOVER').sum())
        )
        
        metrics['perc_manter'] = (metrics['manter'] / metrics['total_codigos']) * 100
        metrics['perc_remover'] = (metrics['remover'] / metrics['total_codigos']) * 100
        metrics['taxa_erro'] = metrics['perc_remover']
        
        metrics.sort_values('taxa_erro', ascending=False).to_csv(self.tables_dir / "entity_decision_metrics.csv")
        self.plot_error_rate(metrics)
        return metrics

    def analyze_audit_errors(self):
        """4. Matriz de Entidade vs Tipo de Erro (apenas para REMOVER)."""
        df_remover = self.df_main[self.df_main['decisao'] == 'REMOVER'].explode('entity_types')
        
        matrix = pd.crosstab(df_remover['entity_types'], df_remover['tipo_erro_auditoria'])
        matrix.to_csv(self.tables_dir / "entity_error_matrix.csv")
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='YlGnBu')
        plt.title('Matriz: Tipo de Entidade vs Tipo de Erro de Auditoria')
        plt.tight_layout()
        plt.savefig(self.figures_dir / "heatmap_errors.png", dpi=300)
        plt.close()

    def analyze_semantic_confidence(self):
        """5. Análise estatística de confiança semântica entre decisões."""
        manter_scores = self.df_main[self.df_main['decisao'] == 'MANTER']['confidence_embedding'].dropna()
        remover_scores = self.df_main[self.df_main['decisao'] == 'REMOVER']['confidence_embedding'].dropna()
        
        stats_desc = self.df_main.groupby('decisao')['confidence_embedding'].describe()
        stats_desc.to_csv(self.tables_dir / "confidence_statistics.csv")
        
        # Teste de Mann-Whitney (indicado para distribuições que podem não ser normais)
        u_stat, p_val = stats.mannwhitneyu(manter_scores, remover_scores)
        
        with open(self.tables_dir / "statistical_test.txt", "w") as f:
            f.write(f"Teste de Mann-Whitney U\nEstatística U: {u_stat}\np-valor: {p_val}\n")
            f.write("H0: Não há diferença entre as distribuições de confiança.\n")
            f.write("Resultado: " + ("Significativo" if p_val < 0.05 else "Não Significativo"))

        plt.figure(figsize=(8, 6))
        sns.boxplot(x='decisao', y='confidence_embedding', data=self.df_main, palette='Set2')
        plt.title('Confiança de Embedding por Decisão Clínica')
        plt.savefig(self.figures_dir / "boxplot_confidence.png", dpi=300)
        plt.close()

    def analyze_icd_chapters(self):
        """6. Análise por Capítulo CID-11."""
        chapter_stats = self.df_main.groupby('capitulo').agg(
            total_codigos=('cid_code', 'count'),
            taxa_erro=('decisao', lambda x: (x == 'REMOVER').mean() * 100)
        ).reset_index()
        
        chapter_stats.to_csv(self.tables_dir / "chapter_analysis.csv", index=False)
        
        # Entidade mais frequente por capítulo
        df_exploded = self.df_main.explode('entity_types')
        top_entities = df_exploded.groupby(['capitulo', 'entity_types']).size().reset_index(name='count')
        top_entities = top_entities.sort_values(['capitulo', 'count'], ascending=[True, False]).drop_duplicates('capitulo')
        top_entities.to_csv(self.tables_dir / "top_entities_per_chapter.csv", index=False)

    def plot_error_rate(self, metrics):
        """7. Visualização da taxa de erro por entidade."""
        plt.figure(figsize=(10, 6))
        metrics = metrics.sort_values('taxa_erro', ascending=True)
        plt.barh(metrics.index, metrics['taxa_erro'], color='salmon')
        plt.xlabel('Taxa de Erro (%)')
        plt.ylabel('Tipo de Entidade')
        plt.title('Taxa de Auditoria (REMOVER) por Tipo de Entidade')
        plt.tight_layout()
        plt.savefig(self.figures_dir / "error_rate_by_entity.png", dpi=300)
        plt.close()

    def run_pipeline(self):
        """Executa todas as etapas da análise."""
        print("--- Iniciando Análise de Auditoria Clínica ---")
        self.load_data()
        if self.df_main.empty: return
        
        self.analyze_cid_entities()
        self.analyze_clinical_decisions()
        self.analyze_audit_errors()
        self.analyze_semantic_confidence()
        self.analyze_icd_chapters()
        
        print(f"--- Processo concluído. Resultados salvos em: {self.output_dir} ---")

# --- Exemplo de Execução ---
if __name__ == "__main__":
    # Substitua 'caminho/para/seus/jsons' pelo diretório real
    # Ex: analyzer = ClinicalAuditAnalyzer(input_path='./dados_prontuarios')
    
    # Para teste, assumindo pasta local 'data'
    path_test = './analise/prontuarios_auditados/' 
    if not os.path.exists(path_test):
        os.makedirs(path_test)
        print(f"Aviso: Crie a pasta '{path_test}' e insira seus JSONs nela.")
    else:
        analyzer = ClinicalAuditAnalyzer(input_path=path_test)
        analyzer.run_pipeline()