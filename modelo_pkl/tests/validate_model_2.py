#!/usr/bin/env python3
"""
Script para validar a qualidade de um modelo treinado.
Uso: python validate_model.py [caminho_do_modelo.pkl]
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from termcolor import colored

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Adiciona o diretório raiz ao path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_model(path: str) -> Dict[str, Any]:
    """Carrega o modelo do arquivo pkl."""
    try:
        return joblib.load(path)
    except Exception as e:
        raise ValueError(f"Erro ao carregar modelo: {e}")


def interpret_roc_auc(score: float) -> Tuple[str, str]:
    """Interpreta o score ROC-AUC."""
    if score >= 0.90:
        return ("EXCEPCIONAL", "green")
    elif score >= 0.85:
        return ("EXCELENTE", "green")
    elif score >= 0.75:
        return ("BOM", "blue")
    elif score >= 0.65:
        return ("RAZOÁVEL", "yellow")
    elif score >= 0.55:
        return ("FRACO", "yellow")
    else:
        return ("MUITO FRACO", "red")


def interpret_brier(score: float) -> Tuple[str, str]:
    """Interpreta o Brier Score."""
    if score <= 0.08:
        return ("EXCEPCIONAL", "green")
    elif score <= 0.12:
        return ("EXCELENTE", "green")
    elif score <= 0.15:
        return ("BOM", "blue")
    elif score <= 0.20:
        return ("RAZOÁVEL", "yellow")
    elif score <= 0.25:
        return ("FRACO", "yellow")
    else:
        return ("MUITO FRACO", "red")


def get_recommendations(metadata: Dict[str, Any]) -> List[str]:
    """Gera recomendações baseadas nos metadados do modelo."""
    recommendations = []
    
    n_samples = metadata.get("n_samples", 0)
    n_features = metadata.get("n_features", 0)
    metrics = metadata.get("metrics", {})
    roc_auc = metrics.get("roc_auc", 0) or 0.0
    brier = metrics.get("brier_score", 1) or 1.0
    model_type = metadata.get("model_type", "unknown")

    # Análise de tamanho do dataset
    if n_samples < 300:
        recommendations.append(
            "🔴 CRÍTICO: Dataset muito pequeno (<300 amostras).\n"
            "   • Coletar pelo menos 500-1000 amostras para resultados confiáveis\n"
            "   • Considere usar validação leave-one-out ao invés de k-fold\n"
            "   • Modelo pode estar overfitting severamente"
        )
    elif n_samples < 500:
        recommendations.append(
            "⚠️  Dataset pequeno (<500 amostras).\n"
            "   • Coletar mais dados melhorará significativamente o modelo\n"
            "   • Evite modelos muito complexos (use regularização forte)"
        )
    elif n_samples < 1000:
        recommendations.append(
            "💡 Dataset razoável, mas mais dados sempre ajudam.\n"
            "   • Meta: 1000+ amostras para modelos mais robustos"
        )

    # Análise de performance
    if roc_auc < 0.55:
        recommendations.append(
            "🔴 CRÍTICO: ROC-AUC muito próximo do aleatório (0.50).\n"
            "   • Verificar se as features têm poder preditivo\n"
            "   • Revisar se o target está corretamente mapeado\n"
            "   • Checar balanceamento de classes\n"
            "   • Considere feature selection/engineering profundo"
        )
    elif roc_auc < 0.70:
        recommendations.append(
            "📊 ROC-AUC abaixo do ideal. Considere:\n"
            "   • Feature engineering (criar novas features relevantes)\n"
            "   • Análise de correlação entre features e target\n"
            "   • Remover features ruidosas\n"
            "   • Otimização de hiperparâmetros\n"
            "   • Experimentar outros algoritmos (XGBoost, CatBoost)"
        )
    elif roc_auc < 0.80:
        recommendations.append(
            "✅ ROC-AUC bom, mas há espaço para melhoria:\n"
            "   • Fine-tuning de hiperparâmetros\n"
            "   • Ensemble de modelos\n"
            "   • Feature engineering avançado"
        )

    if brier > 0.20:
        recommendations.append(
            "🎯 Brier Score alto indica má calibração. Considere:\n"
            "   • Aplicar calibração (Platt Scaling ou Isotonic Regression)\n"
            "   • Revisar dados de treino/validação\n"
            "   • Verificar se há outliers nos dados\n"
            "   • Ajustar threshold de decisão baseado em custo-benefício"
        )
    elif brier > 0.15:
        recommendations.append(
            "⚖️  Calibração pode ser melhorada:\n"
            "   • Testar diferentes métodos de calibração\n"
            "   • Verificar distribuição das probabilidades preditas"
        )

    # Análise de features
    if n_features > 100 and n_samples < 1000:
        recommendations.append(
            "⚠️  Razão features/amostras desfavorável.\n"
            "   • Considere feature selection (remove features irrelevantes)\n"
            "   • Use regularização forte (L1 ou ElasticNet)\n"
            "   • Aplique PCA/dimensionality reduction se apropriado"
        )
    elif n_features > 50 and n_samples < 500:
        recommendations.append(
            "💡 Muitas features para poucos dados:\n"
            "   • Feature selection pode melhorar generalização\n"
            "   • Use regularização para prevenir overfitting"
        )

    # Análise de balanceamento
    class_mapping = metadata.get("class_mapping", {})
    if class_mapping:
        recommendations.append(
            "⚖️  Verificar balanceamento de classes:\n"
            "   • Se muito desbalanceado (>80/20), considere:\n"
            "     - SMOTE ou outras técnicas de oversampling\n"
            "     - Ajustar class_weight no modelo\n"
            "     - Usar métricas apropriadas (F1, Precision-Recall AUC)\n"
            "   • Se balanceado, está ok!"
        )

    # Recomendações por tipo de modelo
    if model_type == "lightgbm":
        if roc_auc < 0.80:
            recommendations.append(
                "🌳 LightGBM específico:\n"
                "   • Ajustar num_leaves e max_depth\n"
                "   • Testar diferentes learning_rates\n"
                "   • Experimentar min_child_samples para evitar overfitting"
            )
    elif model_type == "logreg":
        if roc_auc < 0.75:
            recommendations.append(
                "📈 Regressão Logística:\n"
                "   • Pode ser muito simples para este problema\n"
                "   • Considere modelos não-lineares (LightGBM, XGBoost)\n"
                "   • Adicione features polinomiais ou interações"
            )

    return recommendations


def print_header(text: str):
    """Imprime cabeçalho formatado."""
    print("\n" + "=" * 80)
    print(colored(f"  {text}", "cyan", attrs=["bold"]))
    print("=" * 80)


def print_metric(name: str, value: float, interpretation: str, color: str, info: str = ""):
    """Imprime métrica formatada."""
    print(f"\n{colored(name + ':', 'white', attrs=['bold'])} {value:.4f}")
    print(f"  ├─ Avaliação: {colored(interpretation, color, attrs=['bold'])}")
    if info:
        print(f"  └─ Info: {info}")


def calculate_additional_metrics(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Calcula métricas adicionais úteis."""
    additional = {}
    
    n_samples = metadata.get("n_samples", 0)
    n_features = metadata.get("n_features", 0)
    
    if n_samples > 0 and n_features > 0:
        additional["samples_per_feature"] = n_samples / n_features
        
        # Regra de ouro: 10+ samples por feature
        if additional["samples_per_feature"] >= 10:
            additional["data_sufficiency"] = "ADEQUADO"
            additional["sufficiency_color"] = "green"
        elif additional["samples_per_feature"] >= 5:
            additional["data_sufficiency"] = "ACEITÁVEL"
            additional["sufficiency_color"] = "yellow"
        else:
            additional["data_sufficiency"] = "INSUFICIENTE"
            additional["sufficiency_color"] = "red"
    
    return additional


def compare_with_baseline(metadata: Dict[str, Any]):
    """Compara o modelo com baseline aleatório."""
    metrics = metadata.get("metrics", {})
    roc_auc = metrics.get("roc_auc", 0) or 0.0
    brier = metrics.get("brier_score", 1) or 1.0
    
    baseline_roc = 0.50
    baseline_brier = 0.25
    
    improvement_roc = ((roc_auc - baseline_roc) / baseline_roc) * 100
    improvement_brier = ((baseline_brier - brier) / baseline_brier) * 100
    
    print_header("📈 COMPARAÇÃO COM BASELINE")
    
    print(f"\n{'ROC-AUC:':<30}")
    print(f"  {'Modelo Atual:':<25} {colored(f'{roc_auc:.4f}', 'cyan')}")
    print(f"  {'Baseline (aleatório):':<25} {baseline_roc:.4f}")
    
    if improvement_roc > 50:
        color_roc = "green"
        emoji_roc = "🎯"
    elif improvement_roc > 20:
        color_roc = "blue"
        emoji_roc = "✓"
    elif improvement_roc > 0:
        color_roc = "yellow"
        emoji_roc = "⚠️"
    else:
        color_roc = "red"
        emoji_roc = "❌"
    
    print(f"  {'Melhoria:':<25} {colored(f'{emoji_roc} +{improvement_roc:.1f}%', color_roc, attrs=['bold'])}")
    
    print(f"\n{'Brier Score:':<30}")
    print(f"  {'Modelo Atual:':<25} {colored(f'{brier:.4f}', 'cyan')}")
    print(f"  {'Baseline (aleatório):':<25} {baseline_brier:.4f}")
    
    if improvement_brier > 40:
        color_brier = "green"
        emoji_brier = "🎯"
    elif improvement_brier > 20:
        color_brier = "blue"
        emoji_brier = "✓"
    elif improvement_brier > 0:
        color_brier = "yellow"
        emoji_brier = "⚠️"
    else:
        color_brier = "red"
        emoji_brier = "❌"
    
    print(f"  {'Melhoria:':<25} {colored(f'{emoji_brier} +{improvement_brier:.1f}%', color_brier, attrs=['bold'])}")
    
    if improvement_roc < 10:
        print(colored("\n❌ ALERTA: Modelo marginalmente melhor que baseline!", "red", attrs=["bold"]))
        print("   • Revisar completamente a estratégia de modelagem")
        print("   • Verificar qualidade e relevância dos dados")
        print("   • Considere se o problema é realmente previsível")


def plot_metrics_history(artifacts_dir: Path):
    """Plota histórico de métricas dos modelos."""
    if not HAS_MATPLOTLIB:
        return
    
    from datetime import datetime
    
    models = sorted(artifacts_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime)
    if len(models) < 2:
        print("\n💡 Apenas um modelo encontrado. Histórico será gerado com mais treinos.")
        return
    
    history = []
    for model_path in models:
        try:
            artifact = load_model(str(model_path))
            metadata = artifact.get("metadata", {})
            metrics = metadata.get("metrics", {})
            timestamp_str = metadata.get("timestamp", "")
            
            if not timestamp_str:
                continue
            
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            
            history.append({
                "timestamp": timestamp,
                "roc_auc": metrics.get("roc_auc", 0) or 0.0,
                "brier": metrics.get("brier_score", 0) or 0.0,
                "name": model_path.name,
                "n_samples": metadata.get("n_samples", 0),
            })
        except Exception as e:
            print(f"⚠️  Erro ao processar {model_path.name}: {e}")
            continue
    
    if len(history) < 2:
        return
    
    # Configurar estilo
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    timestamps = [h["timestamp"] for h in history]
    roc_values = [h["roc_auc"] for h in history]
    brier_values = [h["brier"] for h in history]
    sample_counts = [h["n_samples"] for h in history]
    
    # ROC-AUC ao longo do tempo
    axes[0, 0].plot(timestamps, roc_values, marker="o", linewidth=2, markersize=8)
    axes[0, 0].axhline(y=0.75, color="green", linestyle="--", alpha=0.5, label="Bom (0.75)")
    axes[0, 0].axhline(y=0.85, color="darkgreen", linestyle="--", alpha=0.5, label="Excelente (0.85)")
    axes[0, 0].axhline(y=0.50, color="red", linestyle="--", alpha=0.5, label="Baseline (0.50)")
    axes[0, 0].set_title("ROC-AUC ao Longo do Tempo", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("ROC-AUC Score")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0.4, 1.0])
    
    # Brier Score ao longo do tempo
    axes[0, 1].plot(timestamps, brier_values, marker="o", linewidth=2, markersize=8, color="orange")
    axes[0, 1].axhline(y=0.15, color="green", linestyle="--", alpha=0.5, label="Bom (0.15)")
    axes[0, 1].axhline(y=0.12, color="darkgreen", linestyle="--", alpha=0.5, label="Excelente (0.12)")
    axes[0, 1].axhline(y=0.25, color="red", linestyle="--", alpha=0.5, label="Baseline (0.25)")
    axes[0, 1].set_title("Brier Score ao Longo do Tempo", fontsize=12, fontweight="bold")
    axes[0, 1].set_ylabel("Brier Score (menor é melhor)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Tamanho do dataset
    axes[1, 0].plot(timestamps, sample_counts, marker="s", linewidth=2, markersize=8, color="purple")
    axes[1, 0].set_title("Tamanho do Dataset de Treino", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Número de Amostras")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Melhoria vs Baseline
    improvements = [((roc - 0.5) / 0.5) * 100 for roc in roc_values]
    colors = ["green" if imp > 50 else "yellow" if imp > 20 else "red" for imp in improvements]
    axes[1, 1].bar(range(len(improvements)), improvements, color=colors, alpha=0.7)
    axes[1, 1].axhline(y=20, color="blue", linestyle="--", alpha=0.5, label="Mínimo Aceitável")
    axes[1, 1].set_title("Melhoria vs Baseline (%)", fontsize=12, fontweight="bold")
    axes[1, 1].set_ylabel("Melhoria (%)")
    axes[1, 1].set_xlabel("Versões do Modelo")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    metrics_path = artifacts_dir / "metrics_history.png"
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(metrics_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\n📊 Gráfico salvo: {colored(str(metrics_path), 'cyan')}")


def validate_model(artifact_path: str):
    """Função principal de validação."""
    print_header("🔍 VALIDAÇÃO DE MODELO - ANÁLISE COMPLETA")
    print(f"\nArquivo: {colored(artifact_path, 'yellow')}")
    
    # Carregar modelo
    try:
        artifact = load_model(artifact_path)
    except Exception as e:
        print(colored(f"\n❌ Erro ao carregar modelo: {e}", "red"))
        sys.exit(1)
    
    metadata = artifact.get("metadata", {})
    metrics = metadata.get("metrics", {})
    
    # Verificar integridade
    if "model" not in artifact:
        print(colored("\n⚠️  AVISO: Chave 'model' não encontrada no artifact!", "yellow"))
    
    if not metadata:
        print(colored("\n⚠️  AVISO: Metadados vazios ou ausentes!", "yellow"))
    
    # Métricas de Desempenho
    print_header("📊 MÉTRICAS DE DESEMPENHO")
    
    roc_auc = metrics.get("roc_auc", 0) or 0.0
    roc_interp, roc_color = interpret_roc_auc(roc_auc)
    print_metric(
        "ROC-AUC Score",
        roc_auc,
        roc_interp,
        roc_color,
        "Mede capacidade de discriminação (0.5=aleatório, 1.0=perfeito)"
    )
    
    brier = metrics.get("brier_score", 1) or 1.0
    brier_interp, brier_color = interpret_brier(brier)
    print_metric(
        "Brier Score",
        brier,
        brier_interp,
        brier_color,
        "Mede calibração das probabilidades (0.0=perfeito, 0.25=aleatório)"
    )
    
    # Avaliação Geral
    print_header("✅ AVALIAÇÃO GERAL DO MODELO")
    
    scores = {
        "EXCEPCIONAL": 5,
        "EXCELENTE": 4,
        "BOM": 3,
        "RAZOÁVEL": 2,
        "FRACO": 1,
        "MUITO FRACO": 0
    }
    avg_score = (scores.get(roc_interp, 0) + scores.get(brier_interp, 0)) / 2
    
    if avg_score >= 4.5:
        print(colored("\n🏆 MODELO EXCEPCIONAL - PRONTO PARA PRODUÇÃO", "green", attrs=["bold"]))
        print("   ✓ Métricas excelentes em todos os aspectos")
        print("   ✓ Alta confiabilidade para uso em produção")
        print("   ✓ Pode ser usado para decisões críticas")
    elif avg_score >= 3.5:
        print(colored("\n✅ MODELO EXCELENTE - PRONTO PARA PRODUÇÃO", "green", attrs=["bold"]))
        print("   ✓ Métricas muito boas")
        print("   ✓ Apto para uso em produção")
        print("   ✓ Monitoramento regular recomendado")
    elif avg_score >= 2.5:
        print(colored("\n✓ MODELO UTILIZÁVEL - COM RESSALVAS", "blue", attrs=["bold"]))
        print("   • Desempenho aceitável para uso não-crítico")
        print("   • Recomenda-se melhorias antes de produção")
        print("   • Use com supervisão humana para decisões importantes")
    elif avg_score >= 1.5:
        print(colored("\n⚠️  MODELO PRECISA MELHORAR", "yellow", attrs=["bold"]))
        print("   • Não recomendado para produção")
        print("   • Retreinar com mais dados ou ajustar estratégia")
        print("   • Considere revisão completa do pipeline")
    else:
        print(colored("\n❌ MODELO INADEQUADO", "red", attrs=["bold"]))
        print("   • NÃO usar em produção")
        print("   • Pouco ou nenhum poder preditivo")
        print("   • Revisar completamente dados e abordagem")
    
    # Comparação com Baseline
    compare_with_baseline(metadata)
    
    # Informações do Dataset
    print_header("📁 INFORMAÇÕES DO DATASET E TREINAMENTO")
    
    n_samples = metadata.get("n_samples", 0)
    n_features = metadata.get("n_features", 0)
    cv_splits = metadata.get("cv_splits", 0)
    model_type = metadata.get("model_type", "N/A")
    timestamp = metadata.get("timestamp", "N/A")
    
    print(f"\n{'Amostras treinadas:':<30} {colored(f'{n_samples:,}', 'cyan')}")
    print(f"{'Features utilizadas:':<30} {colored(f'{n_features:,}', 'cyan')}")
    print(f"{'Cross-validation splits:':<30} {colored(cv_splits, 'cyan')}")
    print(f"{'Tipo de modelo:':<30} {colored(model_type.upper(), 'cyan')}")
    
    if timestamp != "N/A":
        print(f"{'Data de treinamento:':<30} {colored(timestamp[:19].replace('T', ' '), 'cyan')}")
    
    # Métricas adicionais
    additional = calculate_additional_metrics(metadata)
    if additional:
        samples_per_feature = additional.get("samples_per_feature", 0)
        colored_value = colored(f"{samples_per_feature:.1f}", "cyan")
        print(f"\n{'Amostras por feature:':<30} {colored_value}")
        suff = additional.get("data_sufficiency", "N/A")
        suff_color = additional.get("sufficiency_color", "white")
        print(f"{'Suficiência de dados:':<30} {colored(suff, suff_color, attrs=['bold'])}")
        
        if suff == "INSUFICIENTE":
            print(colored("   ⚠️  Regra de ouro: 10+ amostras por feature", "yellow"))
    
    # Mapeamento de classes
    class_mapping = metadata.get("class_mapping", {})
    if class_mapping:
        print(f"\n{'Mapeamento de Classes:':<30}")
        negative = class_mapping.get("negative", [])
        positive = class_mapping.get("positive", [])
        print(f"  {'• Classe 0 (Negativa):':<28} {', '.join(negative) if negative else 'N/A'}")
        print(f"  {'• Classe 1 (Positiva):':<28} {', '.join(positive) if positive else 'N/A'}")
    
    # Hiperparâmetros (se disponível)
    hyperparams = metadata.get("hyperparameters", {})
    best_params = metadata.get("best_params", {})
    
    if best_params:
        print(f"\n{'Hiperparâmetros Otimizados:':<30}")
        for key, value in best_params.items():
            print(f"  • {key:<26} {value}")
    elif hyperparams and len(hyperparams) <= 10:
        print(f"\n{'Hiperparâmetros:':<30}")
        for key, value in list(hyperparams.items())[:10]:
            print(f"  • {key:<26} {value}")
    
    # Recomendações
    recommendations = get_recommendations(metadata)
    if recommendations:
        print_header("💡 RECOMENDAÇÕES DE MELHORIA")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec}")
    else:
        print_header("💡 RECOMENDAÇÕES")
        print("\n✅ Modelo está em ótimas condições!")
        print("   • Continue monitorando performance em produção")
        print("   • Retreine periodicamente com novos dados")
    
    # Histórico de métricas
    artifacts_dir = Path(artifact_path).parent
    if HAS_MATPLOTLIB:
        try:
            plot_metrics_history(artifacts_dir)
        except Exception as e:
            print(f"\n⚠️  Erro ao gerar gráficos: {e}")
    else:
        print(f"\n💡 Instale matplotlib e seaborn para visualizações:")
        print(f"   {colored('pip install matplotlib seaborn', 'yellow')}")
    
    # Próximos passos
    print_header("🚀 PRÓXIMOS PASSOS RECOMENDADOS")
    
    if avg_score >= 3.5:
        print("\n1. ✅ Testar com dados de validação externos:")
        print(f"   {colored('python test_predictions.py', 'green')}")
        
        print("\n2. 📊 Analisar predições individuais:")
        print(f"   {colored('from src.inference import load_model, predict_proba', 'green')}")
        
        print("\n3. 🚀 Preparar para deploy:")
        print("   • Documentar versão e métricas")
        print("   • Configurar monitoramento de drift")
        print("   • Estabelecer pipeline de retreinamento")
    else:
        print("\n1. 🔧 Melhorar o modelo:")
        print("   • Revisar feature engineering")
        print("   • Coletar mais dados de qualidade")
        print("   • Experimentar diferentes algoritmos")
        
        print("\n2. 📊 Análise exploratória:")
        print("   • Verificar distribuição de classes")
        print("   • Analisar correlações features vs target")
        print("   • Identificar outliers")
    
    print("\n3. 📈 Monitoramento contínuo:")
    print("   • Calcular métricas em dados novos mensalmente")
    print("   • Verificar feature drift")
    print("   • Retreinar quando performance degradar")
    print("   • Manter histórico de versões\n")
    
    # Sumário final
    print("=" * 80)
    print(colored("SUMÁRIO:", "cyan", attrs=["bold"]))
    print(f"  ROC-AUC: {colored(f'{roc_auc:.4f}', roc_color)} ({roc_interp})")
    print(f"  Brier:   {colored(f'{brier:.4f}', brier_color)} ({brier_interp})")
    print(f"  Status:  ", end="")
    if avg_score >= 3.5:
        print(colored("APROVADO PARA PRODUÇÃO ✓", "green", attrs=["bold"]))
    elif avg_score >= 2.5:
        print(colored("UTILIZÁVEL COM RESSALVAS ⚠", "yellow", attrs=["bold"]))
    else:
        print(colored("REQUER MELHORIAS ✗", "red", attrs=["bold"]))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    
    if len(sys.argv) < 2:
        # Buscar modelo mais recente
        artifacts_dir = project_root / "artifacts"
        if artifacts_dir.exists():
            models = list(artifacts_dir.glob("*.pkl"))
            if models:
                latest = max(models, key=lambda p: p.stat().st_mtime)
                print(colored(
                    f"Modelo mais recente encontrado: {latest.name}",
                    "yellow",
                    attrs=["bold"],
                ))
                validate_model(latest)
            else:
                print(colored(
                    "Nenhum modelo encontrado na pasta artifacts.",
                    "yellow",
                    attrs=["bold"],
                ))
        else:
            print(colored(
                "Pasta artifacts não encontrada. Crie-a primeiro.",
                "yellow",
                attrs=["bold"],
            ))
    else:
        model_path = Path(sys.argv[1])
        if model_path.exists():
            validate_model(model_path)
        else:
            print(colored(
                f"Modelo {model_path.name} não encontrado.",
                "red",
                attrs=["bold"],
            ))
