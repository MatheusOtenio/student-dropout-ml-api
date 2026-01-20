#!/usr/bin/env python3
"""
Script para validar a qualidade de um modelo treinado.
Uso: python validate_model.py artifacts/model_20260119T231430.pkl
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import joblib
import pandas as pd
from termcolor import colored

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_model(path: str) -> Dict[str, Any]:
    return joblib.load(path)


def interpret_roc_auc(score: float) -> Tuple[str, str]:
    if score >= 0.85:
        return ("EXCELENTE", "green")
    elif score >= 0.75:
        return ("BOM", "blue")
    elif score >= 0.65:
        return ("RAZOÁVEL", "yellow")
    else:
        return ("FRACO", "red")


def interpret_brier(score: float) -> Tuple[str, str]:
    if score <= 0.10:
        return ("EXCELENTE", "green")
    elif score <= 0.15:
        return ("BOM", "blue")
    elif score <= 0.20:
        return ("RAZOÁVEL", "yellow")
    else:
        return ("FRACO", "red")


def get_recommendations(metadata: Dict[str, Any]) -> List[str]:
    recommendations = []

    n_samples = metadata.get("n_samples", 0)
    metrics = metadata.get("metrics", {})
    roc_auc = metrics.get("roc_auc", 0)
    brier = metrics.get("brier_score", 1)

    if n_samples < 500:
        recommendations.append(
            "⚠️  Dataset pequeno (<500 amostras). Coletar mais dados pode melhorar o modelo."
        )

    if roc_auc < 0.75:
        recommendations.append(
            "📊 ROC-AUC abaixo do ideal. Considere:\n"
            "   • Feature engineering (criar novas features relevantes)\n"
            "   • Otimização de hiperparâmetros\n"
            "   • Experimentar outros algoritmos"
        )

    if brier > 0.18:
        recommendations.append(
            "🎯 Brier Score alto pode indicar má calibração ou modelo fraco. Considere:\n"
            "   • Revisar dados de treino/validação\n"
            "   • Revisar estratégia de calibração (CV, método)\n"
            "   • Ajustar threshold de decisão"
        )

    class_mapping = metadata.get("class_mapping", {})
    if class_mapping:
        recommendations.append(
            "⚖️  Verifique o balanceamento de classes nos seus dados.\n"
            "   • Se muito desbalanceado, considere SMOTE ou ajustar class_weight"
        )

    return recommendations


def print_header(text: str):
    print("\n" + "=" * 70)
    print(colored(f"  {text}", "cyan", attrs=["bold"]))
    print("=" * 70)


def print_metric(name: str, value: float, interpretation: str, color: str):
    print(f"\n{colored(name + ':', 'white', attrs=['bold'])} {value:.4f}")
    print(f"  └─ Avaliação: {colored(interpretation, color, attrs=['bold'])}")


def compare_with_baseline(metadata: Dict[str, Any]):
    metrics = metadata.get("metrics", {})
    roc_auc = metrics.get("roc_auc", 0) or 0.0
    baseline_roc = 0.50
    improvement = ((roc_auc - baseline_roc) / baseline_roc) * 100

    print_header("📈 COMPARAÇÃO COM BASELINE")
    print(f"\n{'Modelo Atual:':<20} {colored(f'{roc_auc:.4f}', 'cyan')}")
    print(f"{'Baseline (aleatório):':<20} {baseline_roc:.4f}")

    if improvement > 50:
        color = "green"
        emoji = "🎯"
    elif improvement > 20:
        color = "blue"
        emoji = "✓"
    else:
        color = "yellow"
        emoji = "⚠️"

    print(f"{'Melhoria:':<20} {colored(f'{emoji} +{improvement:.1f}%', color, attrs=['bold'])}")

    if improvement < 20:
        print(colored("⚠️  Modelo pouco melhor que baseline. Considere:", "yellow"))
        print("   • Coletar mais dados de qualidade")
        print("   • Revisar feature engineering")


def plot_metrics_history(artifacts_dir: Path):
    from datetime import datetime

    models = sorted(artifacts_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime)
    if len(models) < 2:
        return

    history = []
    for model_path in models:
        artifact = load_model(str(model_path))
        metadata = artifact.get("metadata", {})
        metrics = metadata.get("metrics", {})
        timestamp_str = metadata.get("timestamp", "")
        if not timestamp_str:
            continue
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except Exception:
            continue
        history.append(
            {
                "timestamp": timestamp,
                "roc_auc": metrics.get("roc_auc", 0),
                "brier": metrics.get("brier_score", 0),
                "name": model_path.name,
            }
        )

    if len(history) < 2:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    timestamps = [h["timestamp"] for h in history]
    roc_values = [h["roc_auc"] for h in history]
    brier_values = [h["brier"] for h in history]

    ax1.plot(timestamps, roc_values, marker="o", linewidth=2)
    ax1.axhline(y=0.75, color="green", linestyle="--", alpha=0.5, label="Threshold Bom")
    ax1.set_title("ROC-AUC ao longo do tempo")
    ax1.set_ylabel("ROC-AUC")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(timestamps, brier_values, marker="o", linewidth=2, color="orange")
    ax2.axhline(y=0.15, color="green", linestyle="--", alpha=0.5, label="Threshold Bom")
    ax2.set_title("Brier Score ao longo do tempo")
    ax2.set_ylabel("Brier Score")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    metrics_path = artifacts_dir / "metrics_history.png"
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(metrics_path, dpi=150)
    print(f"\n📊 Gráfico salvo: {colored(str(metrics_path), 'cyan')}")


def validate_model(artifact_path: str):
    print_header("🔍 VALIDANDO MODELO")
    print(f"\nArquivo: {colored(artifact_path, 'yellow')}")

    try:
        artifact = load_model(artifact_path)
    except Exception as e:
        print(colored(f"\n❌ Erro ao carregar modelo: {e}", "red"))
        sys.exit(1)

    metadata = artifact.get("metadata", {})
    metrics = metadata.get("metrics", {})

    print_header("📊 MÉTRICAS DE DESEMPENHO")

    roc_auc = metrics.get("roc_auc", 0)
    roc_interp, roc_color = interpret_roc_auc(roc_auc)
    print_metric("ROC-AUC Score", roc_auc, roc_interp, roc_color)
    print(f"  └─ Info: Mede a capacidade de discriminação (0.5 = aleatório, 1.0 = perfeito)")

    brier = metrics.get("brier_score", 1)
    brier_interp, brier_color = interpret_brier(brier)
    print_metric("Brier Score", brier, brier_interp, brier_color)
    print(f"  └─ Info: Mede calibração das probabilidades (0.0 = perfeito, 0.25 = aleatório)")

    print_header("✅ AVALIAÇÃO GERAL")

    scores = {"EXCELENTE": 4, "BOM": 3, "RAZOÁVEL": 2, "FRACO": 1}
    avg_score = (scores[roc_interp] + scores[brier_interp]) / 2

    if avg_score >= 3.5:
        print(colored("\n✅ MODELO PRONTO PARA PRODUÇÃO", "green", attrs=["bold"]))
        print("   O modelo apresenta métricas excelentes e está apto para uso real.")
    elif avg_score >= 2.5:
        print(colored("\n✓ MODELO UTILIZÁVEL", "blue", attrs=["bold"]))
        print("   O modelo apresenta desempenho aceitável, mas pode ser melhorado.")
    else:
        print(colored("\n⚠️  MODELO PRECISA MELHORAR", "yellow", attrs=["bold"]))
        print("   Considere retreinar com mais dados ou ajustar hiperparâmetros.")

    compare_with_baseline(metadata)

    print_header("📁 INFORMAÇÕES DO DATASET")

    print(f"\n{'Amostras treinadas:':<25} {colored(metadata.get('n_samples', 'N/A'), 'cyan')}")
    print(f"{'Features utilizadas:':<25} {colored(metadata.get('n_features', 'N/A'), 'cyan')}")
    print(f"{'Cross-validation splits:':<25} {colored(metadata.get('cv_splits', 'N/A'), 'cyan')}")
    print(f"{'Tipo de modelo:':<25} {colored(metadata.get('model_type', 'N/A'), 'cyan')}")
    print(f"{'Data de treinamento:':<25} {colored(metadata.get('timestamp', 'N/A')[:19], 'cyan')}")

    class_mapping = metadata.get("class_mapping", {})
    if class_mapping:
        print(f"\n{'Mapeamento de classes:':<25}")
        print(f"  {'• Classe Negativa (0):':<25} {', '.join(class_mapping.get('negative', []))}")
        print(f"  {'• Classe Positiva (1):':<25} {', '.join(class_mapping.get('positive', []))}")

    recommendations = get_recommendations(metadata)
    if recommendations:
        print_header("💡 RECOMENDAÇÕES DE MELHORIA")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec}")

    artifacts_dir = Path(artifact_path).parent
    if HAS_MATPLOTLIB:
        plot_metrics_history(artifacts_dir)
    else:
        print(f"\n💡 Instale matplotlib para visualizações: {colored('pip install matplotlib', 'yellow')}")

    print_header("🚀 PRÓXIMOS PASSOS")
    print("\n1. Testar com dados reais:")
    print(f"   {colored('python test_predictions.py', 'green')}")

    print("\n2. Analisar predições individuais:")
    print(f"   {colored('from src.inference import load_model, predict_proba', 'green')}")

    print("\n3. Monitorar em produção:")
    print("   • Calcular métricas em dados novos mensalmente")
    print("   • Verificar drift de features")
    print("   • Retreinar se performance degradar\n")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    if len(sys.argv) < 2:
        artifacts_dir = project_root / "artifacts"
        if artifacts_dir.exists():
            models = list(artifacts_dir.glob("*.pkl"))
            if models:
                latest = max(models, key=lambda p: p.stat().st_mtime)
                print(colored(f"📂 Nenhum arquivo especificado. Usando o mais recente: {latest}", "yellow"))
                validate_model(str(latest))
            else:
                print(colored("❌ Nenhum modelo encontrado em 'artifacts/'", "red"))
                sys.exit(1)
        else:
            print(colored("Uso: python validate_model.py <caminho_do_modelo.pkl>", "yellow"))
            print(colored("Exemplo: python validate_model.py artifacts/model_20260119T231430.pkl", "yellow"))
            sys.exit(1)
    else:
        path_arg = Path(sys.argv[1])
        if not path_arg.is_absolute():
            path_arg = project_root / path_arg
        validate_model(str(path_arg))
