import os
import glob


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.normpath(
        os.path.join(base_dir, "..", "modelo_pkl", "artifacts")
    )
    relative_path = "../modelo_pkl/artifacts/"

    try:
        os.listdir(artifacts_dir)
    except FileNotFoundError:
        print(
            "MISSING_DATA: Diretório "
            f"'{relative_path}' não encontrado. "
            "Coloque os arquivos de modelo treinados (.pkl) nesse caminho."
        )
        return

    pattern = os.path.join(artifacts_dir, "*.pkl")
    pkl_files = glob.glob(pattern)

    if not pkl_files:
        print(
            "MISSING_DATA: Nenhum arquivo .pkl encontrado em "
            f"'{relative_path}'. "
            "Coloque os arquivos de modelo treinados (.pkl) nesse caminho."
        )
        return

    print(f"Arquivos .pkl encontrados em '{relative_path}':")
    for file_path in pkl_files:
        print(f"- {os.path.basename(file_path)}")


if __name__ == "__main__":
    main()

