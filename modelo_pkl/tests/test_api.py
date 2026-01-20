import io
import json
import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import app


client = TestClient(app)


def _build_valid_csv() -> str:
    return """situacao,sexo,cor_raca,municipio_residencia,uf_residencia,data_nascimento,idade,curso,campus,turno,modalidade_ingresso,tipo_cota,coeficiente_rendimento,disciplinas_aprovadas,disciplinas_reprovadas_nota,disciplinas_reprovadas_frequencia,periodo,ano_ingresso,semestre_ingresso,nota_enem_humanas,nota_enem_linguagem,nota_enem_matematica,nota_enem_natureza,nota_enem_redacao
regular,f,branca,cidade_a,sp,18/09/1992,26,engenharia,campus_a,integral,ampla_concorrencia,nenhuma,7.5,10,2,1,8,2018,1,600,650,700,620,720
desistente,m,parda,cidade_b,rj,05/03/1990,28,direito,campus_b,noturno,cotas,renda,6.5,8,3,0,8,2019,1,550,600,650,580,680
trancado,f,preta,cidade_c,mg,12/07/1993,25,medicina,campus_c,integral,ampla_concorrencia,nenhuma,8.0,12,1,0,8,2017,2,620,630,640,650,660
formado,m,branca,cidade_d,ba,23/11/1989,30,engenharia,campus_d,integral,ampla_concorrencia,nenhuma,7.0,14,0,0,8,2016,2,580,590,600,610,620
"""


def _build_vest_only_csv() -> str:
    return """situacao,sexo,cor_raca,municipio_residencia,uf_residencia,data_nascimento,idade,curso,campus,turno,modalidade_ingresso,tipo_cota,coeficiente_rendimento,disciplinas_aprovadas,disciplinas_reprovadas_nota,disciplinas_reprovadas_frequencia,periodo,ano_ingresso,semestre_ingresso,nota_vestibular_matematica,nota_vestibular_lingua_portuguesa,nota_vestibular_biologia,nota_vestibular_historia
regular,f,branca,cidade_a,sp,18/09/1992,26,engenharia,campus_a,integral,ampla_concorrencia,nenhuma,7.5,10,2,1,8,2018,1,8.5,9.0,7.5,8.0
desistente,m,parda,cidade_b,rj,05/03/1990,28,direito,campus_b,noturno,cotas,renda,6.5,8,3,0,8,2019,1,7.2,8.1,6.8,7.5
"""


class TestTrainEndpoint:
    def test_train_endpoint_success(self):
        """Teste básico de sucesso: treino com CSV válido deve retornar métricas."""
        csv_content = _build_valid_csv()
        file_data = io.StringIO(csv_content).getvalue()

        files = {
            "file": ("dummy.csv", file_data, "text/csv"),
        }

        config = {"cv_splits": 2, "calibrate": False}
        data = {"config": json.dumps(config)}

        response = client.post("/train", files=files, data=data)

        assert response.status_code == 200

        payload = response.json()

        assert "artifact_path" in payload
        assert "metrics" in payload
        assert "model_version" in payload

        metrics = payload["metrics"]

        assert isinstance(metrics, dict)
        assert "roc_auc" in metrics
        assert "brier_score" in metrics

    def test_train_endpoint_vest_only_success(self):
        csv_content = _build_vest_only_csv()
        file_data = io.StringIO(csv_content).getvalue()
        files = {"file": ("dummy.csv", file_data, "text/csv")}
        config = {"cv_splits": 2, "calibrate": False}
        data = {"config": json.dumps(config)}
        response = client.post("/train", files=files, data=data)
        assert response.status_code == 200
        payload = response.json()
        assert "metrics" in payload
        metrics = payload["metrics"]
        assert isinstance(metrics, dict)
        assert "roc_auc" in metrics
        assert "brier_score" in metrics

    def test_train_endpoint_vest_only_lightgbm_optimize_trials(self):
        csv_content = _build_vest_only_csv()
        file_data = io.StringIO(csv_content).getvalue()
        files = {"file": ("dummy.csv", file_data, "text/csv")}
        config = {
            "model_type": "lightgbm",
            "cv_splits": 2,
            "calibrate": False,
            "optimize_trials": 5,
        }
        data = {"config": json.dumps(config)}
        response = client.post("/train", files=files, data=data)
        assert response.status_code == 200
        payload = response.json()
        metrics = payload.get("metrics") or {}
        assert isinstance(metrics, dict)
        assert "roc_auc" in metrics
        assert "brier_score" in metrics

    def test_train_endpoint_lightgbm_config_success(self):
        csv_content = _build_valid_csv()
        file_data = io.StringIO(csv_content).getvalue()
        files = {"file": ("dummy.csv", file_data, "text/csv")}
        config = {
            "model_type": "lightgbm",
            "cv_splits": 2,
            "calibrate": False,
            "optimize_trials": 3,
        }
        data = {"config": json.dumps(config)}
        response = client.post("/train", files=files, data=data)
        assert response.status_code == 200
        payload = response.json()
        metrics = payload.get("metrics") or {}
        assert isinstance(metrics, dict)
        assert "roc_auc" in metrics
        assert "brier_score" in metrics

    def test_train_endpoint_missing_target_column(self):
        """CSV sem coluna 'situacao' deve retornar 400 com mensagem de erro clara."""
        csv_content = """sexo,cor_raca,municipio_residencia,uf_residencia,data_nascimento,idade,curso,campus,turno,modalidade_ingresso,tipo_cota,coeficiente_rendimento,disciplinas_aprovadas,disciplinas_reprovadas_nota,disciplinas_reprovadas_frequencia,periodo,ano_ingresso,semestre_ingresso,nota_enem_humanas,nota_enem_linguagem,nota_enem_matematica,nota_enem_natureza,nota_enem_redacao
f,branca,cidade_a,sp,18/09/1992,26,engenharia,campus_a,integral,ampla_concorrencia,nenhuma,7.5,10,2,1,8,2018,1,600,650,700,620,720
"""
        file_data = io.StringIO(csv_content).getvalue()
        files = {"file": ("dummy.csv", file_data, "text/csv")}
        response = client.post("/train", files=files)

        assert response.status_code == 400
        payload = response.json()
        assert "Coluna target 'situacao' obrigatória." in payload.get("detail", "")

    def test_train_endpoint_non_csv_file(self):
        """.txt ou outros formatos que não .csv devem ser rejeitados com 400."""
        file_data = "conteudo qualquer"
        files = {"file": ("dummy.txt", file_data, "text/plain")}
        response = client.post("/train", files=files)

        assert response.status_code == 400
        payload = response.json()
        assert "Only .csv files are supported." == payload.get("detail")

    def test_train_endpoint_empty_file(self):
        """CSV vazio (0 bytes) deve ser tratado como erro de leitura com 400."""
        file_data = ""
        files = {"file": ("empty.csv", file_data, "text/csv")}
        response = client.post("/train", files=files)

        assert response.status_code == 400
        payload = response.json()
        assert "Error reading CSV:" in payload.get("detail", "")

    def test_train_endpoint_invalid_json_config(self):
        """Configuração JSON malformada deve retornar 400 com mensagem de JSON inválido."""
        csv_content = _build_valid_csv()
        file_data = io.StringIO(csv_content).getvalue()
        files = {"file": ("dummy.csv", file_data, "text/csv")}

        bad_config = "{invalid_json"
        data = {"config": bad_config}

        response = client.post("/train", files=files, data=data)

        assert response.status_code == 400
        payload = response.json()
        assert "Invalid JSON config" in payload.get("detail", "")

    def test_train_endpoint_validates_metrics_range(self):
        """Treino bem-sucedido deve gerar métricas em range válido e artefato .pkl existente."""
        csv_content = _build_valid_csv()
        file_data = io.StringIO(csv_content).getvalue()

        files = {
            "file": ("dummy.csv", file_data, "text/csv"),
        }

        config = {"cv_splits": 2, "calibrate": False}
        data = {"config": json.dumps(config)}

        response = client.post("/train", files=files, data=data)

        assert response.status_code == 200
        payload = response.json()
        artifact_path = payload.get("artifact_path")
        metrics = payload.get("metrics") or {}

        assert isinstance(metrics, dict)
        roc_auc = metrics.get("roc_auc")
        brier = metrics.get("brier_score")

        assert roc_auc is not None
        assert brier is not None
        assert 0.0 <= roc_auc <= 1.0
        assert 0.0 <= brier <= 1.0

        assert isinstance(artifact_path, str)
        assert artifact_path.endswith(".pkl")
        assert os.path.exists(artifact_path)
