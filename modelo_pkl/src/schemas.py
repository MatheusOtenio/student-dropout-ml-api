from datetime import date, datetime
from typing import Any, Optional, List

from pydantic import BaseModel, ConfigDict, field_validator


class StudentFeatures(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id_aluno: Optional[int] = None
    nome: Optional[str] = None
    data_nascimento: Optional[date] = None
    sexo: Optional[str] = None
    escola: Optional[str] = None
    serie: Optional[str] = None
    turno: Optional[str] = None
    ano_letivo: Optional[int] = None

    @field_validator("data_nascimento", mode="before")
    @classmethod
    def parse_data_nascimento(cls, value: Any) -> Optional[date]:
        if value is None or value == "":
            return None
        if isinstance(value, date):
            return value
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, str):
            for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
                try:
                    return datetime.strptime(value, fmt).date()
                except ValueError:
                    continue
        raise ValueError("data_nascimento must be a valid date")


class TrainingSample(StudentFeatures):
    situacao: str


class PredictionSample(StudentFeatures):
    situacao: Optional[str] = None


class TrainingDataset(BaseModel):
    records: List[TrainingSample]


class PredictionDataset(BaseModel):
    records: List[PredictionSample]


class SplitConfig(BaseModel):
    train_range: List[int]  # [start_year, end_year]
    val_year: int
    test_year: int

    @field_validator("train_range")
    @classmethod
    def validate_range(cls, v: List[int]) -> List[int]:
        if len(v) != 2:
            raise ValueError("train_range must have exactly 2 elements: [start, end]")
        if v[0] > v[1]:
            raise ValueError("train_range start must be <= end")
        return v


class ModelConfig(BaseModel):
    model_type: str = "lightgbm"
    split_config: Optional[SplitConfig] = None
    model_params: Optional[Dict[str, Any]] = None
    random_state: int = 42
    calibrate: bool = False
    calibration_cv: int = 5
    calibration_method: str = "isotonic"
    optimize_trials: int = 20
    artifact_path: Optional[str] = None
    version: str = "1.0.0"
