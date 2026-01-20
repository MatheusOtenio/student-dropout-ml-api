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
