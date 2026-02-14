from datetime import date
from typing import Optional, List

from pydantic import BaseModel, Field


class AlunoTarget(BaseModel):
    sexo: str = Field(
        ...,
        description="Sexo declarado pelo aluno",
    )
    municipio_residencia: str = Field(
        ...,
        description="Município de residência do aluno",
    )
    data_nascimento: Optional[date] = Field(
        None,
        description="Data de nascimento do aluno, quando disponível",
    )
    idade: Optional[int] = Field(
        None,
        description="Idade do aluno, preferencialmente calculada na data de ingresso",
    )
    curso: Optional[str] = Field(
        None,
        description="Nome do curso de graduação",
    )
    campus: str = Field(
        ...,
        description="Campus ou sede em que o curso é ofertado",
    )
    turno: str = Field(
        ...,
        description="Turno de funcionamento do curso (matutino, vespertino, noturno, integral)",
    )
    modalidade_ingresso: str = Field(
        ...,
        description="Forma de ingresso do aluno na instituição (ENEM, vestibular, transferência, etc.)",
    )
    tipo_cota: Optional[str] = Field(
        None,
        description="Tipo de cota utilizada pelo aluno no ingresso, se houver",
    )
    situacao: Optional[str] = Field(
        None,
        description="Rótulo da situação do aluno: Matriculado, Evadido, etc.",
    )
    coeficiente_rendimento: float = Field(
        ...,
        description="Coeficiente de rendimento acumulado do aluno",
    )
    disciplinas_aprovadas: int = Field(
        ...,
        description="Número total de disciplinas aprovadas pelo aluno",
    )
    disciplinas_reprovadas_nota: int = Field(
        ...,
        description="Número total de disciplinas reprovadas por nota",
    )
    total_semestres_cursados: int = Field(
        ...,
        description="Total de semestres efetivamente cursados pelo aluno no curso atual",
    )
    periodo: Optional[int] = Field(
        None,
        description="Período ou semestre atual do aluno",
    )
    ano_ingresso: int = Field(
        ...,
        description="Ano de ingresso do aluno no curso atual",
    )
    semestre_ingresso: int = Field(
        ...,
        description="Semestre de ingresso do aluno no curso atual (1 ou 2)",
    )
    nota_enem_humanas: Optional[float] = Field(
        None,
        description="Nota em Ciências Humanas no ENEM utilizada para ingresso ou análise",
    )
    nota_enem_linguagem: Optional[float] = Field(
        None,
        description="Nota em Linguagens, Códigos e suas Tecnologias no ENEM",
    )
    nota_enem_matematica: Optional[float] = Field(
        None,
        description="Nota em Matemática e suas Tecnologias no ENEM",
    )
    nota_enem_natureza: Optional[float] = Field(
        None,
        description="Nota em Ciências da Natureza no ENEM",
    )
    nota_enem_redacao: Optional[float] = Field(
        None,
        description="Nota de Redação no ENEM",
    )
    nota_vestibular_biologia: Optional[float] = Field(
        None,
        description="Nota de Biologia no vestibular institucional",
    )
    nota_vestibular_filosofia_sociologia: Optional[float] = Field(
        None,
        description="Nota de Filosofia e Sociologia no vestibular institucional",
    )
    nota_vestibular_fisica: Optional[float] = Field(
        None,
        description="Nota de Física no vestibular institucional",
    )
    nota_vestibular_geografia: Optional[float] = Field(
        None,
        description="Nota de Geografia no vestibular institucional",
    )
    nota_vestibular_historia: Optional[float] = Field(
        None,
        description="Nota de História no vestibular institucional",
    )
    nota_vestibular_literatura_brasileira: Optional[float] = Field(
        None,
        description="Nota de Literatura Brasileira no vestibular institucional",
    )
    nota_vestibular_lingua_estrangeira: Optional[float] = Field(
        None,
        description="Nota de Língua Estrangeira Moderna no vestibular institucional",
    )
    nota_vestibular_lingua_portuguesa: Optional[float] = Field(
        None,
        description="Nota de Língua Portuguesa no vestibular institucional",
    )
    nota_vestibular_matematica: Optional[float] = Field(
        None,
        description="Nota de Matemática no vestibular institucional",
    )
    nota_vestibular_quimica: Optional[float] = Field(
        None,
        description="Nota de Química no vestibular institucional",
    )
    # New fields
    calouro: int = Field(..., description="Indicador se é calouro (0/1)")
    codigo_aluno: Optional[str] = Field(None, description="Identificador único do aluno")
    ano_referencia: Optional[int] = Field(None, description="Ano do snapshot dos dados")
    periodo_referencia: Optional[int] = Field(None, description="Período/Semestre do snapshot dos dados")


try:
    COLUNAS_WHITELIST: List[str] = list(AlunoTarget.model_fields.keys())
except AttributeError:
    COLUNAS_WHITELIST = list(AlunoTarget.__fields__.keys())


# ---------------------------------------------------------------------------
# Ordem EXATA das colunas que o modelo espera como entrada (Features).
# Deve ser mantida consistente entre Backend e Treinamento.
# ---------------------------------------------------------------------------
FEATURE_COLUMNS: List[str] = [
    "sexo",
    "municipio_residencia",
    "curso",
    "campus",
    "turno",
    "modalidade_ingresso",
    "tipo_cota",
    "coeficiente_rendimento",
    "disciplinas_aprovadas",
    "disciplinas_reprovadas_nota",
    "periodo",
    "ano_ingresso",
    "semestre_ingresso",
    "nota_enem_humanas",
    "nota_enem_linguagem",
    "nota_enem_matematica",
    "nota_enem_natureza",
    "nota_enem_redacao",
    "nota_vestibular_biologia",
    "nota_vestibular_filosofia_sociologia",
    "nota_vestibular_fisica",
    "nota_vestibular_geografia",
    "nota_vestibular_historia",
    "nota_vestibular_literatura_brasileira",
    "nota_vestibular_lingua_estrangeira",
    "nota_vestibular_lingua_portuguesa",
    "nota_vestibular_matematica",
    "nota_vestibular_quimica",
    "calouro",
    "idade",
]


# ---------------------------------------------------------------------------
# Definição dos Status para Filtragem (Consistência com Trainer)
# ---------------------------------------------------------------------------
STATUS_SUCESSO = {"formado"}
STATUS_FRACASSO = {"desistente", "trancado"}
STATUS_VALIDOS = STATUS_SUCESSO | STATUS_FRACASSO

