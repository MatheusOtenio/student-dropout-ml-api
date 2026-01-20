from datetime import date
from typing import Optional, List

from pydantic import BaseModel, Field


class AlunoTarget(BaseModel):
    sexo: str = Field(
        ...,
        description="Sexo declarado pelo aluno",
    )
    cor_raca: str = Field(
        ...,
        description="Cor ou raça autodeclarada do aluno",
    )
    municipio_residencia: str = Field(
        ...,
        description="Município de residência do aluno",
    )
    uf_residencia: str = Field(
        ...,
        description="Unidade federativa de residência do aluno",
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
    disciplinas_reprovadas_frequencia: int = Field(
        ...,
        description="Número total de disciplinas reprovadas por frequência",
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


try:
    COLUNAS_WHITELIST: List[str] = list(AlunoTarget.model_fields.keys())
except AttributeError:
    COLUNAS_WHITELIST = list(AlunoTarget.__fields__.keys())


