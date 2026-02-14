from typing import Dict, List, Optional
import logging

from unidecode import unidecode
from rapidfuzz import fuzz, process

try:
    from .schemas import COLUNAS_WHITELIST
except ImportError:
    from schemas import COLUNAS_WHITELIST


logger = logging.getLogger(__name__)

MANUAL_HINTS: Dict[str, List[str]] = {
    "coeficiente_rendimento": ["cr", "coeficiente", "rendimento", "ira"],
    "municipio_residencia": ["municipio"],
}

ALIASES_CONHECIDOS: Dict[str, Optional[str]] = {
    # --- Dados pessoais ---
    "genero": "sexo",
    "sexo": "sexo",
    "idade": "idade",
    "data de nascimento": "data_nascimento",

    # --- Município / UF ---
    "municipio": "municipio_residencia",

    # --- Curso / matrícula ---
    "curso": "curso",
    "nome do curso": "curso",
    "curso graduacao": "curso",
    "campus": "campus",
    "turno": "turno",

    # --- Ingresso ---
    "forma de ingresso": "modalidade_ingresso",
    "modalidade de ingresso": "modalidade_ingresso",
    "tipo de cota": "tipo_cota",
    "cota": "tipo_cota",

    # --- Situação ---
    "situacao": "situacao",
    "situacao atual": "situacao",
    "status": "situacao",
    "status do aluno": "situacao",

    # --- Períodos / datas ---
    "periodo": "periodo",
    "serie": "periodo",
    "semestre atual": "periodo",
    "ano de ingresso": "ano_ingresso",
    "ano ingresso": "ano_ingresso",
    "semestre de ingresso no curso": "semestre_ingresso",
    "semestre de ingresso": "semestre_ingresso",
    "semestre ingresso": "semestre_ingresso",
    "total de semestres cursados": "total_semestres_cursados",

    # --- Disciplinas (formato 1: sem prefixo) ---
    "disciplinas aprovadas": "disciplinas_aprovadas",
    "disciplinas reprovadas por nota": "disciplinas_reprovadas_nota",

    # --- Disciplinas (formato 2: prefixo "Nr") ---
    "nr disciplinas aprovadas": "disciplinas_aprovadas",
    "nr disciplinas reprovadas por nota": "disciplinas_reprovadas_nota",

    # --- Novos Mapeamentos (x_columns_report.json) ---
    "provavel jubilamento": None,
    "retencao parcial": None,
    "retencao total": None,
    "calouro": "calouro",
    "coeficiente de rendimento absoluto": "coeficiente_rendimento",

    # --- Ignorar Explicitamente ---
    "disciplinas reprovadas por frequencia": None,
    "nr disciplinas reprovadas por frequencia": None,

    # --- ENEM (inclui typo "Liguagem" que vem do sistema de origem) ---
    "nota enem humanas": "nota_enem_humanas",
    "nota enem linguagem": "nota_enem_linguagem",
    "nota enem liguagem": "nota_enem_linguagem",      
    "nota enem matematica": "nota_enem_matematica",
    "nota enem natureza": "nota_enem_natureza",
    "nota enem redacao": "nota_enem_redacao",

    # --- Vestibular ---
    "nota vestibular biologia": "nota_vestibular_biologia",
    "nota vestibular filosofia e sociologia": "nota_vestibular_filosofia_sociologia",
    "nota vestibular fisica": "nota_vestibular_fisica",
    "nota vestibular geografia": "nota_vestibular_geografia",
    "nota vestibular historia": "nota_vestibular_historia",
    "nota vestibular literatura brasileira": "nota_vestibular_literatura_brasileira",
    "nota vestibular lingua estrangeira moderna (espanhol ou ingles)": "nota_vestibular_lingua_estrangeira",
    "nota vestibular lingua portuguesa": "nota_vestibular_lingua_portuguesa",
    "nota vestibular matematica": "nota_vestibular_matematica",
    "nota vestibular quimica": "nota_vestibular_quimica",
}


def gerar_sugestao_mapeamento(
    colunas_csv_bruto: List[str], threshold: int = 80
) -> Dict[str, Optional[str]]:
    colunas_normalizadas = [
        unidecode(c).strip().lower() for c in colunas_csv_bruto
    ]

    logger.info("Gerando mapeamento para %d colunas de entrada", len(colunas_csv_bruto))

    mapeamento: Dict[str, Optional[str]] = {}

    for coluna_schema in COLUNAS_WHITELIST:
        if not colunas_normalizadas:
            mapeamento[coluna_schema] = None
            continue

        indice_alias: Optional[int] = None
        for i, nome_normalizado in enumerate(colunas_normalizadas):
            alvo_alias = ALIASES_CONHECIDOS.get(nome_normalizado)
            if alvo_alias == coluna_schema:
                indice_alias = i
                break

        if indice_alias is not None:
            mapeamento[coluna_schema] = colunas_csv_bruto[indice_alias]
            continue

        query = unidecode(coluna_schema).strip().lower()

        allowed_indices: List[int] = []
        for i, nome_coluna in enumerate(colunas_normalizadas):
            if "enem" in coluna_schema and "enem" not in nome_coluna:
                continue
            if "vestibular" in coluna_schema and "vestibular" not in nome_coluna:
                continue
            allowed_indices.append(i)

        if not allowed_indices:
            mapeamento[coluna_schema] = None
            continue

        if coluna_schema == "municipio_residencia":
            indices_municipio = [
                i for i in allowed_indices if "municipio" in colunas_normalizadas[i]
            ]
            if indices_municipio:
                indices_preferidos = [
                    i for i in indices_municipio
                    if "sisu" not in colunas_normalizadas[i]
                ]
                indice_escolhido = (
                    indices_preferidos[0] if indices_preferidos else indices_municipio[0]
                )
                mapeamento[coluna_schema] = colunas_csv_bruto[indice_escolhido]
                continue

        indice_manual: Optional[int] = None
        padroes = MANUAL_HINTS.get(coluna_schema)
        if padroes:
            melhor_pontuacao = 0
            for i in allowed_indices:
                nome_coluna = colunas_normalizadas[i]
                pontuacao = sum(1 for p in padroes if p in nome_coluna)
                if pontuacao > melhor_pontuacao:
                    melhor_pontuacao = pontuacao
                    indice_manual = i
            if indice_manual is not None and melhor_pontuacao > 0:
                mapeamento[coluna_schema] = colunas_csv_bruto[indice_manual]
                continue

        candidatos = [colunas_normalizadas[i] for i in allowed_indices]

        melhor_correspondencia = process.extractOne(
            query,
            candidatos,
            scorer=fuzz.WRatio,
        )

        if melhor_correspondencia is None:
            mapeamento[coluna_schema] = None
            continue

        _, score, indice_local = melhor_correspondencia
        indice_global = allowed_indices[indice_local]

        if score < threshold:
            mapeamento[coluna_schema] = None
        else:
            mapeamento[coluna_schema] = colunas_csv_bruto[indice_global]

    return mapeamento


if __name__ == "__main__":
    colunas_csv1 = [
        "Campus", "Sede", "Turno", "Curso", "Sexo", "Situação",
        "Ano de ingresso", "Coeficiente de rendimento", "Cor ou raça",
        "Data de nascimento", "Disciplinas aprovadas",
        "Disciplinas reprovadas por frequência", "Disciplinas reprovadas por nota",
        "Forma de ingresso", "Idade", "Município", "Município SiSU",
        "Nota ENEM Humanas", "Nota ENEM Liguagem", "Nota ENEM Matemática",
        "Nota ENEM Natureza", "Nota ENEM Redação",
        "Nota Vestibular Biologia", "Nota Vestibular Matemática",
        "Período", "Semestre de ingresso no curso", "Tipo de cota",
        "Total de semestres cursados", "UF",
    ]

    colunas_csv2 = [
        "Campus", "Sede", "Curso", "Turno", "Gênero",
        "Ano de ingresso", "Coeficiente de rendimento absoluto",
        "Data de nascimento", "Forma de ingresso", "Idade",
        "Município", "Município (SISU)",
        "Nota ENEM Humanas", "Nota ENEM Liguagem", "Nota ENEM Matemática",
        "Nota ENEM Natureza", "Nota ENEM Redação",
        "Nr disciplinas aprovadas", "Nr disciplinas reprovadas por frequência",
        "Nr disciplinas reprovadas por nota",
        "Período", "Semestre de ingresso", "Situação atual", "Tipo de cota",
        "Total de semestres cursados", "UF", "UF (SISU)",
    ]

    print("=" * 70)
    print("CSV 1 — sugestões")
    print("=" * 70)
    for k, v in gerar_sugestao_mapeamento(colunas_csv1).items():
        print(f"  {k:50s} <- {v}")

    print("\n" + "=" * 70)
    print("CSV 2 — sugestões")
    print("=" * 70)
    for k, v in gerar_sugestao_mapeamento(colunas_csv2).items():
        print(f"  {k:50s} <- {v}")
