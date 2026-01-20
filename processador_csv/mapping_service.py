from typing import Dict, List, Optional

from unidecode import unidecode
from rapidfuzz import fuzz, process

from schemas import COLUNAS_WHITELIST


MANUAL_HINTS: Dict[str, List[str]] = {
    "coeficiente_rendimento": ["cr", "coeficiente", "rendimento", "ira"],
    "municipio_residencia": ["municipio"],
}

ALIASES_CONHECIDOS: Dict[str, str] = {
    "tipo de cota": "tipo_cota",
    "cor ou raca": "cor_raca",
    "raca": "cor_raca",
    "e-mail": "email",
    "curso": "curso",
    "nome do curso": "curso",
    "curso graduacao": "curso",
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
    "periodo": "periodo",
    "serie": "periodo",
    "semestre atual": "periodo",
    "genero": "sexo",
    "situacao": "situacao",
    "situacao atual": "situacao",
    "status": "situacao",
    "status do aluno": "situacao",
}


def gerar_sugestao_mapeamento(
    colunas_csv_bruto: List[str], threshold: int = 80
) -> Dict[str, Optional[str]]:
    colunas_normalizadas = [
        unidecode(c).strip().lower() for c in colunas_csv_bruto
    ]

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
                    i for i in indices_municipio if "sisu" not in colunas_normalizadas[i]
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
    colunas_exemplo = [
        "Campus",
        "Turno",
        "Curso",
        "Sexo",
        "Ano de ingresso",
        "Nota ENEM Matemática",
        "Tipo de cota",
        "Cor ou raça",
    ]

    sugestao = gerar_sugestao_mapeamento(colunas_exemplo)
    print("Sugestão para cor_raca:", sugestao.get("cor_raca"))
