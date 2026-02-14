
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.target_engineering import generate_future_dropout_target

def test_dropout_next_semester():
    print("Testing dropout next semester...")
    df = pd.DataFrame({
        "codigo_aluno": ["A", "A"],
        "ano_referencia": [2020, 2020],
        "periodo_referencia": [1, 2],
        "situacao": ["Regular", "Desistente"]
    })
    
    res = generate_future_dropout_target(df, horizon=2)
    
    row1 = res.iloc[0]
    assert row1["target_evasao"] == 1.0, f"Expected 1.0, got {row1['target_evasao']}"
    assert row1["is_target_valid"] == True, "Expected True"
    
    row2 = res.iloc[1]
    assert row2["is_target_valid"] == False, "Expected False (terminal)"
    print("PASS")

def test_dropout_within_horizon():
    print("Testing dropout within horizon...")
    df = pd.DataFrame({
        "codigo_aluno": ["B", "B", "B"],
        "ano_referencia": [2020, 2020, 2021],
        "periodo_referencia": [1, 2, 1],
        "situacao": ["Regular", "Regular", "Trancado"]
    })
    
    res = generate_future_dropout_target(df, horizon=2)
    
    assert res.iloc[0]["target_evasao"] == 1.0, "T=1 fail"
    assert res.iloc[1]["target_evasao"] == 1.0, "T=2 fail"
    print("PASS")

def test_graduation_safe():
    print("Testing graduation...")
    df = pd.DataFrame({
        "codigo_aluno": ["C", "C", "C"],
        "ano_referencia": [2020, 2020, 2021],
        "periodo_referencia": [1, 2, 1],
        "situacao": ["Regular", "Regular", "Formado"]
    })
    
    res = generate_future_dropout_target(df, horizon=2)
    
    assert res.iloc[0]["target_evasao"] == 0.0, "Graduation should be 0"
    print("PASS")

def test_multiple_students_isolation():
    print("Testing student isolation...")
    df = pd.DataFrame({
        "codigo_aluno": ["A", "B"],
        "ano_referencia": [2020, 2020],
        "periodo_referencia": [1, 1],
        "situacao": ["Regular", "Desistente"]
    })
    
    res = generate_future_dropout_target(df, horizon=2)
    
    val = res.iloc[0]["target_evasao"]
    assert val == 0.0, f"Leaking future from B to A. Expected 0.0, got {val}"
    print("PASS")

if __name__ == "__main__":
    try:
        test_dropout_next_semester()
        test_dropout_within_horizon()
        test_graduation_safe()
        test_multiple_students_isolation()
        print("\nAll target engineering tests passed!")
    except Exception as e:
        print(f"\nFAIL: {e}")
        sys.exit(1)
