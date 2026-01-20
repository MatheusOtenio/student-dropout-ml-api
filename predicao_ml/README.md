# Cross-validation do Modelo de Evasão

Este documento explica, de forma didática, como o código realiza a validação cruzada (cross-validation) ao treinar o modelo de predição de evasão.

---

## 1. Como o código decide quantos folds usar

No `_cross_validate`:

```python
min_samples = int(y.value_counts().min())
requested_splits = int(n_splits)
actual_splits = min(requested_splits, min_samples)
```

- `n_splits` vem da configuração (`config["cv_splits"]`, ex.: `5`).
- `y.value_counts().min()` pega o **número mínimo de alunos em qualquer uma das classes** (`0` ou `1`).
- `actual_splits` é o número real de folds usados:
  - Se você pediu `5` folds, mas a classe minoritária só tem `3` alunos, faz **3 folds**.
  - Se você tem bastante dados nas duas classes, usa o valor pedido (ex.: `5`).
- Se `actual_splits < 2`, o código trata como caso especial (treina sem CV), mas assumindo que você tem dados suficientes, consideramos o caso normal (2 ou mais folds).

---

## 2. Em números e porcentagens: quanto é treino e quanto é validação

Com `StratifiedKFold`:

```python
skf = StratifiedKFold(
    n_splits=actual_splits,
    shuffle=True,
    random_state=random_state,
)
```

Em cada *fold*:

- Aproximadamente `1 / n_splits` dos alunos vão para **validação**.
- Aproximadamente `(n_splits − 1) / n_splits` vão para **treino**.

Se `n_splits = 5` (caso típico):

- Em cada rodada:
  - ~20% dos alunos → validação  
  - ~80% dos alunos → treino

### Exemplo 1: 1000 alunos no CSV

Suponha:

- 1000 linhas válidas (após descartar `situacao` desconhecida).
- `cv_splits = 5`.
- Cada classe (`0` e `1`) tem pelo menos 5 exemplos.

Então:

- `actual_splits = min(5, min_samples)` → digamos que vira `5`.
- Em cada *fold*:
  - Treino ≈ `1000 × (4/5) = 800` alunos.
  - Validação ≈ `1000 × (1/5) = 200` alunos.

E isso é estratificado:

- A proporção de **evasão** e **não evasão** fica parecida em treino e validação.

### Exemplo 2: 300 alunos

- 300 linhas válidas.
- `cv_splits = 5` e `min_samples` das classes é ≥ 5.

Então por *fold*:

- Treino ≈ `300 × 80% = 240` alunos.  
- Validação ≈ `300 × 20% = 60` alunos.

Se a classe minoritária tiver, por exemplo, só `3` alunos:

- `min_samples = 3`.
- `actual_splits = min(5, 3) = 3`.

Então por *fold*:

- ~66% treino, ~33% validação.

---

## 3. Como todos os alunos recebem predição de validação (out-of-fold)

No `_cross_validate`, o fluxo é:

```python
oof_probs = np.zeros(len(y), dtype=float)

for train_idx, val_idx in skf.split(X, y):
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_val = X.iloc[val_idx]

    # Preprocessamento e treino no grupo de treino
    ...

    proba = fold_model.predict_proba(X_val_proc)[:, 1]
    oof_probs[val_idx] = proba
```

- `skf.split(X, y)` gera `actual_splits` partições.
- Em cada partição:
  - `train_idx`: índices usados para treinar o modelo daquele *fold*.
  - `val_idx`: índices usados para **validar** (o modelo nunca viu esses alunos nesse *fold*).

O ponto importante:

- Ao longo de todos os *folds*, cada aluno entra em:
  - **Treino** em `n_splits - 1` *folds*.
  - **Validação** em **exatamente 1** *fold*.

Isso quer dizer:

- Sim, o sistema “volta” e troca quem é treino e quem é validação em cada rodada:
  - No *Fold 1*, o aluno 1 pode estar na validação, e nos outros 4 *folds* ele é usado como treino.
  - No *Fold 2*, o aluno 2 pode estar na validação, e assim por diante.
- O vetor `oof_probs` guarda, para cada aluno, a **probabilidade predita quando ele estava no grupo de validação**, ou seja, uma predição que não foi “contaminada” por treino com aquele mesmo aluno.

---

## Resumo em linguagem simples

- Partindo de um CSV completo depois da limpeza (removendo linhas com `situacao` inválida):
  - Gera-se uma base `X` (features) e `y` (evasão 0/1).
- Com `cv_splits = 5`, em cada rodada:
  - Aproximadamente **80%** dos alunos são usados para **treinar**.
  - Aproximadamente **20%** são usados para **validar**.
- O sistema repete esse processo 5 vezes, cada vez com um subconjunto diferente na validação.

No fim:

- Todo aluno é visto como **validação** exatamente **uma vez**.  
- Todo aluno é visto como **treino** em `n_splits - 1` rodadas.

