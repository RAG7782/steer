# A2RAG — Resultados das 3 Perguntas Abertas

Data: 2026-03-27

## Pergunta 2: Rotacionar o corpus em vez da query? (Dual-Index)

**Hipotese:** Criar D' = {T(d) for d in D} e buscar q em D e D' separadamente. Merge.
Se funcionar, elimina o trade-off completamente.

**Status:** COMPLETO (6 modelos x 2 datasets x 4 alphas)

### Resultado Principal

O dual-index **preserva nDCG@10 perfeitamente** (delta = 0.0000 em todos os casos),
mas **nao traz ganho novo** sobre o baseline.

| Modelo | Dataset | Base nDCG@10 | QueryRot Δ | DualOrig Δ | DualMixed Δ | NewDocs avg |
|--------|---------|:----:|:----:|:----:|:----:|:----:|
| all-MiniLM-L6-v2 | scifact | 0.6451 | +0.0040 | +0.0000 | +0.0000 | 2.1 |
| BAAI/bge-small-en-v1.5 | scifact | 0.7200 | -0.0077 | +0.0000 | +0.0000 | 4.4 |
| all-mpnet-base-v2 | scifact | 0.6557 | +0.0016 | +0.0000 | +0.0000 | 2.1 |
| BAAI/bge-base-en-v1.5 | scifact | 0.7376 | -0.0081 | +0.0000 | +0.0000 | 3.8 |
| intfloat/e5-small-v2 | scifact | 0.6796 | -0.0003 | +0.0000 | -0.0001 | 5.6 |
| thenlper/gte-small | scifact | 0.7270 | -0.0042 | +0.0000 | +0.0000 | 4.8 |
| all-MiniLM-L6-v2 | arguana | 0.5017 | +0.0003 | +0.0000 | +0.0000 | 1.9 |
| BAAI/bge-small-en-v1.5 | arguana | 0.5950 | -0.0026 | +0.0000 | +0.0000 | 4.6 |
| all-mpnet-base-v2 | arguana | 0.4652 | +0.0024 | +0.0000 | +0.0000 | 2.1 |
| BAAI/bge-base-en-v1.5 | arguana | 0.6362 | -0.0048 | +0.0000 | +0.0000 | 3.9 |
| intfloat/e5-small-v2 | arguana | 0.4687 | +0.0007 | +0.0000 | +0.0000 | 4.5 |
| thenlper/gte-small | arguana | 0.5527 | +0.0008 | +0.0000 | +0.0000 | 4.4 |

### Insights
- O rotated index surfaca 2-10 docs novos por query, mas eles nao sao relevantes segundo qrels BEIR.
- **Dual-index e um "safe fallback"**: zero degradacao, mas zero ganho.
- Implicacao para o paper: mencionar como "alternativa de producao" (sem trade-off) mas sem claim de melhoria.

---

## Pergunta 3: Addition + Scaled Subtraction composta

**Hipotese:** T(q) = normalize(q + alpha*target - beta*proj_exclude(q)) combina
"puxar" (addition) + "empurrar" (subtraction) de forma mais potente que cada isolada.

**Status:** COMPLETO (6 modelos x 2 datasets x 3 alphas x 4 betas = 144 configs)

### Melhor config por modelo/dataset

| Modelo | DS | Base | Add(a=0.1) | Best Composed | Config | Δ best |
|--------|-----|:----:|:----:|:----:|--------|:----:|
| all-MiniLM-L6-v2 | scifact | 0.6451 | 0.6491 | **0.6570** | α=0.2 β=0.75 | **+0.0119** |
| BAAI/bge-small-en-v1.5 | scifact | 0.7200 | 0.7123 | 0.7176 | α=0.05 β=0.0 | -0.0024 |
| all-mpnet-base-v2 | scifact | 0.6557 | 0.6573 | 0.6573 | α=0.1 β=0.0 | +0.0016 |
| BAAI/bge-base-en-v1.5 | scifact | 0.7376 | 0.7295 | **0.7423** | α=0.05 β=0.5 | **+0.0047** |
| intfloat/e5-small-v2 | scifact | 0.6796 | 0.6793 | 0.6793 | α=0.1 β=0.0 | -0.0003 |
| thenlper/gte-small | scifact | 0.7270 | 0.7228 | 0.7264 | α=0.05 β=0.0 | -0.0006 |
| all-MiniLM-L6-v2 | arguana | 0.5017 | 0.5020 | 0.5023 | α=0.2 β=0.0 | +0.0006 |
| BAAI/bge-small-en-v1.5 | arguana | 0.5950 | 0.5924 | **0.5996** | α=0.05 β=0.25 | **+0.0046** |
| all-mpnet-base-v2 | arguana | 0.4652 | 0.4676 | **0.4682** | α=0.2 β=0.75 | **+0.0030** |
| BAAI/bge-base-en-v1.5 | arguana | 0.6362 | 0.6314 | 0.6347 | α=0.05 β=0.0 | -0.0015 |
| intfloat/e5-small-v2 | arguana | 0.4687 | 0.4694 | 0.4694 | α=0.1 β=0.0 | +0.0007 |
| thenlper/gte-small | arguana | 0.5527 | 0.5535 | 0.5537 | α=0.05 β=0.0 | +0.0010 |

### Insights
- **Distilled (MiniLM):** β agressivo (0.75) funciona bem. Parece "limpar" a query antes de adicionar target.
- **Contrastive (bge):** β moderado (0.25-0.5) e melhor. O treinamento contrastivo ja cria embeddings "limpos".
- **Instruction-tuned (e5) e general (gte):** Subtraction nao ajuda. A composicao nao supera addition-only.
- Subtraction-only sempre degrada nDCG (ate -0.0715 para gte-small/scifact), confirmando que e melhor como "limpeza" combinada com addition.
- Implicacao: operacao composta tem valor, mas **e model-dependent**. Nao e universal.

---

## Pergunta 1: Addition com Reranker elimina o trade-off?

**Hipotese:** Addition amplia recall mas pode degradar nDCG. Cross-encoder reranker filtra o ruido.

**Status:** COMPLETO (timeout no alpha=0.3 arguana, mas dados suficientes capturados)

### Resultados scifact (6 modelos completos)

| Modelo | Base | Base+RR | Add(0.1) | Add+RR | Merged+RR |
|--------|:----:|:----:|:----:|:----:|:----:|
| all-MiniLM-L6-v2 | 0.6451 | 0.6892 | 0.6491 | **0.6889** | 0.6886 |
| BAAI/bge-small-en-v1.5 | 0.7200 | 0.7002 | 0.7123 | 0.6938 | 0.6937 |
| all-mpnet-base-v2 | 0.6557 | 0.6956 | 0.6573 | 0.6955 | 0.6954 |
| BAAI/bge-base-en-v1.5 | 0.7376 | 0.7022 | 0.7295 | 0.6950 | 0.6948 |
| intfloat/e5-small-v2 | 0.6796 | 0.7037 | 0.6793 | 0.7037 | 0.7022 |
| thenlper/gte-small | 0.7270 | 0.6950 | 0.7228 | 0.6955 | 0.6954 |

### Resultados arguana (6 modelos, alpha 0.1-0.2 capturados antes do timeout)

Cross-encoder ms-marco-MiniLM **DEGRADA** arguana massivamente em todos os modelos:

| Modelo | Base | Base+RR | Add(0.1)+RR | Merged(0.1)+RR | Add(0.2)+RR |
|--------|:----:|:----:|:----:|:----:|:----:|
| all-MiniLM-L6-v2 | 0.5017 | 0.4197 | 0.4185 | 0.4181 | 0.4200 |
| BAAI/bge-small-en-v1.5 | 0.5950 | 0.4190 | 0.4214 | 0.4192 | 0.4242 |
| all-mpnet-base-v2 | 0.4652 | 0.4279 | 0.4279 | 0.4273 | 0.4270 |
| BAAI/bge-base-en-v1.5 | 0.6362 | 0.4153 | 0.4203 | 0.4185 | 0.4235 |
| intfloat/e5-small-v2 | 0.4687 | 0.4125 | 0.4125 | 0.4120 | 0.4113 |
| thenlper/gte-small | 0.5527 | 0.4188 | 0.4142 | 0.4141 | 0.4142 |

**Nota:** Job expirou (timeout 5400s) durante alpha=0.3 no arguana. O cross-encoder no arguana
(1406 queries x ~106 candidatos x 7 rounds de reranking) excedeu 90 minutos.

### Insights
- **scifact:** Reranker ajuda modelos distilled (MiniLM: +0.0438, mpnet: +0.0398) e e5 (+0.0241).
  Mas DEGRADA modelos contrastive (bge-small: -0.0262, bge-base: -0.0427, gte: -0.0315).
- **arguana:** ms-marco cross-encoder NAO e calibrado para argumentacao. Degrada TODOS os modelos
  por -0.04 a -0.22 pontos. Catastrophic domain mismatch.
- **Addition+Reranker vs Baseline+Reranker:** Diferenca minima (~0.002). O reranker domina o efeito.
- **Merged+Reranker = Add+Reranker:** Ampliar o pool de candidatos nao ajuda o cross-encoder.
- Implicacao: reranker so funciona para **domain-matched scenarios**.
  O gain vem do reranker, nao da addition. Resultado negativo claro para a hipotese.

---

## Conclusoes Consolidadas

| Pergunta | Hipotese | Resultado | Implicacao para o paper |
|----------|----------|-----------|------------------------|
| P2: Corpus rotacionado | Dual-index elimina trade-off | **Parcialmente confirmada**: zero degradacao, mas zero ganho | Mencionar como "alternativa de producao" (safe, mas nao melhor) |
| P3: Add+Sub composta | Composicao > operacoes isoladas | **Model-dependent**: funciona para distilled, nao para instruction-tuned | Reportar como "exploration", com caveat de generalidade |
| P1: Addition+Reranker | Reranker filtra ruido da rotation | **Negativo**: gain vem do reranker, nao da addition. Domain-mismatch anula tudo | Resultado negativo honesto — nao incluir como "melhoria" |

### Achado mais importante
Nenhuma das 3 abordagens e uma "bala de prata". O A2RAG funciona melhor como
**ferramenta de exploracao semantica controlavel** (o argumento original do paper)
do que como booster de nDCG. As operacoes algebricas permitem steering semantico
preciso, mas o ganho quantitativo em benchmarks e inerentemente limitado quando
os qrels nao premiam cross-domain retrieval.

---

## Scripts criados
- `modal_rotated_corpus.py` — Pergunta 2 (dual-index)
- `modal_reranker_test.py` — Pergunta 1 (addition + cross-encoder)
- `modal_composed_addition.py` — Pergunta 3 (addition + scaled subtraction)

## Resultados salvos
- Volume Modal `a2rag-results`: `/rotated_corpus/`, `/composed_addition/`
- Reranker: resultados nao salvos ao volume (timeout antes do vol.commit). Dados completos capturados dos logs.
- Local: `~/results_download/rotated_corpus/`, `~/results_download/composed_addition/`
