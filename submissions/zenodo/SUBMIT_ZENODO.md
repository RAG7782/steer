# Zenodo — Submissão Direta (DOI instantâneo)

## URL: https://zenodo.org/deposit/new

## Passo a passo

### 1. Login
- Acessar https://zenodo.org
- Login com GitHub (botão "Log in with GitHub") — mais rápido
- Ou criar conta com email

### 2. New Upload
- Clicar "New Upload" (botão verde no canto superior direito)
- Arrastar o arquivo `paper.pdf` para a área de upload

### 3. Preencher metadados

**Upload type:** Publication
**Publication type:** Preprint

**Title:**
```
STEER: Semantic Transformation for Embedding-space Exploration in Retrieval — An Honest Evaluation of Sixteen Algebraic Operations
```

**Authors:**
```
Gomes, Renato Aparecido — Independent Researcher — ORCID: (preencher se tiver)
```

**Description (copiar):**
```
What happens when you algebraically transform a query embedding before retrieval? We investigate this question systematically, evaluating sixteen operations across six sentence transformer architectures and five BEIR benchmarks. We present STEER (Semantic Transformation for Embedding-space Exploration in Retrieval), an adaptive stack that composes isotropy correction, adaptive alpha, and multi-vector fusion to enable controllable semantic navigation in embedding spaces without retraining. Key findings: contrastive steer achieves +3.5% nDCG, rotate-away provides +3.6% as bias correction, and an automatic operation router with per-query LLM targets achieves F1=0.91 on biomedical benchmarks. The contribution is both an honest empirical map of what algebraic steering can and cannot do, and a practical system that makes it deployable.
```

**Keywords (separar por vírgula):**
```
semantic steering, embedding manipulation, information retrieval, dense retrieval, vector arithmetic, STEER, retrieval-augmented generation, adaptive retrieval
```

**License:** Creative Commons Attribution 4.0 International (CC-BY-4.0)

**Related identifiers:** (deixar vazio por enquanto — adicionar arXiv ID depois)

**Grants:** (deixar vazio)

### 4. Publicar
- Clicar "Preview"
- Verificar tudo
- Clicar "Publish"
- **DOI será gerado IMEDIATAMENTE** (formato: 10.5281/zenodo.XXXXXXX)

### 5. Após publicação
- Copiar DOI
- Adicionar ao README do pip package
- Adicionar aos emails de outreach
- Adicionar ao paper (pode atualizar no Zenodo sem perder DOI)

## Tempo estimado: 5 minutos
