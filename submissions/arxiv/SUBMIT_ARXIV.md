# arXiv — Submissão (pode precisar de endorsement)

## URL: https://arxiv.org/submit

## Verificação prévia: você precisa de endorser?

### Passo 1: Criar conta
- Acessar https://arxiv.org/user/register
- Preencher dados (nome, email, instituição: "Independent Researcher")
- Confirmar email

### Passo 2: Verificar se precisa de endorsement
- Após login, tentar iniciar submissão
- Selecionar categoria: **cs.AI** (Artificial Intelligence) — RECOMENDADO
  - cs.AI é mais aberto para first-time submitters que cs.IR
  - O paper se encaixa perfeitamente
- Cross-list para: cs.IR, cs.CL
- Se o sistema pedir endorsement:

### Opção A: Pedir endorsement
O arXiv tem um sistema formal. Após criar conta:
1. Acessar https://arxiv.org/auth/endorse
2. O sistema mostra um link para solicitar endorsement
3. Enviar o link para um pesquisador que já publicou em cs.AI
4. Sugestões de quem pedir:
   - Qualquer autor de paper citado no seu paper que tenha arXiv ID
   - Pesquisadores brasileiros em AI (ICMC-USP, UNICAMP, UFMG têm vários)
   - Via Twitter/LinkedIn: "I'm submitting a paper on semantic steering for retrieval, would you be willing to endorse my arXiv submission?"

### Opção B: Submeter sem endorsement
- Alguns novos usuários são auto-aprovados pelo sistema
- Depende do conteúdo e da categoria
- Vale tentar submeter diretamente — o pior que acontece é pedir endorsement depois

## Se conseguir submeter:

### Passo 3: Preencher metadados

**Title:**
```
STEER: Semantic Transformation for Embedding-space Exploration in Retrieval -- An Honest Evaluation of Sixteen Algebraic Operations
```

**Authors:**
```
Renato Aparecido Gomes
```

**Abstract:** (mesmo texto)

**Primary category:** cs.AI
**Cross-list:** cs.IR, cs.CL

**Comments:**
```
27 pages, 22 tables, 1 algorithm. Code: https://github.com/RAG7782/steer
```

**MSC-class:** 68T50
**ACM-class:** H.3.3

### Passo 4: Upload source
- Upload do `paper.tex` (arXiv compila LaTeX)
- OU upload do `paper.pdf` (menos preferido mas aceito)

### Passo 5: Preview e Submit
- arXiv gera preview do PDF
- Verificar formatação
- Confirmar submissão
- Moderação: 1-2 dias úteis
- ID atribuído: arXiv:26XX.XXXXX

## Tempo estimado: 15-20 minutos (se não precisar de endorsement)
## Disponibilização: 1-2 dias úteis após aprovação
## Contingência: Se endorsement for necessário, usar Zenodo + SSRN + TechRxiv enquanto resolve

## Nota sobre categorias
- cs.IR (Information Retrieval): mais específico, mais visível para o público-alvo, MAS mais restrito para endorsement
- cs.AI (Artificial Intelligence): mais amplo, mais fácil de entrar, boa visibilidade
- cs.CL (Computation and Language): alternativa se cs.AI não funcionar
- cs.LG (Machine Learning): última opção, muito genérico mas aceita quase tudo
