#!/usr/bin/env python3
"""
atualiza_kanban.py — Atualiza linhas do KANBAN.md por ID, com segurança.

Existe porque uma edição anterior deste board usou `re.sub(..., flags=re.S)` com
um grupo final `(\\|.*\\n)+`: sob DOTALL o `.` casa quebra de linha, o grupo
engoliu o resto do arquivo, e o KANBAN caiu de 321 para 51 linhas. O erro passou
por dois commits porque `git add -A` levou junto e o diff não foi conferido.

Regras aqui: substituição **literal** por linha, nunca regex multilinha; e uma
verificação de integridade que aborta se o número de tarefas mudar.
"""

import re
import sys
from collections import defaultdict

ARQ = "KANBAN.md"
LINHA_TAREFA = re.compile(r"^\| (KB-[A-F]\d+) \|", re.M)


def carrega():
    with open(ARQ, encoding="utf-8") as fh:
        return fh.read()


def salva(texto, n_antes):
    n_depois = len(LINHA_TAREFA.findall(texto))
    if n_depois != n_antes:
        sys.exit(f"ABORTADO: tarefas {n_antes} -> {n_depois}. "
                 f"Uma atualização de conteúdo nunca deve mudar a contagem.")
    with open(ARQ, "w", encoding="utf-8") as fh:
        fh.write(texto)


def atualiza(texto, tid, *, tarefa=None, evidencia=None, conf=None, status=None):
    """Reescreve campos de UMA linha, preservando os demais."""
    alvo = next((l for l in texto.split("\n")
                 if l.startswith(f"| {tid} |")), None)
    if alvo is None:
        sys.exit(f"ABORTADO: {tid} não encontrado")
    c = alvo.split("|")
    if len(c) < 8:
        sys.exit(f"ABORTADO: {tid} tem {len(c)} campos, esperados 8")
    if tarefa is not None:
        c[3] = f" {tarefa} "
    if evidencia is not None:
        c[4] = f" {evidencia} "
    if conf is not None:
        c[5] = f" {conf} "
    if status is not None:
        c[6] = f" `{status}` "
    return texto.replace(alvo, "|".join(c))


def recontagem(texto):
    """Regenera a tabela-resumo a partir das linhas reais."""
    d = defaultdict(lambda: defaultdict(int))
    conf = {"✅": 0, "🟡": 0, "⬜": 0}
    for l in texto.split("\n"):
        if not LINHA_TAREFA.match(l):
            continue
        campos = l.split("|")
        d[campos[1].strip()[3]][campos[2].strip()] += 1
        glifos = [ch for ch in l if ch in conf]
        if glifos:
            conf[glifos[-1]] += 1

    nomes = {"A": "A — Cobertura da auditoria", "B": "B — Núcleo & CI",
             "C": "C — Propriedades", "D": "D — Evidências & artigo",
             "E": "E — IC3/PDR", "F": "F — Higiene"}
    linhas, tot = [], [0, 0, 0]
    for k in "ABCDEF":
        p = [d[k][x] for x in ("P0", "P1", "P2")]
        tot = [a + b for a, b in zip(tot, p)]
        fmt = lambda n: str(n) if n else "—"
        linhas.append(f"| {nomes[k]} | {fmt(p[0])} | {fmt(p[1])} | "
                      f"{fmt(p[2])} | {sum(p)} |")
    linhas.append(f"| **Total** | **{tot[0]}** | **{tot[1]}** | "
                  f"**{tot[2]}** | **{sum(tot)}** |")

    # substituicao delimitada linha a linha, sem DOTALL
    saida, dentro = [], False
    for l in texto.split("\n"):
        if l.startswith("| A — "):
            dentro = True
            saida.extend(linhas)
            continue
        if dentro:
            if l.startswith("| **Total**"):
                dentro = False
            continue
        saida.append(l)
    texto = "\n".join(saida)

    texto = texto.replace(
        re.search(r"Confiança: \*\*\d+ ✅[^\n]*", texto).group(0),
        f"Confiança: **{conf['✅']} ✅ verificado** · "
        f"**{conf['🟡']} 🟡 relatado por agente** · "
        f"**{conf['⬜']} ⬜ não auditado**")
    m = re.search(r"É o que separa os \d+ ✅ dos \d+ 🟡\.", texto)
    if m:
        texto = texto.replace(
            m.group(0), f"É o que separa os {conf['✅']} ✅ dos {conf['🟡']} 🟡.")
    return texto, conf, sum(tot)
