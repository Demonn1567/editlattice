"""Supplementary material for the Knowledge-Based Systems submission (paper/kbs_supp.tex).

The body is the appendix of paper/main.tex, taken verbatim between \\appendix and \\end{document}, with these changes:
references to appendices become references to supplementary sections, assumptions are cited by number, the three
tables that the journal version moves out of the main text (gain pilots, branched lineages, pooled recombination) are
added to the full results, and the reproducibility section points to the data repository. Cross-references to the
main paper resolve through the xr package against kbs.aux, so kbs.tex must be compiled first.

Usage: python scripts/kbs_supp.py   (then latexmk -pdf kbs.tex kbs_supp.tex)
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[a4paper,textwidth=6.5in,textheight=9.4in]{geometry}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{longtable}
\usepackage{array}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{xr-hyper}
\usepackage[hidelinks]{hyperref}
\usepackage{url}
\externaldocument{kbs}
\lstset{basicstyle=\scriptsize\ttfamily, breaklines=true, columns=fullflexible, frame=single, rulecolor=\color{black!40},
  keepspaces=true, showstringspaces=false}
\setlength{\emergencystretch}{3em}
\setlength{\tabcolsep}{4.5pt}
\input{math_commands.tex}
\input{macros.tex}
\newtheorem{lemma}{Lemma}
\renewcommand{\thelemma}{S\arabic{lemma}}
\renewcommand{\thesection}{S\arabic{section}}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\AtBeginDocument{\renewcommand{\thelstlisting}{S\arabic{lstlisting}}}

\title{Supplementary material for\\``Which self-made edits matter? Exact attribution for self-improving language-model
agents through variational replay''}
\author{Krish Sharma and Jatin Bedi\\[2pt]\normalsize Department of Computer Science and Engineering\\\normalsize Thapar
Institute of Engineering and Technology, Patiala, India}
\date{}

\begin{document}
\maketitle
\noindent Numbers prefixed with S refer to this supplement; all other section, table, figure, lemma, proposition and
assumption numbers refer to the main paper.
"""

EXTRA_TABLES = r"""Tables~\ref{tab:pilots}, \ref{tab:branched} and~\ref{tab:pools} give the gain pilots of every phase, the branched
lineages and the recombination over pooled lattices, which the main text summarises.

\input{tables/pilots.tex}
\input{tables/branched.tex}
\input{tables/pools.tex}
"""


def body() -> str:
    main = (ROOT / "paper" / "main.tex").read_text()
    start = main.index("\\appendix") + len("\\appendix")
    end = main.index("\\end{document}")
    text = main[start:end]
    replacements = [
        ("Appendix~\\ref{", "Section~\\ref{"),
        ("Appendices~\\ref{", "Sections~\\ref{"),
        ("sec:productline", "sec:model"),
        ("The supplementary material contains the code, the configurations, the registered plan, the decision log\n"
         "and every result file the paper reads;",
         "The data repository (see the data availability statement of the main paper) contains the code, the\n"
         "configurations, the registered plan, the decision log and every result file the paper reads;"),
        ("the appendix tables and listings", "the supplementary tables and listings"),
        ("This appendix lists every change", "This section lists every change"),
        ("we recomputed RQ2, and no conclusion changed", "we recomputed the comparison of Shapley estimators, and no\nconclusion changed"),
        ("give the values behind RQ2, the planted transfer", "give the values behind the comparison of Shapley estimators, the planted transfer"),
    ]
    for old, new in replacements:
        if old in ("Appendices~\\ref{",):
            text = text.replace(old, new)
            continue
        assert old in text, old
        text = text.replace(old, new)
    text = re.sub(r"\bA\\ref\{(a:[^}]*)\}", r"Assumption~\\ref{\1}", text)
    text = re.sub(r"A\\ref\{(a:[^}]*)\}", r"Assumption~\\ref{\1}", text)
    anchor = "\\input{appendix/all_loops.tex}"
    assert text.count(anchor) == 1
    text = text.replace(anchor, EXTRA_TABLES + "\n" + anchor)
    assert "\\cite" not in text, "the supplement has no bibliography"
    return text


def main() -> None:
    (ROOT / "paper" / "kbs_supp.tex").write_text(PREAMBLE + body() + "\\end{document}\n")
    print("wrote paper/kbs_supp.tex")


if __name__ == "__main__":
    main()
