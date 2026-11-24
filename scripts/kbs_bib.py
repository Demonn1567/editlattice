"""Bibliography for the Knowledge-Based Systems manuscript, derived from the verified paper/refs.bib.

KBS asks that preprints be marked as such and given a DOI, and that references carry DOIs where available. This script
rewrites every arXiv entry as a preprint with its arXiv DOI (10.48550/arXiv.<id>), drops URL fields that duplicate a DOI,
and gives the Shapley (1953) chapter its book title from Crossref. Entries in paper/refs_kbs_extra.bib, each checked
against the source named in its comment, are then added, replacing any entry of refs.bib with the same key.

Usage: python scripts/kbs_bib.py   (writes paper/refs_kbs.bib)
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def entries(text: str) -> list[str]:
    """Split a .bib file into entries by brace matching."""
    out, i = [], 0
    while (start := text.find("@", i)) >= 0:
        depth, j = 0, text.index("{", start)
        while True:
            depth += text[j] == "{"
            depth -= text[j] == "}"
            j += 1
            if depth == 0:
                break
        out.append(text[start:j])
        i = j
    return out


def drop_field(entry: str, name: str) -> str:
    """Remove one field, whatever its brace nesting."""
    m = re.search(rf",?\s*\b{name}\s*=\s*", entry, re.I)
    if not m:
        return entry
    k = m.end()
    if entry[k] == "{":
        depth, q = 0, k
        while True:
            depth += entry[q] == "{"
            depth -= entry[q] == "}"
            q += 1
            if depth == 0:
                break
    else:
        q = k
        while q < len(entry) and entry[q] not in ",}":
            q += 1
    return entry[:m.start()] + entry[q:]


def add_field(entry: str, name: str, value: str) -> str:
    """Append a field before the closing brace."""
    body = entry.rstrip()
    assert body.endswith("}")
    return body[:-1].rstrip().rstrip(",") + f",\n  {name} = {{{value}}}\n}}"


PROPER = {"Shapley", "Meta-Harness", "Self-Harness", "Qwen3", "Möbius"}


def protect(word: str) -> str:
    """Brace a word whose capitals must survive sentence casing: acronyms, CamelCase names and proper nouns."""
    core = word.strip(":?,.")
    inner_caps = any(ch.isupper() for segment in core.split("-") for ch in segment[1:])
    if core in PROPER or inner_caps:
        return word.replace(core, "{" + core + "}", 1)
    return word


def sentence_case_title(entry: str) -> str:
    """Turn a brace-protected Title Case title into one that the Elsevier style sets in sentence case."""
    m = re.search(r"(\btitle\s*=\s*)\{\{([^{}]*)\}\}", entry) or re.search(r"(\btitle\s*=\s*)\{([^{}]*)\}", entry)
    if not m:
        return entry
    words = m.group(2).split(" ")
    out = []
    for i, w in enumerate(words):
        w = protect(w)
        if i and words[i - 1].endswith(("?", "!")) and w[:1].isupper():
            w = "{" + w[0] + "}" + w[1:]
        out.append(w)
    title = " ".join(out)
    return entry[:m.start()] + m.group(1) + "{" + title + "}" + entry[m.end():]


def convert(entry: str) -> str:
    entry = sentence_case_title(entry)
    eprint = re.search(r"\beprint\s*=\s*\{([^}]*)\}", entry)
    if eprint:
        arxiv_id = eprint.group(1).strip()
        for name in ("eprint", "archivePrefix", "primaryClass", "url", "journal", "howpublished", "note", "doi"):
            entry = drop_field(entry, name)
        entry = add_field(entry, "howpublished", f"Preprint, arXiv:{arxiv_id}")
        entry = add_field(entry, "doi", f"10.48550/arXiv.{arxiv_id}")
        return entry
    if re.search(r"\bdoi\s*=", entry, re.I):
        entry = drop_field(entry, "url")
    if entry.lstrip().startswith("@inbook{shapley1953") or "{shapley1953," in entry:
        entry = re.sub(r"title\s*=\s*\{17\. A Value for n-Person Games\}", "title={A value for n-person games}", entry)
        entry = entry.replace("@inbook{shapley1953", "@incollection{shapley1953", 1)
        if "booktitle" not in entry:
            entry = add_field(entry, "booktitle", "Contributions to the Theory of Games (AM-28), Volume II")
    return entry


def key(entry: str) -> str:
    return re.match(r"@\w+\s*\{\s*([^,\s]+)", entry).group(1)


def main() -> None:
    source = (ROOT / "paper" / "refs.bib").read_text()
    converted = [convert(e) for e in entries(source)]
    assert len(converted) == len(entries(source))
    extra_text = re.sub(r"(?m)^%.*$", "", (ROOT / "paper" / "refs_kbs_extra.bib").read_text())
    extras = {key(e): e for e in entries(extra_text)}
    keys = [key(e) for e in converted]
    upgraded = [k for k in keys if k in extras]
    merged = [extras.get(key(e), e) for e in converted] + [e for k, e in extras.items() if k not in keys]
    assert len({key(e) for e in merged}) == len(merged), "duplicate keys"
    (ROOT / "paper" / "refs_kbs.bib").write_text("\n\n".join(merged) + "\n")
    print(f"{len(merged)} entries written to paper/refs_kbs.bib; upgraded: {', '.join(upgraded)}")


if __name__ == "__main__":
    main()
