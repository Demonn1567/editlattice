"""Check every entry of a BibTeX file against its source, and report preprints that now have a published version.

For an entry with a DOI the record is fetched from Crossref (arXiv DOIs from the arXiv API). The entry passes when the
title matches the source (similarity at least 0.9), the first author's family name appears among the source's
authors, and the year differs by at most one. For every arXiv entry, OpenAlex is asked whether the same title exists as
a conference paper or journal article, which KBS asks us to cite instead of the preprint.

Usage: python scripts/verify_refs.py paper/refs_kbs.bib [--keys k1,k2]   (prints one line per entry)
"""

import argparse
import difflib
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kbs_bib import entries  # noqa: E402

HEADERS = {"User-Agent": "editlattice-refcheck/1.0"}


def get(url: str) -> bytes:
    error = None
    for attempt in range(6):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60).read()
        except Exception as exc:  # retried: rate limits and transient network failures
            error = exc
            time.sleep(30 * (attempt + 1))
    raise error


def field(entry: str, name: str) -> str | None:
    m = re.search(rf"\b{name}\s*=\s*", entry, re.I)
    if not m:
        return None
    k = m.end()
    if entry[k] == "{":
        depth, q = 0, k
        while True:
            depth += entry[q] == "{"
            depth -= entry[q] == "}"
            q += 1
            if depth == 0:
                return entry[k + 1:q - 1]
    return re.match(r"[^,}]*", entry[k:]).group(0).strip()


def plain(text: str) -> str:
    text = re.sub(r"\\[a-zA-Z]+\s*|[{}\\\"'`^~]", "", text or "")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, plain(a), plain(b)).ratio()


def source_record(doi: str) -> tuple[str, list[str], int]:
    if doi.lower().startswith("10.48550/arxiv."):
        arxiv_id = doi.split("arxiv.", 1)[1] if "arxiv." in doi else doi.split("arXiv.", 1)[1]
        root = ET.fromstring(get("https://export.arxiv.org/api/query?id_list=" + arxiv_id))
        ns = {"a": "http://www.w3.org/2005/Atom"}
        item = root.find("a:entry", ns)
        title = " ".join(item.find("a:title", ns).text.split())
        authors = [a.text for a in item.findall("a:author/a:name", ns)]
        time.sleep(8)
        return title, authors, int(item.find("a:published", ns).text[:4])
    record = json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"]
    issued = (record.get("issued") or record.get("published") or {}).get("date-parts", [[0]])[0][0]
    return record["title"][0], [a.get("family", a.get("name", "")) for a in record.get("author", [])], int(issued or 0)


def published_version(title: str) -> str:
    query = urllib.parse.quote(plain(title))
    data = json.loads(get(f"https://api.openalex.org/works?search={query}&per-page=5"
                          "&select=title,publication_year,type,doi,primary_location"))
    for work in data.get("results", []):
        source = ((work.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
        if similarity(work.get("title") or "", title) >= 0.9 and work.get("type") in ("article", "conference-paper") \
                and "arxiv" not in source.lower() and "arxiv" not in (work.get("doi") or "").lower():
            return f"{work['publication_year']} {source} {work.get('doi') or ''}"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bib")
    parser.add_argument("--keys", default="")
    args = parser.parse_args()
    keys = set(filter(None, args.keys.split(",")))
    failures = 0
    for entry in entries(Path(args.bib).read_text()):
        key = re.match(r"@\w+\s*\{\s*([^,\s]+)", entry).group(1)
        if keys and key not in keys:
            continue
        title, doi = field(entry, "title") or "", field(entry, "doi")
        year = int(re.sub(r"\D", "", field(entry, "year") or "0")[:4] or 0)
        first = (field(entry, "author") or "").split(" and ")[0]
        family = plain(first.split(",")[0] if "," in first else (first.split()[-1] if first else ""))
        if not doi:
            print(f"{key:24s} NO-DOI   {plain(title)[:60]}")
            continue
        try:
            source_title, authors, source_year = source_record(doi)
        except Exception as exc:
            failures += 1
            print(f"{key:24s} ERROR    {exc}")
            continue
        ok = similarity(title, source_title) >= 0.9 and (not family or any(family in plain(a) for a in authors)) \
            and (not year or abs(year - source_year) <= 1)
        failures += not ok
        note = ""
        if "arxiv" in doi.lower():
            note = published_version(title)
            note = f"  PUBLISHED AS: {note}" if note else ""
        print(f"{key:24s} {'OK   ' if ok else 'CHECK'}  sim={similarity(title, source_title):.2f} year={year}/{source_year}"
              f" author={family!r}{note}")
    print(f"{failures} entries need attention")


if __name__ == "__main__":
    main()
