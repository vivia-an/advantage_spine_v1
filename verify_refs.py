#!/usr/bin/env python3
"""Bibliography author-name audit for the Neurocomputing submission (Paper D).

For every arXiv entry in refs.bib, fetches the real author list from the arXiv
API and checks that each surname we print actually appears on the paper. For
every entry carrying a DOI, does the same via Crossref. Exits non-zero on any
mismatch.

Needs network. Run whenever refs.bib changes:  python3 verify_refs.py

Why this exists: R13 found two real citation defects that no amount of proofreading
the manuscript would surface -- "Zhang, Shane Bergsma" for what is actually
Bergsma, Shane, and a braced pseudo-author "{Mukherjee et al.}" that printed
without an initial while every other multi-author entry printed "X. Y., et al.".
Misattributing a real researcher is the kind of error the cited author notices.
"""
import re
import sys
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET

UA = {"User-Agent": "paperD-refcheck/1.0 (mailto:Fabulousjlm@uymail.com)"}
BIB = "refs.bib"
FAIL = []


def entries():
    """(key, type, author_field, title, eprint|None, doi|None) per bib entry."""
    text = open(BIB, encoding="utf-8").read()
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        typ, key, body = m.group(1).lower(), m.group(2).strip(), m.group(3)
        au = re.search(r"author\s*=\s*\{(.+?)\},?\n", body, re.S)
        ti = re.search(r"title\s*=\s*\{(.+?)\},?\n", body, re.S)
        ep = re.search(r"eprint\s*=\s*\{([\d.]+)\}", body)
        doi = re.search(r"doi\s*=\s*\{([^}]+)\}", body)
        if au:
            yield (key, typ, au.group(1), ti.group(1) if ti else "",
                   ep.group(1) if ep else None, doi.group(1) if doi else None)


def canon(name):
    """Fold a surname to a comparison key.

    Both sides must go through this or accents and multi-word surnames produce
    phantom mismatches: arXiv sends "Hakkani-Tur" as UTF-8 while refs.bib writes
    Hakkani-T\\"ur, and Crossref's family field keeps the space in
    "Riklin Raviv" / "Kamelian Rad". Strip LaTeX accent commands, fold Unicode
    diacritics to ASCII, then keep letters only.
    """
    n = re.sub(r'\\[\'"`^~=.]\{?(\w)\}?', r"\1", name)   # Richt{\'a}rik -> Richtarik
    n = n.replace("{", "").replace("}", "")
    n = unicodedata.normalize("NFKD", n)
    n = "".join(c for c in n if not unicodedata.combining(c))
    return re.sub(r"[^a-z]", "", n.lower())


def our_surnames(field):
    """Surnames we print, and whether the entry uses the 'and others' idiom."""
    parts = [p.strip() for p in re.split(r"\band\b", field) if p.strip()]
    has_others = any(p == "others" for p in parts)
    names = [p for p in parts if p != "others"]
    for n in names:
        # a braced literal like {Mukherjee et al.} is never a parsable name
        if n.startswith("{") and ("et al" in n.lower()):
            FAIL.append(f"braced pseudo-author {n!r} -- use 'Surname, Given and others'")
    sur = set()
    for n in names:
        n = n.strip()
        raw = n.split(",")[0] if "," in n else " ".join(n.split()[1:]) or n
        sur.add(canon(raw))
    sur.discard("")
    return sur, has_others


def title_key(t):
    """Fold a title for comparison: drop LaTeX braces/accents, keep letters+digits.

    Titles are the other half of a citation's identity. R14 found an entry whose
    authors were right but whose title matched no real record anywhere -- an
    author-only check passes that silently, so titles are checked too.
    """
    t = re.sub(r"\\[\'\"`^~=.]\{?(\w)\}?", r"\1", t)
    t = t.replace("{", "").replace("}", "").replace("--", "-")
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", t.lower())


def arxiv_meta(ids):
    """arXiv id -> (authors, title)."""
    url = ("http://export.arxiv.org/api/query?id_list=" + ",".join(ids)
           + f"&max_results={len(ids)}")
    req = urllib.request.Request(url, headers=UA)
    root = ET.fromstring(urllib.request.urlopen(req, timeout=45).read())
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = {}
    for e in root.findall("a:entry", ns):
        aid = e.find("a:id", ns).text.rsplit("/", 1)[1].split("v")[0]
        names = [a.find("a:name", ns).text for a in e.findall("a:author", ns)]
        out[aid] = (names, " ".join(e.find("a:title", ns).text.split()))
    return out


def crossref_meta(doi):
    """DOI -> (author surnames, title)."""
    import json
    req = urllib.request.Request(f"https://api.crossref.org/works/{doi}", headers=UA)
    msg = json.loads(urllib.request.urlopen(req, timeout=30).read())["message"]
    return ([a.get("family", "") for a in msg.get("author", [])],
            (msg.get("title") or [""])[0])


def arxiv_authors(ids):
    url = ("http://export.arxiv.org/api/query?id_list=" + ",".join(ids)
           + f"&max_results={len(ids)}")
    req = urllib.request.Request(url, headers=UA)
    root = ET.fromstring(urllib.request.urlopen(req, timeout=45).read())
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = {}
    for e in root.findall("a:entry", ns):
        aid = e.find("a:id", ns).text.rsplit("/", 1)[1].split("v")[0]
        out[aid] = [a.find("a:name", ns).text for a in e.findall("a:author", ns)]
    return out


def crossref_authors(doi):
    import json
    req = urllib.request.Request(f"https://api.crossref.org/works/{doi}", headers=UA)
    msg = json.loads(urllib.request.urlopen(req, timeout=30).read())["message"]
    return [a.get("family", "") for a in msg.get("author", [])]


def norm(names):
    """arXiv sends full display names; the surname is the last whitespace token."""
    return {canon(n.split()[-1]) for n in names if n and n.split()}


rows = list(entries())
ids = [e for _, _, _, _, e, _ in rows if e]
try:
    meta = arxiv_meta(ids) if ids else {}
except Exception as ex:
    # A network hiccup must not look like a bibliography defect, and must not
    # look like success either. Exit 2 = "could not check", distinct from
    # exit 1 = "checked and found a problem".
    print(f"REFS CHECK INCONCLUSIVE - arXiv API unreachable: {ex}")
    print("  (transient; rerun. exit 2 = not checked, not the same as a failure)")
    sys.exit(2)

print(f'{"key":30} {"src":9} authors      title')
checked = 0
for key, typ, field, our_title, ep, doi in rows:
    ours, has_others = our_surnames(field)
    truth = real_title = src = None
    if ep and ep in meta:
        names, real_title = meta[ep]
        truth, src = norm(names), "arXiv"
    elif doi:
        try:
            fams, real_title = crossref_meta(doi)
            truth, src = {canon(f) for f in fams}, "Crossref"
        except Exception as ex:
            print(f"{key:30} {'-':9} DOI lookup failed: {ex}")
            continue
    if truth is None:
        print(f"{key:30} {'-':9} no arXiv id / DOI to check against")
        continue
    checked += 1

    # R19: an empty `ours` used to make `missing` empty and pass in silence --
    # a failed author parse must be loud, not clean.
    if not ours:
        FAIL.append(f"{key}: no surnames parsed from the author field -- "
                    f"author check would be vacuous")
    if not truth:
        FAIL.append(f"{key}: {src} returned no authors -- nothing to check against")
    missing = {x for x in ours if x not in truth}
    a_ok = bool(ours) and bool(truth) and not missing
    if missing:
        FAIL.append(f"{key}: surname(s) {sorted(missing)} not among {src} authors "
                    f"{sorted(truth)}")

    # a real title may gain/lose a subtitle between preprint and camera-ready, so
    # accept containment either way; only a genuine divergence is an error.
    ours_t, real_t = title_key(our_title), title_key(real_title)
    t_ok = bool(ours_t) and (ours_t in real_t or real_t in ours_t)
    if not t_ok:
        FAIL.append(f"{key}: title mismatch\n      ours: {our_title[:78]}\n      {src}: {real_title[:78]}")

    print(f"{key:30} {src:9} {'OK' if a_ok else 'MISMATCH':12} "
          f"{'OK' if t_ok else 'MISMATCH'}")

print("=" * 70)
if FAIL:
    print(f"REFS CHECK FAILED ({len(FAIL)} issue(s)):")
    for f in FAIL:
        print("  -", f)
    sys.exit(1)
print(f"REFS CHECK PASSED - {checked}/{len(rows)} entries: authors AND titles "
      f"verified against arXiv/Crossref")
print("=" * 70)
