"""Turn cleaned prose into the countable expressions.

Three feature kinds share one namespace so that everything downstream (series,
emergence, clustering, scoring) treats them uniformly:

* word n-grams (1-3), counted over a *canonical* token stream where hyphens and
  apostrophes are split. This makes "load-bearing" and "load bearing" the same
  bigram, which is the expression family the project cares about.
* surface forms that carry their own signal, kept as separate features:
  hyphenated compounds (``HYPH:load-bearing``) and typography markers
  (``TYPO:em_dash``).
* grammatical constructions (``CONSTR:not_x_but_y``), a small hand-written set of
  rhetorical frames that no n-gram captures because the slots vary.

Counting is by *document frequency*: a feature is present or absent in a
document, never counted twice. A single author repeating a phrase eight times in
one comment should not look like eight adoptions.
"""

from __future__ import annotations

import re

MAX_TOKEN_LEN = 24
MAX_N = 3

_WORD = re.compile(r"[a-z][a-z''’-]*[a-z]|[a-z]")
_SPLIT_INNER = re.compile(r"[-'’]")

# Tokens that mark a boundary no n-gram should cross: crossing sentence and
# clause boundaries invents phrases nobody wrote.
_BOUNDARY = "\x00"
_SENT_SPLIT = re.compile(r"[.!?;:,()\[\]{}\"“”\n]+|\s+[-–—]\s+")


def tokenize(text: str) -> list[str]:
    """Canonical token stream with boundary markers.

    Hyphens and apostrophes are dissolved so that surface variants collapse:
    "load-bearing" -> ["load", "bearing"], "isn't" -> ["isnt"].
    """
    low = text.lower()
    out: list[str] = []
    for seg in _SENT_SPLIT.split(low):
        if not seg or not seg.strip():
            continue
        for w in _WORD.findall(seg):
            if len(w) > MAX_TOKEN_LEN:
                continue
            out.extend(p for p in _SPLIT_INNER.split(w) if p)
        out.append(_BOUNDARY)
    return out


def word_ngrams(tokens: list[str], max_n: int = MAX_N) -> set[str]:
    """All 1..max_n grams that do not cross a boundary marker."""
    grams: set[str] = set()
    run: list[str] = []
    for tok in tokens + [_BOUNDARY]:
        if tok == _BOUNDARY:
            for n in range(1, max_n + 1):
                for i in range(len(run) - n + 1):
                    grams.add(" ".join(run[i : i + n]))
            run = []
        else:
            run.append(tok)
    return grams


# ------------------------------------------------------------------ surface forms

_HYPHENATED = re.compile(r"\b([a-z]{2,}(?:-[a-z]{2,}){1,2})\b")


def hyphen_features(text: str) -> set[str]:
    return {f"HYPH:{m}" for m in _HYPHENATED.findall(text.lower())}


TYPO_MARKERS: dict[str, re.Pattern] = {
    "em_dash": re.compile(r"—"),
    "en_dash_spaced": re.compile(r"\s–\s"),
    "curly_apostrophe": re.compile(r"’"),
    "curly_quotes": re.compile(r"[“”]"),
    "ellipsis_char": re.compile(r"…"),
    "bullet_char": re.compile(r"^\s*[•▪●]", re.M),
    "arrow": re.compile(r"→|->"),
    "checkmark": re.compile(r"[✅✔✓]"),
    "emoji_heading": re.compile(r"^[\U0001F300-\U0001FAFF]", re.M),
    "bold_lead_colon": re.compile(r"^\s*\*\*[^*\n]{2,40}\*\*\s*:", re.M),
    "numbered_bold": re.compile(r"^\s*\d+\.\s+\*\*", re.M),
}


def typo_features(raw_text: str) -> set[str]:
    return {f"TYPO:{k}" for k, p in TYPO_MARKERS.items() if p.search(raw_text)}


# ------------------------------------------------------------------ constructions

# Rhetorical frames with variable slots. Written against the *cleaned, lowercased*
# text. Each is deliberately narrow; a loose pattern would fire on ordinary prose
# and drown its own signal.
CONSTRUCTIONS: dict[str, re.Pattern] = {
    "not_just_but": re.compile(r"\bnot (?:just|only|merely) \w[\w ]{0,30}?,? but (?:also )?\w"),
    "isnt_x_its_y": re.compile(r"\b(?:isnt|is not|its not|it is not) (?:about )?\w[\w ]{0,25}?,? (?:its|it is|but) \w"),
    "the_x_is_y_the_x_is": re.compile(r"\bthe (?:real|actual|key|core) (?:issue|problem|question|point) (?:here )?is\b"),
    "what_x_does_is": re.compile(r"\bwhat (?:this|that|it) (?:does|means|gives us|buys us) is\b"),
    "lets_verb": re.compile(r"\blets (?:take|walk|dig|dive|unpack|break|think)\b"),
    "worth_noting": re.compile(r"\b(?:worth|it is worth|its worth) (?:noting|mentioning|calling out|flagging)\b"),
    "to_be_clear": re.compile(r"\bto be (?:clear|fair|honest|precise)\b"),
    "that_said": re.compile(r"\bthat (?:said|being said)\b|\bhaving said that\b"),
    "here_is_the_thing": re.compile(r"\bhere(?:s| is) the (?:thing|catch|rub|problem)\b"),
    "the_tricky_part": re.compile(r"\bthe (?:tricky|subtle|interesting|annoying|fun) (?:part|bit|thing) (?:is|here)\b"),
    "under_the_hood": re.compile(r"\bunder the hood\b"),
    "out_of_the_box": re.compile(r"\bout of the box\b"),
    "first_class": re.compile(r"\bfirst[- ]class (?:citizen|support|primitive)\b"),
    "single_source_truth": re.compile(r"\bsingle source of truth\b"),
    "happy_path": re.compile(r"\bhappy path\b"),
    "foot_gun": re.compile(r"\bfoot[- ]?gun"),
    "surface_area": re.compile(r"\b(?:api |attack |surface )?surface area\b"),
    "load_bearing": re.compile(r"\bload[- ]bearing\b"),
    "source_of_truth": re.compile(r"\bsource of truth\b"),
    "escape_hatch": re.compile(r"\bescape hatch\b"),
    "sharp_edges": re.compile(r"\bsharp edges\b"),
    "in_practice": re.compile(r"\bin practice(?:,| this| it)\b"),
    "concretely": re.compile(r"^\s*concretely\b|\bmore concretely\b"),
    "importantly": re.compile(r"\b(?:more |most )?importantly,"),
    "note_that_lead": re.compile(r"^\s*note that\b", re.M),
    "one_thing_to_note": re.compile(r"\bone (?:thing|caveat|nit)(?: to (?:note|flag|call out))?\b"),
    "does_x_make_sense": re.compile(r"\bdoes (?:that|this) (?:make sense|sound (?:right|good))\b"),
    "happy_to_x": re.compile(r"\bhappy to (?:iterate|adjust|change|revisit|split|discuss)\b"),
    "let_me_know_if": re.compile(r"\blet me know if\b"),
    "i_went_ahead": re.compile(r"\b(?:i|we) went ahead and\b"),
    "rather_than_x_we": re.compile(r"\brather than \w[\w ]{0,25}?, (?:we|i|this)\b"),
    "tradeoff_framing": re.compile(r"\b(?:the )?trade[- ]?offs? (?:here |is |are )"),
    "for_context": re.compile(r"^\s*(?:for|some) context\b", re.M),
    "tldr": re.compile(r"\btl;?dr\b"),
    "gotcha": re.compile(r"\bgotchas?\b"),
    "nit_prefix": re.compile(r"^\s*nit(?:pick)?s?\s*[:.\-]", re.M),
    "as_an_aside": re.compile(r"\bas an aside\b|\bside note\b"),
    "double_check": re.compile(r"\bdouble[- ]check(?:ing|ed)?\b"),
    "sanity_check": re.compile(r"\bsanity[- ]check"),
    "reasonable_to": re.compile(r"\b(?:seems|sounds) (?:reasonable|good to me|sensible)\b"),
    "defensive_hedge": re.compile(r"\b(?:i (?:might|may) be (?:missing|wrong)|correct me if i(?:m| am) wrong)\b"),
    "no_op": re.compile(r"\bno[- ]?ops?\b"),
    "in_the_wild": re.compile(r"\bin the wild\b"),
    "orthogonal": re.compile(r"\borthogonal\b"),
    "idempotent": re.compile(r"\bidempotent\b"),
    "canonical": re.compile(r"\bcanonical(?:ly)?\b"),
    "downstream_upstream": re.compile(r"\b(?:downstream|upstream) (?:consumers|callers|effects)\b"),
    "guardrail": re.compile(r"\bguard[- ]?rails?\b"),
    "invariant": re.compile(r"\binvariants?\b"),
}


def construction_features(text: str) -> set[str]:
    low = text.lower()
    return {f"CONSTR:{k}" for k, p in CONSTRUCTIONS.items() if p.search(low)}


# ------------------------------------------------------------------ char n-grams

def char_ngrams(text: str, n: int = 5, limit: int = 4000) -> set[str]:
    """Character n-grams over collapsed whitespace.

    Kept available because morphological and typographic habits can show up
    below the word level, but off by default: the vocabulary is enormous and the
    hits are hard to interpret next to word n-grams.
    """
    s = re.sub(r"\s+", " ", text.lower())[:limit]
    return {f"CHR:{s[i : i + n]}" for i in range(len(s) - n + 1)}


# ------------------------------------------------------------------ document features

MAX_DOC_TOKENS = 400


def features(text: str, use_char: bool = False,
             max_tokens: int = MAX_DOC_TOKENS) -> set[str]:
    """The full present/absent feature set of one cleaned document.

    Only the first `max_tokens` words count. Document frequency treats every
    document as one vote, but a very long document votes on far more expressions
    than a short one: the median document here is 30 tokens, while generated
    project write-ups and hackathon submissions run past 1000 and carry thousands
    of distinct n-grams each. A few dozen of them push many expressions over the
    discovery threshold at once, and those expressions then co-emerge perfectly --
    because they are literally the same documents. Truncating bounds any single
    document's leverage without discarding it.
    """
    if max_tokens:
        parts = text.split()
        if len(parts) > max_tokens:
            text = " ".join(parts[:max_tokens])
    toks = tokenize(text)
    f = word_ngrams(toks)
    f |= hyphen_features(text)
    f |= typo_features(text)
    f |= construction_features(text)
    if use_char:
        f |= char_ngrams(text)
    return f


# ------------------------------------------------------------------ families

_IRREGULAR_KEEP = {
    "is", "was", "has", "does", "this", "its", "us", "less", "class", "process",
    "yes", "always", "unless", "always", "plus", "thus", "bus", "gas", "as",
    "series", "analysis", "basis", "status", "focus", "versus", "https",
}


_IRREGULAR = {
    # -ex/-ix and Greek/Latin plurals that suffix rules get wrong, and that no
    # purely mechanical singulariser recovers
    "indices": "index", "matrices": "matrix", "vertices": "vertex",
    "appendices": "appendix", "suffixes": "suffix", "criteria": "criterion",
    "phenomena": "phenomenon", "schemata": "schema", "analyses": "analysis",
    "bases": "basis", "crises": "crisis", "theses": "thesis", "axes": "axis",
    "children": "child", "people": "person", "men": "man", "women": "woman",
    "feet": "foot", "teeth": "tooth", "data": "datum", "media": "medium",
}


def _wordnet():
    """WordNet's noun lemmatiser, if the corpus is installed.

    Preferred over suffix rules where available because it is dictionary-grounded:
    it leaves a token alone unless the singular is a real word, so it cannot turn
    "this" into "thi" the way a mechanical stripper does. Optional -- the pipeline
    must not require a corpus download to run.
    """
    if not hasattr(_wordnet, "_cached"):
        try:
            from nltk.stem import WordNetLemmatizer  # type: ignore

            wnl = WordNetLemmatizer()
            wnl.lemmatize("tests", "n")  # forces the corpus load, raising if absent
            _wordnet._cached = wnl
        except Exception:
            _wordnet._cached = None
    return _wordnet._cached


def _rule_singularize(tok: str) -> str:
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith(("ses", "xes", "zes", "ches", "shes")):
        return tok[:-2]
    if tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


def _singularize(tok: str) -> str:
    """Conservative singularisation for family grouping only.

    Never used when counting: inflection carries stylistic information, and
    collapsing it would erase real signal. Here the goal is narrower -- stop one
    habit from being counted as several independent witnesses.
    """
    if tok in _IRREGULAR_KEEP or len(tok) <= 3:
        return tok
    if tok in _IRREGULAR:
        return _IRREGULAR[tok]
    wnl = _wordnet()
    if wnl is not None:
        lemma = wnl.lemmatize(tok, "n")
        if lemma != tok:
            return lemma
        # WordNet leaves unknown words alone; fall through for coinages like
        # "gotchas" or "callbacks" that no lexicon lists
    return _rule_singularize(tok)


def family_key(term: str) -> str:
    """Group inflectional and surface variants of the same expression.

    ``HYPH:load-bearing`` and the bigram ``load bearing`` map to the same family,
    as do "load bearing assumption" and "load bearing assumptions". Prefixed
    feature kinds keep their own namespace.
    """
    if term.startswith(("TYPO:", "CONSTR:", "CHR:")):
        return term
    if term.startswith("HYPH:"):
        term = term[5:]
    parts = term.replace("-", " ").split()
    if not parts:
        return term
    parts[-1] = _singularize(parts[-1])
    return " ".join(parts)


def is_word_ngram(term: str) -> bool:
    return not term.startswith(("TYPO:", "CONSTR:", "CHR:", "HYPH:"))


def n_words(term: str) -> int:
    if not is_word_ngram(term):
        return 1
    return len(term.split())
