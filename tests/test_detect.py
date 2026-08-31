"""What the classifier must keep doing.

The page next door is a drawing of a fit; this one is the fit itself, running. So these tests are
mostly about agreement rather than about layout: the page has its own copy of the tokeniser and
its own copy of the centres, and a copy that has drifted is a second model wearing the first
one's name -- it would still answer, confidently, and nothing on the screen would look wrong.

    uv pip install --python .venv/bin/python3 pytest-playwright
    .venv/bin/python3 -m pytest tests -q
"""

import json
import sys

import pytest

from conftest import BOARD, PHONE, RETINA, ROOT

# Every rule in `analyze.tokens`, one string apiece: a link, tags around prose, an advisory
# identifier and two things shaped like one, the trimming, what is dropped for having no letter,
# the em dash spaced and unspaced, a host with no scheme, letters the vocabulary cannot hold, and
# the markdown a description is actually written in.
SAMPLES = [
    "Fixes a load-bearing assert — see https://github.com/foo/bar/pull/12 for the trace.",
    '<sup>reviewed</sup> by <a href="https://cursor.com/x">bugbot</a>, and a > b in prose',
    "snyk-js-lodash-1040724 and snyk-foo-12 and SNYK-PYTHON-PIP-9999 and snyk-only",
    "_other example_ *example* --all-targets src/main trailing- _/mixed/_ __dunder__",
    "27.49 589/1000 2025-06-24 ------- -> +1 —unspaced—em—dashes—",
    "www.example.co.uk http://sub.domain.example.com:8080/p?q=1#f ftp://nope.example.com",
    "MiXeD CaSe UPPER lower Ünïcodé 日本語 ß ıi",
    "```\ncode fence\n```\n| a | b |\n|---|---|\n| 1 | 2 |\n<!-- a comment -->",
    "https://example.com/a)b (in a paren) <br> $1 \\escape\\ 100%",
    "",
]
LONG = (
    "The failing assertion was not the bug — it was load-bearing. The cache handed back a stale "
    "row before the write had landed, so the test passed for a reason nobody had written down."
)
SHORT = "fix the typo in the readme"


def analyzer():
    """`analyze.py`, which the page is checked against. Skipped where it cannot be imported."""
    sys.path.insert(0, str(ROOT))
    return pytest.importorskip("analyze", reason="numpy, scipy and numba are needed for this")


@pytest.fixture(scope="session")
def detect(server):
    return f"{server}/detect/index.html"


@pytest.fixture
def page(browser, detect):
    page = browser.new_page(viewport=BOARD, device_scale_factor=RETINA)
    page.goto(detect)
    page.wait_for_selector(".bar")
    yield page
    page.close()


def typed(page, text):
    page.fill("#text", text)
    return page.evaluate("t => classify(t).p", text)


def test_every_component_gets_a_row(page):
    k = page.evaluate("() => M.k")
    assert page.locator(".bar").count() == k == 10


def test_nothing_typed_is_the_prior(page):
    """With no words there is no likelihood, so the posterior is the prior and nothing else."""
    for recent in (0, 1):
        page.click(f'.pri[data-recent="{recent}"]')
        p = page.evaluate("() => classify('').p")
        # normalised, because the shipped prior is rounded to six places and a posterior is not:
        # the page divides by the sum it actually has
        prior = page.evaluate("() => (RECENT ? M.recent : M.prior)")
        total = sum(prior)
        assert p == pytest.approx([v / total for v in prior], abs=1e-12)


@pytest.mark.parametrize("text", ["", SHORT, LONG, "zzqqxx not a word here"])
def test_the_probabilities_add_up(page, text):
    assert sum(typed(page, text)) == pytest.approx(1.0, abs=1e-12)


def test_the_page_reads_a_text_the_way_the_corpus_was_read(page):
    """The page's tokeniser against `analyze.tokens`, string for string.

    Two copies of one rule in two languages, and the only thing keeping them in step is this. A
    page that split a word differently would classify a text the corpus never contained, and the
    answer would look exactly as confident as a right one.
    """
    az = analyzer()
    for s in SAMPLES:
        assert page.evaluate("t => tokens(t)", s) == az.tokens(s), s


def test_the_shipped_model_is_the_fitted_one(page):
    """The page's copy of the centres, read back in Python and asked the same question."""
    az = analyzer()
    import numpy as np

    raw = (ROOT / "model.js").read_text(encoding="utf-8")
    m = json.loads(raw[len("window.MODEL = ") : raw.rstrip().rfind(";")])
    vocab = az.un_front_coded(m["vocab"])
    assert len(vocab) == m["words"], "the vocabulary did not come back at the length it says"
    E = az.decode_weights(m["weights"], m["k"], m["words"], m["grid"], m["escape"], m["alphabet"])
    index = {w: j for j, w in enumerate(vocab)}

    for text in (SHORT, LONG):
        known = [index[w] for w in az.tokens(text) if w in index]
        score = np.log(m["prior"]) + len(known) * np.asarray(m["floor"])
        for j in known:
            score += E[:, j]
        p = np.exp(score - score.max())
        assert typed(page, text) == pytest.approx(p / p.sum(), abs=1e-9)


def test_a_word_the_model_never_saw_counts_for_nothing(page):
    """Out of the vocabulary is out of the model: not a rare word, not a word at all."""
    before = typed(page, LONG)
    after = typed(page, LONG + " zzqqxxwv frobnicatoriumly")
    assert after == pytest.approx(before, abs=1e-12)
    assert "zzqqxxwv" in page.locator(".missed").inner_text()


def test_the_words_and_the_prior_are_the_whole_of_the_gap(page):
    """What the strip adds up to is the number printed over it, and the rest is the prior."""
    for text in (SHORT, LONG):
        parts = page.evaluate(
            """t => {
              const r = classify(t);
              const o = r.p.map((v, c) => c).sort((a, b) => r.p[b] - r.p[a]);
              const pri = RECENT ? M.recent : M.prior;
              return [r.score[o[0]] - r.score[o[1]],
                      evidence(r, o[0], o[1]).reduce((s, v) => s + v.d, 0),
                      Math.log(pri[o[0]] / pri[o[1]])];
            }""",
            text,
        )
        assert parts[0] == pytest.approx(parts[1] + parts[2], abs=1e-6)


def test_the_prior_decides_a_short_text_and_not_a_long_one(page):
    """Which is the thing the two buttons are there to show.

    Six words leave the answer to be argued over; eighty do not, and a prior that moved a long
    text would mean the likelihood had stopped doing the work.
    """
    moved = {}
    for text in (SHORT, LONG):
        page.click('.pri[data-recent="0"]')
        corpus = typed(page, text)
        page.click('.pri[data-recent="1"]')
        moved[text] = max(abs(a - b) for a, b in zip(corpus, typed(page, text)))
    assert moved[SHORT] > 0.1, "the prior did nothing to a text of six words"
    assert moved[LONG] < 1e-6, "the prior moved a text of eighty words"


def test_a_reading_too_small_to_print_says_so(page):
    """A share of the corpus is never tiny; a posterior over ten components very often is."""
    typed(page, LONG)
    shown = [page.locator(".bar .pct").nth(i).inner_text() for i in range(10)]
    assert "<0.01%" in shown, shown
    assert not any("e-" in v for v in shown), shown


def test_the_answer_is_marked_on_exactly_one_row(page):
    typed(page, LONG)
    assert page.locator(".bar.win").count() == 1
    winner = page.evaluate(
        "() => { const r = classify(document.querySelector('#text').value);"
        " return r.p.indexOf(Math.max(...r.p)); }"
    )
    assert page.locator(".bar.win").get_attribute("data-c") == str(winner)
    page.fill("#text", "")
    assert page.locator(".bar.win").count() == 0, "an empty box has no answer to mark"


@pytest.mark.parametrize("width", [320, 390, 820, 1400])
def test_nothing_scrolls_sideways(browser, detect, width):
    page = browser.new_page(viewport={"width": width, "height": 844}, device_scale_factor=RETINA)
    page.goto(detect)
    page.wait_for_selector(".bar")
    page.fill("#text", LONG)
    over = page.evaluate("() => document.documentElement.scrollWidth - innerWidth")
    page.close()
    assert over <= 0, f"{over}px of sideways scroll at {width}px"


def test_the_strip_keeps_the_objections(page):
    """Sixty words chosen by size, not by sign: the words that argued against the answer are
    half of what the strip is for, and taking the top sixty by sign would drop all of them."""
    page.fill("#text", LONG * 4)
    kinds = page.evaluate(
        "() => [...document.querySelectorAll('.words-for span')].map(e => e.className)"
    )
    assert "against" in kinds and "for" in kinds, kinds


def test_a_phone_gets_the_board_in_one_column(browser, detect):
    page = browser.new_page(viewport=PHONE, device_scale_factor=RETINA)
    page.goto(detect)
    page.wait_for_selector(".bar")
    lefts = page.evaluate(
        "() => [...document.querySelectorAll('.cell')].map(e => e.getBoundingClientRect().left)"
    )
    page.close()
    assert len(set(round(x) for x in lefts)) == 1, "the cells are not in one column"
