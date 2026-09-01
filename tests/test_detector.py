"""What the classifier must keep doing.

The board next door is a drawing of a fit; this page is the fit itself, running. So these tests
are mostly about agreement rather than about layout: the page has its own copy of the tokeniser
and its own copy of the centres, and a copy that has drifted is a second model wearing the first
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
ARRIVING = (
    "The failing assertion was not the bug — it was load-bearing. The cache handed back a stale "
    "row before the write had landed, so the test passed for a reason nobody had written down."
)
ORDINARY = "bump grpc to 1.62 and regen the protos. fixes #4412, tested locally, ci is green."


def analyzer():
    """`analyze.py`, which the page is checked against. Skipped where it cannot be imported."""
    sys.path.insert(0, str(ROOT))
    return pytest.importorskip("analyze", reason="numpy, scipy and numba are needed for this")


def model():
    raw = (ROOT / "model.js").read_text(encoding="utf-8")
    return json.loads(raw[len("window.MODEL = ") : raw.rstrip().rfind(";")])


@pytest.fixture(scope="session")
def detector(server):
    return f"{server}/detector.html"


@pytest.fixture
def page(browser, detector):
    page = browser.new_page(viewport=BOARD, device_scale_factor=RETINA)
    page.goto(detector)
    page.wait_for_selector(".verdict")
    yield page
    page.close()


def typed(page, text):
    page.fill("#text", text)
    return page.evaluate("t => classify(t).p", text)


def test_the_model_ships_no_prior(page):
    """The page holds none, so the file carries none: there is no second place to put one.

    A prior left in the file is a prior somebody will use, and the shares belong to the corpus
    over the window rather than to the text in the box.
    """
    keys = model().keys()
    assert "prior" not in keys and "recent" not in keys, sorted(keys)
    assert page.evaluate("() => M.prior === undefined && M.recent === undefined")


def test_the_answer_is_the_likelihood_and_nothing_else(page):
    """Ten equal components before the text is read: no words, no reason to prefer any of them."""
    p = page.evaluate("() => classify('').p")
    assert p == pytest.approx([1 / 10] * 10, abs=1e-12)


@pytest.mark.parametrize("text", ["", ORDINARY, ARRIVING, "zzqqxx not a word here"])
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

    m = model()
    vocab = az.un_front_coded(m["vocab"])
    assert len(vocab) == m["words"], "the vocabulary did not come back at the length it says"
    E = az.decode_weights(m["weights"], m["k"], m["words"], m["grid"], m["escape"], m["alphabet"])
    index = {w: j for j, w in enumerate(vocab)}

    for text in (ORDINARY, ARRIVING):
        known = [index[w] for w in az.tokens(text) if w in index]
        score = len(known) * np.asarray(m["floor"])
        for j in known:
            score += E[:, j]
        p = np.exp(score - score.max())
        assert typed(page, text) == pytest.approx(p / p.sum(), abs=1e-9)


def test_a_word_the_model_never_saw_counts_for_nothing(page):
    """Out of the vocabulary is out of the model: not a rare word, not a word at all."""
    before = typed(page, ARRIVING)
    after = typed(page, ARRIVING + " zzqqxxwv frobnicatoriumly")
    assert after == pytest.approx(before, abs=1e-12)


def test_the_words_are_the_whole_of_the_distance(page):
    """What the strip holds adds up to the gap the answer was decided by, exactly."""
    for text in (ORDINARY, ARRIVING):
        gap, summed = page.evaluate(
            """t => {
              const r = classify(t);
              let o = 1;
              for (let c = 2; c < M.k; c++) if (r.score[c] > r.score[o]) o = c;
              return [r.score[0] - r.score[o],
                      evidence(r, o).reduce((s, v) => s + v.d, 0)];
            }""",
            text,
        )
        assert gap == pytest.approx(summed, abs=1e-6)


@pytest.mark.parametrize("i,word", [(0, "Yes"), (1, "No")])
def test_each_example_lands_where_it_says_it_does(page, i, word):
    """The first two buttons are there for the difference between them, so the difference is a
    test. The third is not here: what it is for is the model having almost nothing to go on, and
    which side of a half it comes down on is neither fixed nor the point -- that one is tested by
    `test_a_short_text_is_not_answered_with_certainty` instead."""
    page.click(f'.try[data-i="{i}"]')
    assert page.locator(".verdict b").inner_text() == f"{word}."


def test_a_short_text_is_not_answered_with_certainty(page):
    """Seven words is what the model looks like with almost nothing to go on, and the page has to
    keep showing that rather than rounding it away."""
    page.click('.try[data-i="2"]')
    said = page.locator(".verdict .sure").inner_text()
    assert "over 99%" not in said and "100%" not in said, said


def test_an_empty_box_is_not_a_no(page):
    """With no words every component scores zero, and a verdict off the back of that would be the
    page inventing an answer out of the shape of its own arithmetic."""
    page.fill("#text", "")
    assert page.locator(".verdict.none").count() == 1
    assert page.locator(".verdict b").count() == 0, "an empty box printed an answer"
    assert "Enter some text" in page.locator(".verdict").text_content()
    assert page.locator(".strip span").count() == 0
    assert page.locator(".meter i").evaluate("e => e.style.width") == "0px"


def test_words_the_vocabulary_does_not_have_are_not_an_answer_either(page):
    page.fill("#text", "zzqqxxwv frobnicatoriumly wugglesnorf")
    assert page.locator(".verdict b").count() == 0
    assert "None of these words" in page.locator(".verdict").text_content()


def test_the_strip_keeps_the_objections(page):
    """Sixty words chosen by size, not by sign: the words that argued the other way are half of
    what the strip is for, and taking the top sixty by sign would drop all of them."""
    page.fill("#text", ARRIVING * 4)
    kinds = page.evaluate(
        "() => [...document.querySelectorAll('.strip span')].map(e => e.className)"
    )
    assert "against" in kinds and "for" in kinds, kinds


def test_the_strip_points_the_same_way_whichever_the_answer_is(page):
    """Red is `this vocabulary` on a page that said no, exactly as on one that said yes -- the
    question does not turn round when the answer does."""
    got = {}
    for text in (ORDINARY, ARRIVING):
        page.fill("#text", text)
        got[text] = page.evaluate(
            """() => { const s = document.querySelector('.strip span'); return s.className; }"""
        )
    assert got[ARRIVING] == "for", "the strongest word did not favour the answer that was yes"
    assert page.locator(".strip span.against").count() > 0


# What a link may be, and what GitHub is asked for it. The fragment is the whole of the
# difference between a description and a comment on it, and GitHub writes three kinds of comment.
LINKS = [
    ("https://github.com/o/r/pull/12", "https://api.github.com/repos/o/r/pulls/12"),
    ("https://github.com/o/r/issues/12", "https://api.github.com/repos/o/r/issues/12"),
    ("https://www.github.com/o/r/pull/12", "https://api.github.com/repos/o/r/pulls/12"),
    ("https://github.com/o/r/pull/12/commits/abc", "https://api.github.com/repos/o/r/pulls/12"),
    (
        "https://github.com/o/r/pull/12#issuecomment-456",
        "https://api.github.com/repos/o/r/issues/comments/456",
    ),
    (
        "https://github.com/o/r/pull/12/files#discussion_r789",
        "https://api.github.com/repos/o/r/pulls/comments/789",
    ),
    (
        "https://github.com/o/r/pull/12#pullrequestreview-99",
        "https://api.github.com/repos/o/r/pulls/12/reviews/99",
    ),
    ("https://gitlab.com/o/r/pull/12", None),
    ("not a link at all", None),
    ("https://github.com/o/r", None),
]


@pytest.mark.parametrize("link,want", LINKS)
def test_a_link_names_the_thing_github_is_asked_for(page, link, want):
    assert page.evaluate("l => endpoint(l)", link) == want


def answers(page, payload, status=200, headers=None):
    page.route(
        "https://api.github.com/**",
        lambda route: route.fulfill(
            status=status,
            headers={"content-type": "application/json", **(headers or {})},
            body=json.dumps(payload),
        ),
    )
    page.fill("#url", "https://github.com/o/r/pull/12")
    page.click(".load")
    # the empty slot prints a dash of its own, so the verdict appearing is not the signal: what
    # says the fetch is over is text in the box or a failure printed beside it
    page.wait_for_function(
        "() => document.querySelector('#text').value.length > 0"
        " || document.querySelector('.said').classList.contains('bad')"
    )


def test_a_link_fills_the_box_and_is_answered(page):
    """The link is a second way into the same box, so what comes back is classified the same."""
    answers(page, {"body": ARRIVING})
    assert page.input_value("#text") == ARRIVING
    assert page.locator(".verdict b").inner_text() == "Yes."
    assert "words loaded" in page.locator(".said").text_content()


def test_a_loaded_link_goes_into_the_address(page):
    """What is on the screen can be sent to somebody, which is the point of putting it there --
    and typing over it takes it out again, because an address describing a text nobody is looking
    at is worse than none."""
    answers(page, {"body": ARRIVING})
    assert "url=https%3A%2F%2Fgithub.com%2Fo%2Fr%2Fpull%2F12" in page.url
    page.fill("#text", "something else entirely")
    assert "?" not in page.url


def test_an_address_with_a_link_in_it_loads_the_link(browser, detector):
    """The other half of the same thing: the address is read back on the way in."""
    page = browser.new_page(viewport=BOARD, device_scale_factor=RETINA)
    page.route(
        "https://api.github.com/**",
        lambda route: route.fulfill(
            status=200,
            headers={"content-type": "application/json"},
            body=json.dumps({"body": ARRIVING}),
        ),
    )
    page.goto(f"{detector}?url=https://github.com/o/r/pull/12%23issuecomment-9")
    page.wait_for_function("() => document.querySelector('#text').value.length > 0")
    got, said = page.input_value("#text"), page.locator(".said").text_content()
    page.close()
    assert got == ARRIVING
    assert "words loaded" in said


def test_a_link_that_fails_leaves_the_address_alone(page):
    """An address holding a link that answered nothing is worse than no address at all."""
    answers(page, {}, 404)
    assert "?" not in page.url


@pytest.mark.parametrize(
    "status,headers,payload,said",
    [
        (404, None, {}, "no such thing"),
        (403, None, {"message": "API rate limit exceeded for 1.2.3.4."}, "sixty an hour"),
        (403, None, {"message": "Must have push access."}, "refused that one"),
        (200, None, {"body": "   "}, "no text in it"),
    ],
)
def test_a_link_that_does_not_work_says_which_way_it_failed(page, status, headers, payload, said):
    """The rate limit is the failure a reader will actually hit, and `wait` is the right advice
    where `try again` is not, so the three are told apart rather than answered as one."""
    answers(page, payload, status, headers)
    assert said in page.locator(".said").text_content()
    assert page.input_value("#text") == "", "a failed fetch emptied or overwrote the box"


@pytest.mark.parametrize("width", [320, 390, 820, 1400])
def test_nothing_scrolls_sideways(browser, detector, width):
    page = browser.new_page(viewport={"width": width, "height": 844}, device_scale_factor=RETINA)
    page.goto(detector)
    page.wait_for_selector(".verdict")
    page.fill("#text", ARRIVING)
    over = page.evaluate("() => document.documentElement.scrollWidth - innerWidth")
    page.close()
    assert over <= 0, f"{over}px of sideways scroll at {width}px"


def test_a_phone_gets_the_board_in_one_column(browser, detector):
    page = browser.new_page(viewport=PHONE, device_scale_factor=RETINA)
    page.goto(detector)
    page.wait_for_selector(".verdict")
    lefts = page.evaluate(
        "() => [...document.querySelectorAll('.cell')].map(e => e.getBoundingClientRect().left)"
    )
    page.close()
    assert len(set(round(x) for x in lefts)) == 1, "the cells are not in one column"
