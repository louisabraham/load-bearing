"""What the page must keep doing.

Every test here is a bug the page actually had. They are written against the built page in a
real browser rather than against the source, because every one of these was invisible in the
markup and only showed up once something was clicked, scrolled or resized.

    uv pip install --python .venv/bin/python3 pytest-playwright
    .venv/bin/python3 -m pytest tests -q

The browser is whichever Chromium is available: the one Playwright installs, or the one already
on the machine (`CHROMIUM=/path/to/chromium` overrides).
"""

import functools
import http.server
import os
import socketserver
import threading
from pathlib import Path

import pytest
from playwright.sync_api import Error, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
CHROMIUM_FALLBACKS = [
    "/opt/homebrew/bin/chromium",
    "/usr/bin/chromium",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]
PHONE = {"width": 390, "height": 844}
BOARD = {"width": 1400, "height": 900}
# A retina screen is not a detail here: the browser snaps `scrollTop` to physical pixels, and
# half a pixel is the whole of the rounding that used to hand a click to the wrong word.
RETINA = 2


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture(scope="session")
def site():
    """The page is served rather than opened as a file: it fetches `analysis.js` beside it."""
    handler = functools.partial(Quiet, directory=str(ROOT))
    server = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/index.html"
    server.shutdown()


@pytest.fixture(scope="session")
def browser():
    with sync_playwright() as play:
        candidates = [os.environ.get("CHROMIUM")] if os.environ.get("CHROMIUM") else []
        for launch in [{}] + [{"executable_path": p} for p in candidates + CHROMIUM_FALLBACKS]:
            if launch.get("executable_path") and not Path(launch["executable_path"]).exists():
                continue
            try:
                found = play.chromium.launch(**launch)
            except Error:
                continue
            yield found
            found.close()
            return
        pytest.skip("no Chromium to drive")


@pytest.fixture
def page(browser, site):
    page = browser.new_page(viewport=BOARD, device_scale_factor=RETINA)
    page.goto(site)
    page.wait_for_selector('.wall [data-j="999"]')
    yield page
    page.close()


def chosen(page):
    """The word the panel is drawing, and the row the column has marked."""
    return page.inner_text(".probe .word").strip(), page.get_attribute(".wall .on", "data-j")


def word_at(page, j):
    return page.inner_text(f'.wall [data-j="{j}"]').strip()


def scroll_column(page, top):
    page.evaluate("top => document.querySelector('.wall').scrollTop = top", top)
    page.wait_for_timeout(120)


# --------------------------------------------------------------------------- what it opens on


def test_opens_on_the_first_word(page):
    """The page used to open on an invitation to click. It opens on the word it is named after."""
    word, row = chosen(page)
    assert row == "0"
    assert word == word_at(page, 0)


# ------------------------------------------------------------------- choosing by clicking one


@pytest.mark.parametrize("j", [1, 3, 40, 137, 700])
def test_clicking_a_word_chooses_that_word(page, j):
    """It moved the column to the word and then chose its neighbour.

    The word on the line was found by asking which row's top edge was last above it, and the
    edges are exactly where the rounding lands: `offsetTop` is whole, the run above the words is
    fractional, and the browser snaps `scrollTop` to physical pixels.
    """
    wanted = word_at(page, j)
    page.click(f'.wall [data-j="{j}"]')
    page.wait_for_timeout(150)
    assert chosen(page) == (wanted, str(j))
    assert on_the_line(page) < 14, "the word chosen should be the word on the line"


def on_the_line(page):
    """How far the chosen row's middle is from the middle of the box it is chosen by. If these
    two ever disagree the column moves to a word and then reads back a different one."""
    return page.evaluate("""() => {
      const wall = document.querySelector('.wall'), row = document.querySelector('.wall .on');
      const box = wall.getBoundingClientRect(), r = row.getBoundingClientRect();
      return Math.abs((r.top + r.height / 2) - (box.top + wall.clientHeight / 2));
    }""")


def test_clicking_a_word_does_not_scroll_the_page(browser, site):
    """`scrollIntoView` scrolls every scrollable ancestor: on a phone the page jumped 700px and
    took the chart the word was chosen for off the screen."""
    page = browser.new_page(viewport=PHONE, device_scale_factor=RETINA)
    page.goto(site)
    page.wait_for_selector('.wall [data-j="999"]')
    page.evaluate("scrollTo(0, document.querySelector('.cell.words').offsetTop)")
    page.wait_for_timeout(100)
    before = page.evaluate("scrollY")
    page.click('.wall [data-j="6"]')
    page.wait_for_timeout(200)
    assert page.evaluate("scrollY") == before
    assert chosen(page) == (word_at(page, 6), "6")
    page.close()


# ------------------------------------------------------------------- choosing by scrolling it


def test_scrolling_the_column_chooses(page):
    """The column is the chooser: what is on the line is what is drawn, and both ends are
    reachable -- the first word is the one the page is named after."""
    scroll_column(page, 0)
    assert chosen(page)[1] == "0"
    scroll_column(page, page.evaluate("document.querySelector('.wall').scrollHeight"))
    assert chosen(page)[1] == "999"
    scroll_column(page, 4000)
    middle = int(chosen(page)[1])
    assert 0 < middle < 999


def test_arrow_keys_step_the_choice_and_bring_it_to_the_line(page):
    page.keyboard.press("ArrowDown")
    page.keyboard.press("ArrowDown")
    page.wait_for_timeout(120)
    assert chosen(page) == (word_at(page, 2), "2")
    page.keyboard.press("ArrowUp")
    page.wait_for_timeout(120)
    assert chosen(page) == (word_at(page, 1), "1")
    assert on_the_line(page) < 14, "the chosen word should sit on the line it is chosen by"


# ------------------------------------------------------------------------------ what it looks


def test_the_column_stays_inside_its_box_after_a_resize(page):
    """Portrait and back printed the words over the footer: the run above them was padding, and
    padding is a floor on a border box, so the column could not shrink back."""
    page.set_viewport_size(PHONE)
    page.wait_for_timeout(250)
    page.set_viewport_size(BOARD)
    page.wait_for_timeout(400)
    fits = page.evaluate("""() => {
      const wall = document.querySelector('.wall').getBoundingClientRect();
      const cell = document.querySelector('.cell.words').getBoundingClientRect();
      const foot = document.querySelector('.cell.foot').getBoundingClientRect();
      return { inside: wall.bottom <= cell.bottom + 1, above: wall.bottom <= foot.top + 1 };
    }""")
    assert fits == {"inside": True, "above": True}


def test_choosing_a_word_does_not_move_the_page_on_a_phone(browser, site):
    """The panel is a fixed height there, so the column does not shift under the finger that is
    choosing from it -- whatever the word, and whether its numbers take one line or two."""
    page = browser.new_page(viewport=PHONE, device_scale_factor=RETINA)
    page.goto(site)
    page.wait_for_selector('.wall [data-j="999"]')
    measure = "() => [document.querySelector('.cell.panel').getBoundingClientRect().height,"
    measure += " document.documentElement.scrollHeight]"
    first = page.evaluate(measure)
    for j in (2, 5, 9):
        page.click(f'.wall [data-j="{j}"]')
        page.wait_for_timeout(150)
        assert page.evaluate(measure) == first
    page.close()


@pytest.mark.parametrize("width", [320, 390, 820, 1400])
def test_nothing_scrolls_sideways(browser, site, width):
    page = browser.new_page(viewport={"width": width, "height": 844}, device_scale_factor=RETINA)
    page.goto(site)
    page.wait_for_selector('.wall [data-j="999"]')
    over = page.evaluate(
        "() => document.documentElement.scrollWidth - document.documentElement.clientWidth"
    )
    assert over <= 0
    page.close()


def test_the_week_under_the_pointer_reads_out_on_the_chart(page):
    """The week belongs to the diagram: it is printed at the head of the diagram's own cell, and
    with no pointer on the chart it holds the last week rather than going blank."""
    resting = page.inner_text(".readout")
    box = page.locator("#stack").bounding_box()
    page.mouse.move(box["x"] + box["width"] * 0.3, box["y"] + box["height"] / 2)
    page.wait_for_timeout(120)
    moved = page.inner_text(".readout")
    assert moved != resting
    assert "%" in moved and "of the week" in moved.lower()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] - 40)
    page.wait_for_timeout(120)
    assert page.inner_text(".readout") == resting
