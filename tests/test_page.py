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
import re
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
def chromium():
    """Playwright, the options that launched a Chromium, and the browser they launched.

    The options are handed out with it because one test needs a second browser started with
    different flags -- a phone forcing dark mode is a property of the browser, not of a page --
    and it must be the same Chromium this one found.
    """
    with sync_playwright() as play:
        candidates = [os.environ.get("CHROMIUM")] if os.environ.get("CHROMIUM") else []
        for launch in [{}] + [{"executable_path": p} for p in candidates + CHROMIUM_FALLBACKS]:
            if launch.get("executable_path") and not Path(launch["executable_path"]).exists():
                continue
            try:
                found = play.chromium.launch(**launch)
            except Error:
                continue
            yield play, launch, found
            found.close()
            return
        pytest.skip("no Chromium to drive")


@pytest.fixture(scope="session")
def browser(chromium):
    return chromium[2]


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
    assert on_the_line(page) < 2, "the word chosen should be the word on the line"


def on_the_line(page):
    """How far the top of the chosen row is from the line it is chosen by. If these two ever
    disagree the column moves to a word and then reads back a different one.

    Where the line sits is `--line` on the box, read from the page rather than repeated here: it
    is a third of the way down and it has been half, and a test that carries its own copy of that
    number passes for a while after the page stops agreeing with it.
    """
    return page.evaluate("""() => {
      const wall = document.querySelector('.wall'), row = document.querySelector('.wall .on');
      const f = parseFloat(getComputedStyle(wall).getPropertyValue('--line')) / 100;
      const box = wall.getBoundingClientRect(), r = row.getBoundingClientRect();
      return Math.abs(r.top - (box.top + Math.max(0, wall.clientHeight * f - 14)));
    }""")


def test_the_chosen_box_starts_at_the_same_height_whatever_the_word(page):
    """The words are set in the size of their lift, from the commonest at the top of the column
    to the rarest at the bottom, so a box centred on the line rides up and down with the size of
    the word inside it. Its top edge does not."""
    tops, heights = [], []
    for j in (0, 300, 999):
        page.click(f'.wall [data-j="{j}"]')
        page.wait_for_timeout(150)
        top, height = page.evaluate("""() => {
          const wall = document.querySelector('.wall'), row = document.querySelector('.wall .on');
          const r = row.getBoundingClientRect();
          return [r.top - wall.getBoundingClientRect().top, r.height];
        }""")
        tops.append(top)
        heights.append(height)
    assert max(heights) - min(heights) > 4, (
        f"the rows must differ in size for this to mean anything: {heights}"
    )
    assert max(tops) - min(tops) < 1.5, f"the top of the box moved: {tops}"


def test_clicking_a_word_does_not_scroll_the_page(browser, site):
    """`scrollIntoView` scrolls every scrollable ancestor: on a phone the page jumped 700px and
    took the chart the word was chosen for off the screen."""
    page = browser.new_page(viewport=PHONE, device_scale_factor=RETINA)
    page.goto(site)
    page.wait_for_selector('.wall [data-j="999"]')
    page.evaluate("scrollTo(0, document.querySelector('.cell.words').offsetTop)")
    # The word must be WHOLLY inside the column before it is clicked, and whether any given one
    # is depends on the day's data: the words are set at the size of their lift, so which of
    # them straddles the column's bottom edge at rest changes as the corpus does. A straddling
    # word is scrolled into view by the DRIVER before it can be clicked, and that scroll -- the
    # driver's, not the page's -- moves the page for a reason this test is not about. Left
    # implicit it made the test a coin toss on the corpus.
    page.evaluate(
        """() => { const w = document.querySelector('.wall');
             w.scrollTop = w.querySelector('[data-j="6"]').offsetTop - 40; }"""
    )
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


def test_one_tick_of_a_mouse_wheel_is_one_word(page):
    """A wheel with detents asks for one word. The browser would give it the hundred pixels it
    asked for in the event, which is five of them.

    Beware the numbers here: the driver DOUBLES a wheel delta on the way to the page, so the 120
    below arrives as 240. That is still inside the band `NOTCH` calls a notch -- at the very top
    of it -- so this passes for the right reason, but a ceiling lowered even slightly would break
    it, and the delta to change would be this one rather than the ceiling.
    """
    box = page.locator(".wall").bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    seen = [int(chosen(page)[1])]
    for _ in range(4):
        page.mouse.wheel(0, 120)
        page.wait_for_timeout(120)
        seen.append(int(chosen(page)[1]))
    for _ in range(2):
        page.mouse.wheel(0, -120)
        page.wait_for_timeout(120)
        seen.append(int(chosen(page)[1]))
    assert seen == [0, 1, 2, 3, 4, 3, 2]


def test_a_trackpad_still_moves_the_column_itself(page):
    """It asks in pixels rather than in detents, and a finger that moved the column by three
    words should move it by three words rather than by one per event it happened to send."""
    box = page.locator(".wall").bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    before = int(chosen(page)[1])
    for _ in range(10):
        page.mouse.wheel(0, 8)
        page.wait_for_timeout(40)
    page.wait_for_timeout(250)
    moved = int(chosen(page)[1]) - before
    assert 0 < moved < 10, f"ten small deltas moved {moved} words"


def wheeled(page, mode, delta, n, start=0):
    """One stream of wheel events at a chosen unit, and what the PAGE chose to do with it.

    Dispatched rather than driven, because a browser cannot be asked for a device it has not
    got: `deltaMode` is the reporting unit of whatever is under the reader's hand, and the whole
    of this bug lives in units the machine running the tests will never send. Synthetic events
    reach the handler and do not scroll, so the words counted here are the page's own doing and
    nothing else's -- which is exactly what is under test.
    """
    return page.evaluate(
        """([mode, delta, n, start]) => {
             const w = document.querySelector('.wall');
             pick(start, true);
             let taken = 0;
             for (let i = 0; i < n; i++) {
               const e = new WheelEvent('wheel',
                 {deltaMode: mode, deltaY: delta, bubbles: true, cancelable: true});
               w.dispatchEvent(e);
               if (e.defaultPrevented) taken++;
             }
             return [taken, S.j - start];
           }""",
        [mode, delta, n, start],
    )


def test_a_wheel_reporting_fractions_of_a_line_is_not_a_word_each(page):
    """`deltaMode` was read as if it named the device -- pixels meaning something continuous,
    anything else a detent -- and it does not. Firefox reports a wheel in LINES, and a
    high-resolution wheel reports fractions of a line; every one of those fractions failed the
    `deltaMode === 0` clause, took the stepping branch and moved a whole word, because
    `Math.sign` threw the magnitude away before anything could notice how small it was. Six
    detents' worth of a wheel reporting tenths moved sixty words.

    The guard tests a distance, so the delta is put into pixels before it is tested.
    """
    # a detent, in either unit, is still one word: three lines is a tick in Firefox and a
    # hundred pixels is a tick in Chrome
    assert wheeled(page, 1, 3.0, 6) == [6, 6]
    assert wheeled(page, 0, 100.0, 6) == [6, 6]
    # and one line is the narrowest a real detent gets, which forty pixels to the line clears
    assert wheeled(page, 1, 1.0, 6) == [6, 6]
    # the same spin reported finely is the browser's, as it always was in pixels
    assert wheeled(page, 1, 0.25, 24) == [0, 0]
    assert wheeled(page, 1, 0.10, 60) == [0, 0]
    assert wheeled(page, 0, 8.0, 75) == [0, 0]


def test_a_delta_too_big_to_be_a_notch_is_the_browsers(page):
    """There was a floor under the stepping branch and no ceiling, so a delta of any size above
    it moved exactly one word -- and a device that delivers its distance in few large events
    therefore moved the column less the faster it was spun. One event carrying 720px moved one
    word; the same 720px in events small enough to be handed to the browser moved the whole 720,
    which is thirty-odd words of this column."""
    # a notch, and two of them arriving as one event, are still notches
    assert wheeled(page, 0, 120.0, 4) == [4, 4]
    assert wheeled(page, 0, 240.0, 4) == [4, 4]
    # past that it is a continuous device at speed, and the distance is the browser's to scroll
    assert wheeled(page, 0, 300.0, 4) == [0, 0]
    assert wheeled(page, 0, 720.0, 1) == [0, 0]
    # in either direction: the band is on the size of the delta, not on its sign. Started far
    # enough down the column that stepping up has somewhere to go -- at the top it has not, and
    # the event is still taken, which is a fact about the end of the column and not about this.
    assert wheeled(page, 0, -120.0, 4, start=10) == [4, -4]
    assert wheeled(page, 0, -720.0, 1, start=10) == [0, 0]


def test_arrow_keys_step_the_choice_and_bring_it_to_the_line(page):
    page.keyboard.press("ArrowDown")
    page.keyboard.press("ArrowDown")
    page.wait_for_timeout(120)
    assert chosen(page) == (word_at(page, 2), "2")
    page.keyboard.press("ArrowUp")
    page.wait_for_timeout(120)
    assert chosen(page) == (word_at(page, 1), "1")
    assert on_the_line(page) < 2, "the chosen word should sit on the line it is chosen by"


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


@pytest.mark.parametrize("width", [320, 390, 1080, 1400, 1920])
def test_the_words_stop_before_the_marks_on_the_frame(browser, site, width):
    """A row is a block, so the outline drawn on the chosen one runs the whole width of the
    column whatever the word is. It reached into the strip the arrows stand in at every width
    the board takes, and once there was a rail between them it crossed that too -- an outline
    clipping the tip of an arrow is a blemish, an outline cut by a hairline the length of the
    box is a fault. The column's right pad is the room those marks need."""
    page = browser.new_page(viewport={"width": width, "height": 900}, device_scale_factor=RETINA)
    page.goto(site)
    page.wait_for_selector('.wall [data-j="999"]')
    gap = page.evaluate("""() => {
      const row = document.querySelector('.wall .on').getBoundingClientRect();
      const marks = document.querySelector('.steps').getBoundingClientRect();
      return marks.left - row.right;
    }""")
    page.close()
    assert gap >= 4, f"the chosen row comes within {gap:.1f}px of the marks on the frame"


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


def test_the_chart_is_drawn_at_the_size_of_its_box(page):
    """It was drawn fourteen pixels taller than its box on first paint: the chart is measured
    before it is drawn, and the week's readout arrived afterwards and took its line out of the
    chart's own height. Any resize redrew it correctly, so zooming in and back out changed the
    layout and never changed it back."""

    def geometry():
        return page.evaluate("""() => {
          const s = document.querySelector('#stack'), b = s.getBBox();
          return {box: Math.round(s.clientHeight), drawn: Math.round(b.height)};
        }""")

    first = geometry()
    assert abs(first["box"] - first["drawn"]) <= 1, (
        f"drawn at the wrong size on first paint: {first}"
    )
    page.set_viewport_size({"width": 933, "height": 600})
    page.wait_for_timeout(300)
    page.set_viewport_size(BOARD)
    page.wait_for_timeout(300)
    assert geometry() == first, "a zoom out and back changed the chart"


def test_the_week_under_the_pointer_reads_out_on_the_chart(page):
    """The week belongs to the diagram: it is printed at the head of the diagram's own cell, and
    with no pointer on the chart the line goes back to asking for one."""
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


@pytest.mark.parametrize("readout", [".cell.chart .readout", ".probe .readout"])
def test_an_untouched_chart_asks_to_be_touched(page, readout):
    """Both charts used to rest on the newest week.

    A date and a number sitting above a chart nobody has pointed at read as a fact the board is
    asserting, and the newest week is not what either chart is about. Worse, it hid the only
    thing a reader who has not touched the chart yet needs to know -- that it answers. The line
    now carries the invitation until there is a real reading to put there.

    What is asserted is that the line INVITES -- it names a gesture, and it names what the
    gesture answers -- and not any one wording of it. This held the exact phrase "for any week",
    which has since been reworded to "to see a week", so a test about nothing having gone wrong
    failed for two commits.
    """
    line = page.inner_text(readout).strip().lower()
    assert any(verb in line for verb in ("hover", "touch")), (
        f"no gesture named on an untouched chart: {line!r}"
    )
    assert "week" in line, f"the invitation does not say what it answers: {line!r}"
    assert not re.search(r"\d{4}-\d{2}-\d{2}", line), f"a week is still standing there: {line!r}"
    # and it is set apart from a reading rather than looking like one
    assert "nudge" in (page.get_attribute(readout, "class") or "")


def test_the_invitation_names_the_gesture_the_device_has(browser, site):
    """A phone cannot hover and a mouse cannot touch, so a nudge naming the wrong one is worse
    than no nudge."""
    for touch, wanted in ((False, "hover"), (True, "touch")):
        page = browser.new_page(
            viewport=PHONE, device_scale_factor=RETINA, has_touch=touch, is_mobile=touch
        )
        page.goto(site)
        page.wait_for_selector('.wall [data-j="999"]')
        assert wanted in page.inner_text(".cell.chart .readout").lower()
        page.close()


def test_a_reading_replaces_the_invitation_and_takes_the_ink(page):
    """The two are different kinds of line and are set differently: the reading is a measurement
    in the ink, the invitation is a label in the muted grey."""
    grey = page.eval_on_selector(".cell.chart .readout", "e => getComputedStyle(e).color")
    box = page.locator("#stack").bounding_box()
    page.mouse.move(box["x"] + box["width"] * 0.4, box["y"] + box["height"] / 2)
    page.wait_for_timeout(120)
    assert "nudge" not in (page.get_attribute(".cell.chart .readout", "class") or "")
    ink = page.eval_on_selector(".cell.chart .readout", "e => getComputedStyle(e).color")
    assert ink != grey, f"the reading and the invitation are set the same: {ink}"


# ------------------------------------------------------------------ scrubbing with a finger


def touch_page(browser, site):
    """A phone with a real touchscreen, so the browser makes its own gesture decisions."""
    page = browser.new_page(
        viewport=PHONE, device_scale_factor=RETINA, is_mobile=True, has_touch=True
    )
    page.goto(site)
    page.wait_for_selector('.wall [data-j="999"]')
    return page


# each chart writes its week into the readout at the head of its own cell
READOUT = {"#stack": ".cell.chart .readout", ".probe svg": ".probe .readout"}


def drag_across(page, selector, drift=0):
    """One finger, straight across the chart, reporting where the bar stood at every step.

    The events go through the browser's own input pipeline rather than through synthetic
    `TouchEvent`s: the bug is the browser deciding mid-gesture that the drag belongs to it, and
    a hand-built event never gives it that decision to make. `drift` moves the finger off the
    chart part way through, which is what a thumb actually does.
    """
    cdp = page.context.new_cdp_session(page)
    page.eval_on_selector(selector, "e => e.scrollIntoView({block: 'center'})")
    page.wait_for_timeout(250)
    box = page.locator(selector).bounding_box()
    page.evaluate(
        """sel => {
        window.__cancels = 0;
        document.querySelector(sel)
          .addEventListener('pointercancel', () => window.__cancels++, true);
      }""",
        selector,
    )
    y = box["y"] + box["height"] / 2
    xs = [box["x"] + 12 + i * (box["width"] - 24) / 12 for i in range(13)]
    cdp.send(
        "Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": xs[0], "y": y}]}
    )
    bar, readouts = [], []
    for step, x in enumerate(xs[1:]):
        point = {"x": x, "y": y + (drift if step > 6 else 0)}
        cdp.send("Input.dispatchTouchEvent", {"type": "touchMove", "touchPoints": [point]})
        page.wait_for_timeout(30)
        bar.append(page.get_attribute(f"{selector} .cross", "x1"))
        readouts.append(page.inner_text(READOUT[selector]))
    cancels = page.evaluate("window.__cancels")
    cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
    page.wait_for_timeout(80)
    return bar, readouts, cancels


@pytest.mark.parametrize("selector", ["#stack", ".probe svg"])
def test_a_finger_dragged_across_a_chart_scrubs_it(browser, site, selector):
    """The bar moved once and died.

    Left at `touch-action: auto` the browser read the first sideways millimetre of the drag as a
    pan, took the gesture and sent `pointercancel`: the bar jumped to wherever the finger had
    landed, vanished, and the readout fell back to the last week. The chart answered a tap and
    could not be dragged at all.
    """
    page = touch_page(browser, site)
    bar, readouts, cancels = drag_across(page, selector)
    assert cancels == 0, "the browser took the gesture away from the chart"
    assert len(set(bar)) >= 10, f"the bar stood in {len(set(bar))} places across the whole chart"
    assert bar == sorted(bar, key=float), "the bar did not follow the finger across"
    assert len(set(readouts)) >= 10, "the week under the finger did not keep up with the bar"
    page.close()


def test_the_bar_keeps_a_finger_that_drifts_off_the_chart(browser, site):
    """A thumb dragged sideways does not travel in a straight line, and the chart is 200px tall
    on a phone. Uncaptured, the events went to whatever was underneath and the bar stopped dead
    half way across."""
    page = touch_page(browser, site)
    bar, _, cancels = drag_across(page, "#stack", drift=140)
    assert cancels == 0
    assert len(set(bar)) >= 10, f"the bar stopped when the finger left the box: {bar}"
    page.close()


def test_the_page_still_scrolls_over_a_chart(browser, site):
    """The sideways axis is the chart's; the up-and-down one stays the page's. Taking both would
    make the chart a hole in a page that is scrolled past far more often than it is scrubbed."""
    page = touch_page(browser, site)
    cdp = page.context.new_cdp_session(page)
    page.eval_on_selector("#stack", "e => e.scrollIntoView({block: 'center'})")
    page.wait_for_timeout(250)
    box = page.locator("#stack").bounding_box()
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    before = page.evaluate("scrollY")
    cdp.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": x, "y": y}]})
    for i in range(1, 10):
        cdp.send(
            "Input.dispatchTouchEvent",
            {"type": "touchMove", "touchPoints": [{"x": x, "y": y - i * 14}]},
        )
        page.wait_for_timeout(20)
    cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
    page.wait_for_timeout(400)
    assert page.evaluate("scrollY") > before + 50, "a finger dragged up the chart did not scroll"
    page.close()


def test_the_bar_rests_when_the_finger_lifts(browser, site):
    """A mouse leaves the chart and the bar goes; a finger has no way to leave, so lifting it is
    what says the gesture is over."""
    page = touch_page(browser, site)
    drag_across(page, "#stack")
    assert page.get_attribute("#stack .cross", "opacity") == "0"
    page.close()


# ------------------------------------------------------------------------ the size of the text


def test_the_phone_is_told_not_to_resize_the_text():
    """Only the `-webkit-` spelling was here, which Firefox does not read.

    This one is asserted against the source rather than through the browser, because that is
    where it is visible: a browser reports the one spelling it understands and silently drops
    the two it does not, so Chromium cannot tell us whether Firefox was catered for.
    """
    css = (ROOT / "index.html").read_text(encoding="utf-8")
    for spelling in ("-webkit-text-size-adjust", "-moz-text-size-adjust", "text-size-adjust"):
        assert f"{spelling}: 100%;" in css, f"{spelling} is not declared"


# --------------------------------------------------------------------- the colours it keeps


# What a phone in dark mode does to a page that never said which schemes it supports. Chrome
# spells it this way; Firefox for Android arrives at the same place through its own setting.
FORCE_DARK = ["--enable-features=WebContentsForceDark", "--force-dark-mode"]


def board_shot(browser, site, css=None):
    """The top of the board on a phone, as pixels. Byte for byte reproducible across launches,
    so two renderings can simply be compared."""
    page = browser.new_page(
        viewport=PHONE,
        device_scale_factor=RETINA,
        is_mobile=True,
        has_touch=True,
        color_scheme="dark",
    )
    page.goto(site)
    if css:
        page.add_style_tag(content=css)
    page.wait_for_selector('.wall [data-j="999"]')
    page.wait_for_timeout(600)
    shot = page.screenshot(clip={"x": 0, "y": 0, "width": PHONE["width"], "height": 600})
    page.close()
    return shot


def test_a_phone_in_dark_mode_does_not_repaint_the_board(chromium, site, browser):
    """Declining `prefers-color-scheme` stops the page turning dark. It does not stop the
    browser turning it dark on the page's behalf.

    With no `color-scheme` declared, a phone in dark mode repainted the board: the prose came
    back white on #121212 while the accent, the three primaries and every fill inside the two
    charts stayed as printed, so the labels over the stack were dark ink at 62% on bands that
    had not moved -- grey on grey. Not the board in other colours; the board in two colour
    schemes at once.
    """
    play, launch, _ = chromium
    dark = play.chromium.launch(**launch, args=FORCE_DARK)
    try:
        assert board_shot(dark, site) == board_shot(browser, site), (
            "a phone forcing dark mode repainted the board"
        )
    finally:
        dark.close()


def test_only_light_is_the_half_that_does_the_work(chromium, site, browser):
    """`color-scheme: light` says which schemes the page supports and a browser forcing dark
    overrides it anyway. `only` is the word that withholds the permission, so this asserts the
    weaker spelling really is weaker rather than leaving the choice to taste."""
    play, launch, _ = chromium
    dark = play.chromium.launch(**launch, args=FORCE_DARK)
    try:
        daylight = board_shot(browser, site)
        assert board_shot(dark, site, css=":root { color-scheme: light; }") != daylight
        assert board_shot(dark, site, css=":root { color-scheme: only light; }") == daylight
    finally:
        dark.close()


def grotesk_stack():
    css = (ROOT / "index.html").read_text(encoding="utf-8")
    return css.split("--grotesk:")[1].split(";")[0]


def test_the_page_does_not_ask_for_a_face_it_cannot_vouch_for():
    """A stack is a list of wishes and the browser grants the first it can.

    `Inter` was third in it. On a phone that has Inter installed the cut installed there was a
    hairline, Firefox granted it, and every heading and figure came out at one thin stroke
    whatever weight the CSS asked for -- measured four times thinner than the same text in
    Chrome, which does not see that font at all. Roboto, system-ui and sans-serif all go
    properly bold on the same phone; they were simply queued behind it.
    """
    assert "Inter" not in grotesk_stack(), (
        "Inter is installed as a hairline on some phones and would be granted ahead of the "
        "platform faces that do carry the board's weights"
    )


def test_the_phone_face_is_named_rather_than_left_to_a_keyword():
    """Firefox does not implement `ui-sans-serif`, so left to the keywords the two browsers
    reach the phone's own sans by different routes. Naming it settles that first."""
    stack = grotesk_stack()
    for keyword in ("ui-sans-serif", "system-ui"):
        assert stack.index("Roboto") < stack.index(keyword), (
            f"Roboto must come before {keyword}, or the two browsers choose separately"
        )


# ----------------------------------------------------------------------- filtering the column


def shown(page):
    """The words the column is showing, in the order it is showing them."""
    return page.evaluate(
        "() => [...document.querySelectorAll('.wall [data-j]')]"
        ".filter(el => !el.hidden).map(el => el.textContent)"
    )


def nth_shown(page, n):
    """The `data-j` of the nth word the column is showing, which under a filter is not n."""
    return page.evaluate(
        "n => [...document.querySelectorAll('.wall [data-j]')]"
        ".filter(el => !el.hidden)[n].dataset.j",
        n,
    )


def type_query(page, text):
    page.fill(".find input", text)
    page.wait_for_timeout(200)


def test_the_field_keeps_the_words_that_begin_with_it_and_no_others(page):
    type_query(page, "th")
    words = shown(page)
    assert words, "nothing at all was left"
    assert all(w.startswith("th") for w in words), words
    # a prefix and not a substring, which is the whole of what the field promises: a word that
    # merely contains the query is gone
    assert page.evaluate(
        "() => [...document.querySelectorAll('.wall [data-j]')]"
        ".some(el => el.hidden && el.textContent.includes('th'))"
    ), "a substring match was left in the column"
    assert page.inner_text(".find .n").replace(",", "") == f"{len(words)} OF 1000"


def test_every_way_of_choosing_stays_inside_the_matches(page):
    """The bug the whole filter is built around, and it is the same bug five times.

    A row's place in the column WAS its place in the thousand -- the offsets were read off every
    row, the wheel and the arrows added one to `S.j`, and the rail divided by a thousand. Every
    one of those is that assumption written down, and a filter breaks all of them at once: the
    column shows sixty words and the next one down is not the next `j`.
    """
    type_query(page, "re")
    words = shown(page)
    assert len(words) > 5, f"too few matches for this to mean anything: {words}"

    # the arrows, while the field still has the focus
    assert chosen(page)[0] == words[0]
    page.keyboard.press("ArrowDown")
    page.wait_for_timeout(150)
    assert chosen(page)[0] == words[1]

    # the wheel, over the column
    box = page.locator(".wall").bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.wheel(0, 120)
    page.wait_for_timeout(150)
    assert chosen(page)[0] == words[2]
    assert on_the_line(page) < 2

    # the rail, pressed at its foot
    rail = page.locator(".rail").bounding_box()
    page.mouse.click(rail["x"] + rail["width"] / 2, rail["y"] + rail["height"] - 1)
    page.wait_for_timeout(150)
    assert chosen(page)[0] == words[-1]

    # and the column thrown to its end
    scroll_column(page, 0)
    assert chosen(page)[0] == words[0]
    scroll_column(page, page.evaluate("document.querySelector('.wall').scrollHeight"))
    assert chosen(page)[0] == words[-1]


def test_the_mark_on_the_rail_travels_the_matches_and_not_the_thousand(page):
    """It is the run of the column beside it. Divided by a thousand, sixty matches moved it four
    pixels and it read as broken rather than as a position."""
    type_query(page, "re")
    page.click(f'.wall [data-j="{nth_shown(page, len(shown(page)) - 1)}"]')
    page.wait_for_timeout(150)
    place = page.eval_on_selector(".rail", "e => getComputedStyle(e).getPropertyValue('--p')")
    assert float(place) == 1, f"the mark stood at {place} of the run, and the last match is 1"


def test_a_keystroke_keeps_a_word_that_still_matches(page):
    """Narrowing the column is not choosing from it. A panel that jumped on every letter would
    be useless for the thing the field is mostly for -- looking a word up beside its chart."""
    type_query(page, "re")
    page.click(f'.wall [data-j="{nth_shown(page, 4)}"]')
    page.wait_for_timeout(150)
    was = chosen(page)
    type_query(page, was[0][:3])
    assert chosen(page) == was, "the chart moved to another word on a keystroke"
    assert on_the_line(page) < 2, "the word it kept is not on the line it is chosen by"


def test_a_word_filtered_away_hands_the_choice_to_the_top_match(page):
    type_query(page, "re")
    page.click(f'.wall [data-j="{nth_shown(page, 4)}"]')
    page.wait_for_timeout(150)
    type_query(page, "th")
    assert chosen(page)[0] == shown(page)[0]
    assert on_the_line(page) < 2


def test_nothing_matches_and_the_panel_keeps_its_word(page):
    """Narrowing the column to nothing is not the choice of a different word, so the chart stays
    -- blanking it would throw away the reading the field was opened beside."""
    type_query(page, "seam")
    was = chosen(page)
    type_query(page, "seamzz")
    assert shown(page) == []
    assert chosen(page) == was
    # the run says the query back rather than "no matches": what a reader has usually done is
    # expect the middle of a word to count, and their own letters are what explains the box
    run = page.eval_on_selector(".wall", "e => getComputedStyle(e, '::before').content")
    assert "seamzz" in run.lower(), run
    # and the marks that say the column moves go, the way a scrollbar goes
    assert not page.locator(".steps").is_visible()


def test_clearing_the_field_gives_the_thousand_back_and_keeps_the_word(page):
    type_query(page, "re")
    page.click(f'.wall [data-j="{nth_shown(page, 3)}"]')
    page.wait_for_timeout(150)
    was = chosen(page)
    page.click(".find .clear")
    page.wait_for_timeout(200)
    assert len(shown(page)) == 1000
    assert chosen(page) == was
    assert on_the_line(page) < 2
    assert page.inner_text(".find .n").strip() == "", "a field with no filter printed a reading"


def test_escape_empties_the_field_and_the_slash_reaches_it(page):
    page.click("h1")
    page.keyboard.press("/")
    page.wait_for_timeout(100)
    assert page.evaluate("() => document.activeElement.tagName") == "INPUT"
    assert page.input_value(".find input") == "", "the slash was typed into the field it opened"
    page.keyboard.type("re")
    page.wait_for_timeout(200)
    assert len(shown(page)) < 1000
    page.keyboard.press("Escape")
    page.wait_for_timeout(200)
    assert page.input_value(".find input") == ""
    assert len(shown(page)) == 1000


def test_the_cross_stands_in_the_column_the_arrows_stand_in(page):
    """Three marks down the right edge of one object, so they share a vertical rather than
    nearly sharing one: the cross is laid out inside the field's border and the marks are
    positioned outside the column's, which is a pixel between them if it is not corrected."""
    type_query(page, "re")
    centres = page.evaluate("""() => {
      const mid = s => { const r = document.querySelector(s).getBoundingClientRect();
                         return r.left + r.width / 2; };
      return [mid('.find .clear'), mid('.steps .step'), mid('.rail')];
    }""")
    assert max(centres) - min(centres) < 1, f"the three marks do not line up: {centres}"


def test_the_field_does_not_take_a_row_of_words_from_the_column(browser, site):
    """Stacked, the column and the chart it feeds are one pair at one size. The field is added to
    the cell rather than taken out of the column, or the shorter half of the pair pays for it."""
    page = browser.new_page(viewport=PHONE, device_scale_factor=RETINA)
    page.goto(site)
    page.wait_for_selector('.wall [data-j="999"]')
    page.wait_for_timeout(200)
    box = page.evaluate("""() => {
      const h = s => document.querySelector(s).getBoundingClientRect().height;
      return {words: h('.cell.words'), find: h('.find'), panel: h('.cell.panel')};
    }""")
    page.close()
    assert abs((box["words"] - box["find"]) - box["panel"]) < 1.5, box


# ---------------------------------------------------------------- who the arrow keys belong to


def test_an_arrow_key_moves_the_choice_by_one_word_where_the_page_scrolls(browser, site):
    """Two words a press in Safari, three in Firefox, and one here.

    The guard used to stand aside wherever the PAGE scrolls, on the grounds that the arrows were
    the page's there. They were not. A click inside a scroller is what makes it the browser's
    keyboard scroll target, so the arrows went to the COLUMN and scrolled it by a LINE -- which
    is two or three of these rows -- while the page they were supposedly given to never moved.

    Chromium was the one engine that looked right, and only because it snaps a keyboard scroll to
    the nearest snap point: the bug was invisible in the browser these tests run in. So what is
    asserted is not only the step but that the key was HANDLED, which is the half no engine can
    then add a scroll of its own to.
    """
    page = browser.new_page(viewport=PHONE, device_scale_factor=RETINA)
    page.goto(site)
    page.wait_for_selector('.wall [data-j="999"]')
    assert page.evaluate("() => document.documentElement.scrollHeight > innerHeight + 1"), (
        "this has to be a layout the page itself scrolls, or the guard is not under test"
    )
    # registered after the page's own, on the same target and phase, so it runs after it
    page.evaluate("""() => {
      window.__handled = null;
      addEventListener('keydown', ev => { window.__handled = ev.defaultPrevented; });
    }""")
    page.eval_on_selector(".wall", "e => e.scrollIntoView({block: 'center'})")
    page.wait_for_timeout(200)
    for start in (3, 300, 800):
        page.click(f'.wall [data-j="{start}"]')
        page.wait_for_timeout(200)
        # the click focuses the column, which is what makes "whose key is it" answerable at all
        assert page.evaluate("() => document.activeElement === document.querySelector('.wall')"), (
            "the column is not a tab stop, so a click on a word focused nothing"
        )
        for n in range(1, 4):
            page.keyboard.press("ArrowDown")
            page.wait_for_timeout(160)
            assert page.evaluate("window.__handled") is True, "the key was left to the browser"
            assert chosen(page)[1] == str(start + n), f"from {start}, press {n}"
            assert on_the_line(page) < 2
    page.close()


def test_the_arrows_stay_the_pages_until_the_column_is_touched(browser, site):
    """The half of the old guard that was right, and it is kept: on a window narrow enough to
    scroll, a reader who has not been near the column is scrolling the window."""
    page = browser.new_page(viewport=PHONE, device_scale_factor=RETINA)
    page.goto(site)
    page.wait_for_selector('.wall [data-j="999"]')
    was = chosen(page)
    before = page.evaluate("scrollY")
    for _ in range(3):
        page.keyboard.press("ArrowDown")
        page.wait_for_timeout(150)
    assert page.evaluate("scrollY") > before, "the arrows did not scroll the page"
    assert chosen(page) == was, "the arrows moved the choice as well as the page"
    page.close()


# ------------------------------------------------------------------ what the field says it does


def test_the_slash_is_named_only_where_there_is_a_keyboard_for_it(browser, site):
    """A shortcut printed on a phone is a line of furniture nobody can press. It is drawn behind
    the test the rail's HANDLE is drawn behind: a fine pointer is the nearest thing a page can
    ask about a keyboard."""
    for mobile, wanted in ((False, True), (True, False)):
        page = browser.new_page(
            viewport=PHONE if mobile else BOARD,
            device_scale_factor=RETINA,
            is_mobile=mobile,
            has_touch=mobile,
        )
        page.goto(site)
        page.wait_for_selector('.wall [data-j="999"]')
        drawn = page.eval_on_selector(".find .key", "e => getComputedStyle(e).display") != "none"
        assert drawn is wanted, f"is_mobile={mobile} drew the key: {drawn}"
        if wanted:
            # and it is spent the moment it is taken, or on a query wanting the room
            page.fill(".find input", "re")
            page.wait_for_timeout(200)
            assert page.eval_on_selector(".find .key", "e => getComputedStyle(e).display") == "none"
        page.close()


def test_both_ends_of_the_column_still_reach_the_line(page):
    """The line sits a third of the way down the box rather than half, so the empty run above the
    words is a third of a box and the one below is two thirds. Cut wrong, the word the page is
    named after cannot be brought to the line it is chosen by."""
    scroll_column(page, 0)
    assert chosen(page)[1] == "0"
    assert on_the_line(page) < 2
    scroll_column(page, page.evaluate("document.querySelector('.wall').scrollHeight"))
    assert chosen(page)[1] == "999"
    # and it lands on it, which it did not quite do when the run below was half the box: the
    # last word is the smallest on the wall and used to stop three pixels short of the line
    assert on_the_line(page) < 2


# ------------------------------------------------------------------- stepping between clusters


def cluster(page):
    """The cluster the board is on, as a number.

    The stepper pads it to the width of the count -- the arrows must not move when the reader
    steps between the ninth cluster and the tenth -- and the pad is the strip's typography rather
    than part of the reading, so it is dropped here and every test below says `3` and not `03`.
    """
    return str(int(page.inner_text(".pager .at b").strip()))


def band_fills(page):
    """What every band in the stack is painted, bottom upwards."""
    return page.evaluate(
        "() => [...document.querySelectorAll('#stack [data-c]')].map(p => p.getAttribute('fill'))"
    )


def step_cluster(page, key, n=1):
    for _ in range(n):
        page.keyboard.press(key)
        # the fills carry a transition, and a fill read back mid-transition is neither colour
        page.wait_for_timeout(260)


def test_stepping_moves_the_whole_board_and_not_just_the_chart(page):
    """The point of the stepper. A step that only recoloured a band would leave red meaning two
    things at once -- the band being pointed at, and the cluster the column and the panel are
    still of -- which is worse than not stepping at all."""
    first = word_at(page, 0)
    step_cluster(page, "ArrowRight")
    assert cluster(page) == "2"
    # the fill has moved one band up the stack, which is the order the clusters arrive in
    assert band_fills(page).index("var(--accent)") == 1
    assert word_at(page, 0) != first
    assert chosen(page) == (word_at(page, 0), "0")


def test_a_step_repaints_two_bands_and_leaves_the_other_ten(page):
    """The greys are fixed to a band's place in the stack. Dealt out to whichever bands are not
    chosen -- a ramp of eleven and a counter that skipped the lead, which is what this was --
    every band above the selection took its neighbour's shade the moment the selection moved,
    and one step recoloured the whole chart to say one thing."""
    before = band_fills(page)
    step_cluster(page, "ArrowRight")
    after = band_fills(page)
    moved = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert moved == [0, 1], f"a step repainted {len(moved)} bands: {moved}"


def test_only_one_band_is_ever_red(page):
    """The arrival was given a red edge of its own to keep while the fill was elsewhere. In a
    chart whose whole grammar is that red is where you are, a second red mark reads as a pair
    rather than as a subject and a footnote."""
    n = page.evaluate("() => window.ANALYSIS.components.length")
    for i in range(n):
        if i:
            step_cluster(page, "ArrowRight")
        red = page.evaluate(
            "() => [...document.querySelectorAll('#stack *')].filter(e =>"
            " [e.getAttribute('stroke'), e.getAttribute('fill')].join(' ').includes('--accent')"
            ").length"
        )
        assert red == 1, f"cluster {i + 1} drew {red} red marks"


def test_the_ends_are_ends(page):
    """The clusters are ordered by how much of the last month each one is, so the first and the
    twelfth are the two extremes of that reading and stepping between them would be the one move
    on this stepper that means nothing. An arrow with nowhere to go says so rather than being
    pressed twice and blamed."""
    disabled = lambda: page.evaluate(  # noqa: E731
        "() => [...document.querySelectorAll('.pager .arrow')].map(b => b.disabled)"
    )
    assert disabled() == [True, False]
    step_cluster(page, "ArrowLeft")
    assert cluster(page) == "1", "the first cluster has no cluster before it"
    n = page.evaluate("() => window.ANALYSIS.components.length")
    step_cluster(page, "ArrowRight", n + 2)
    assert cluster(page) == str(n)
    assert disabled() == [False, True]


def test_left_and_right_are_the_caret_while_the_field_has_the_focus(page):
    """They are global -- unlike up and down they scroll nothing on this page, so there is
    nothing to take them from -- and a text field is the one place that is not true."""
    page.click(".find input")
    page.fill(".find input", "re")
    page.keyboard.press("ArrowLeft")
    page.keyboard.press("ArrowRight")
    page.wait_for_timeout(200)
    assert cluster(page) == "1"
    assert page.input_value(".find input") == "re"


def test_a_query_survives_a_step_and_is_asked_of_the_next_cluster(page):
    """A query is a question about the vocabulary, and the natural next thing to do with one is
    to ask it of the next cluster. Clearing it on a step would make stepping and searching two
    modes rather than two hands on the same object."""
    step_cluster(page, "ArrowRight")
    word = word_at(page, 0)
    step_cluster(page, "ArrowLeft")
    type_query(page, word[:2])
    # out of the field first: while it has the focus the arrows are its caret, which is the
    # test above this one. Enter is what the field means by done.
    page.keyboard.press("Enter")
    step_cluster(page, "ArrowRight")
    assert page.input_value(".find input") == word[:2]
    assert word in shown(page)
    assert all(w.startswith(word[:2]) for w in shown(page))


def test_the_stepper_does_not_change_the_height_of_the_row_it_is_in(page):
    """The chart is flex-sized under that row, so a row that grew when the readout was written
    into would shrink the chart under the pointer that was reading it. The ten pixels under the
    row belong to the ROW: left on the readout, its margin box and the arrows' box were two
    different heights and the row was as tall as whichever was taller."""
    box = "() => Math.round(document.querySelector('#stack').getBoundingClientRect().height)"
    idle = page.evaluate(box)
    page.hover("#stack")
    page.mouse.move(
        700,
        page.evaluate("() => document.querySelector('#stack').getBoundingClientRect().top + 40"),
    )
    page.wait_for_timeout(150)
    assert "week" in page.inner_text(".cell.chart .readout").lower()
    assert page.evaluate(box) == idle


def test_every_cluster_can_draw_a_word(page):
    """`analyze.py` carried the weekly counts of a word for the lead component alone, which was
    right while the board was about one component. Twelve of them and the panel answers for one
    cluster and goes blank for the other eleven."""
    n = page.evaluate("() => window.ANALYSIS.components.length")
    for i in range(n):
        if i:
            step_cluster(page, "ArrowRight")
        assert cluster(page) == str(i + 1)
        drawn = page.evaluate(
            "() => [...document.querySelectorAll('.probe svg path')].map(p => p.getAttribute('d'))"
        )
        assert drawn and all(len(d) > 20 for d in drawn), f"cluster {i + 1} drew {drawn}"
        assert "more frequent" in page.inner_text(".probe .meta").lower()


def test_the_rail_is_a_reading_everywhere_and_a_handle_only_where_it_can_be_taken(browser, site):
    """It was drawn only where the pointer is fine, which confused two questions: whether the
    mark can be TAKEN, and whether it can be READ. A finger cannot catch eight pixels -- it has
    the whole column to throw, which is the same gesture with a target the size of the box --
    but a reader on a phone needs to know where in a thousand words the column has got to just
    as much, and was told nothing."""
    for mobile in (False, True):
        page = browser.new_page(
            viewport=PHONE if mobile else BOARD,
            device_scale_factor=RETINA,
            is_mobile=mobile,
            has_touch=mobile,
        )
        page.goto(site)
        page.wait_for_selector('.wall [data-j="999"]')
        drawn, takes = page.eval_on_selector(
            ".rail", "e => { const s = getComputedStyle(e); return [s.display, s.pointerEvents]; }"
        )
        assert drawn != "none", f"is_mobile={mobile} drew no mark on the frame"
        assert (takes == "auto") is not mobile, f"is_mobile={mobile} takes the pointer: {takes}"
        if mobile:
            # and the frame stays a frame: a finger landing on it is not a place in the column
            before = chosen(page)
            box = page.locator(".rail").bounding_box()
            page.touchscreen.tap(box["x"] + box["width"] / 2, box["y"] + box["height"] - 2)
            page.wait_for_timeout(250)
            assert chosen(page) == before, "a finger on the frame moved the column"
        page.close()


# ------------------------------------------------------- the strip as a surface, not two marks


def drag_strip(page, dx, steps=12, drift=0, start=None):
    """A finger dragged `dx` across the stepper, and the cluster it stood on at each move.

    Driven through CDP for the reason `drag_across` is: the whole question here is what the
    BROWSER does with a sideways gesture on a strip it has also been asked to scroll past, and a
    hand-built event never gives it that decision to make.
    """
    cdp = page.context.new_cdp_session(page)
    page.eval_on_selector(".pager", "e => e.scrollIntoView({block: 'center'})")
    page.wait_for_timeout(250)
    box = page.locator(".pager").bounding_box()
    x0 = box["x"] + (box["width"] / 2 if start is None else start)
    y = box["y"] + box["height"] / 2
    cdp.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": x0, "y": y}]})
    seen = []
    for i in range(1, steps + 1):
        cdp.send(
            "Input.dispatchTouchEvent",
            {
                "type": "touchMove",
                "touchPoints": [
                    {"x": x0 + dx * i / steps, "y": y + (drift if i > steps / 2 else 0)}
                ],
            },
        )
        page.wait_for_timeout(35)
        seen.append(cluster(page))
    cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
    page.wait_for_timeout(200)
    return seen


def test_the_strip_names_the_gesture_the_device_has(browser, site):
    """The two arrows are the two arrows on both devices, but the way to the next cluster is the
    whole strip on a phone and the arrows alone on a mouse. A mouse told to swipe would be told to
    do something it cannot; a finger not told to would never find out the strip answers at all."""
    for mobile, wanted in ((False, "change cluster"), (True, "swipe to change cluster")):
        page = touch_page(browser, site) if mobile else browser.new_page(viewport=BOARD)
        if not mobile:
            page.goto(site)
            page.wait_for_selector('.wall [data-j="999"]')
        assert page.inner_text(".pager .says").strip().lower() == wanted
        # and it is still the truth after a step, which is when it is re-read
        page.keyboard.press("ArrowRight")
        page.wait_for_timeout(260)
        assert page.inner_text(".pager .says").strip().lower() == wanted
        page.close()


def test_a_swipe_across_the_strip_steps_one_cluster(browser, site):
    """The strip is the surface and not just the two marks on it: a phone has the whole of it to
    swipe, which is the gesture it already holds for "the next one of these" and the one the arrows
    are the smallest possible target for.

    One swipe is one cluster. The strip is not a thing to scroll -- it is 350 pixels long and the
    clusters it steps through are ten, so a distance that carried several steps made the gesture a
    guess and a flick landed two or three clusters from where it was aimed.
    """
    page = touch_page(browser, site)
    # a long drag, and the drift is the second half of it: the strip is one line tall and a thumb
    # dragged sideways leaves it, which must not make the one step it earned come out as none
    seen = drag_strip(page, -300, steps=24, drift=70)
    assert seen[0] == "1", f"the first millimetres of the drag already stepped: {seen}"
    assert seen[-1] == "2", f"a swipe across the whole strip reached cluster {seen[-1]}"
    assert set(seen) == {"1", "2"}, f"the board moved more than once: {seen}"
    page.close()


def test_the_rest_of_the_drag_is_not_a_second_swipe(browser, site):
    """The gesture is spent when it is answered. A finger that goes on travelling, changes its mind
    and comes back is one swipe still, and the cluster it earned stays earned: a strip that undid
    itself half way through would make the reader hold still to keep an answer they already had."""
    page = touch_page(browser, site)
    cdp = page.context.new_cdp_session(page)
    page.eval_on_selector(".pager", "e => e.scrollIntoView({block: 'center'})")
    page.wait_for_timeout(250)
    box = page.locator(".pager").bounding_box()
    x0, y = box["x"] + box["width"] - 20, box["y"] + box["height"] / 2
    cdp.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": x0, "y": y}]})
    seen = []
    # out to the far end of the strip and all the way back past where it started
    for i in list(range(1, 11)) + list(range(9, -3, -1)):
        cdp.send(
            "Input.dispatchTouchEvent",
            {"type": "touchMove", "touchPoints": [{"x": x0 - i * 30, "y": y}]},
        )
        page.wait_for_timeout(30)
        seen.append(cluster(page))
    cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
    page.wait_for_timeout(200)
    # a run of the cluster it started on, then a run of the next one, and nothing else: the first
    # move of the drag is under the travel a swipe costs, so the board has not stepped yet there
    assert seen[-1] == "2", f"the drag lost the cluster it had earned: {seen}"
    assert set(seen) <= {"1", "2"}, f"one drag was answered more than once: {seen}"
    assert seen == sorted(seen, key=int), f"the drag stepped and then stepped back: {seen}"
    # and lifting the finger arms it again: the second swipe is a second cluster
    assert drag_strip(page, -80, steps=6)[-1] == "3", "a second swipe did not step"
    page.close()


def test_a_swipe_that_ends_on_an_arrow_is_not_also_a_press_of_it(browser, site):
    """The board would step once for the travel and once more for the release, which is the one
    thing that could still take a reader two clusters from a single gesture."""
    page = touch_page(browser, site)
    page.keyboard.press("ArrowRight")
    page.keyboard.press("ArrowRight")
    page.wait_for_timeout(300)
    assert cluster(page) == "3"
    arrow = page.locator(".pager .arrow").nth(1).bounding_box()
    box = page.locator(".pager").bounding_box()
    # begins inside the strip and lifts over the arrow that steps FORWARD, while the drag itself
    # is going back: a press of it as well would show up as a step in the wrong direction
    reach = arrow["x"] + arrow["width"] / 2 - box["x"] - 130
    assert drag_strip(page, 130, steps=13, start=reach)[-1] == "2"
    page.close()


def test_a_flick_is_enough_and_a_twitch_is_not(browser, site):
    """A flick is short, so the travel a swipe costs is about a finger's width -- long enough that
    the wander in a tap does not spend it, short enough that a flick does. Past that the distance
    says nothing: every swipe is worth the same one cluster, however far it runs on."""
    for dx, wanted in ((-14, "1"), (-40, "2"), (-100, "2"), (-170, "2")):
        page = touch_page(browser, site)
        # from the right-hand end, so a drag this long has strip left to travel across
        assert (
            drag_strip(
                page, dx, steps=4, start=0.75 * page.locator(".pager").bounding_box()["width"]
            )[-1]
            == wanted
        ), f"{dx}px of travel was not worth cluster {wanted}"
        page.close()


@pytest.mark.parametrize("drift", [0, 30, 60])
def test_the_page_still_scrolls_over_the_strip(browser, site, drift):
    """The sideways axis is the strip's; the up-and-down one stays the page's. The strip sits in
    the middle of a board that is scrolled past far more often than it is swiped.

    The drift is the half that a threshold of half a cluster puts at risk: a thumb pulled down the
    board does not travel in a straight line, and twenty-four pixels of wander is nothing. It is
    `pan-y` that settles it rather than the arithmetic -- the browser claims the gesture the moment
    it reads as a scroll, and what arrives here afterwards is a `pointercancel`.
    """
    page = touch_page(browser, site)
    cdp = page.context.new_cdp_session(page)
    page.eval_on_selector(".pager", "e => e.scrollIntoView({block: 'center'})")
    page.wait_for_timeout(250)
    box = page.locator(".pager").bounding_box()
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    before, was = page.evaluate("scrollY"), cluster(page)
    cdp.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": x, "y": y}]})
    for i in range(1, 13):
        cdp.send(
            "Input.dispatchTouchEvent",
            {"type": "touchMove", "touchPoints": [{"x": x + drift * i / 12, "y": y - i * 15}]},
        )
        page.wait_for_timeout(20)
    cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
    page.wait_for_timeout(400)
    assert page.evaluate("scrollY") > before + 50, "a finger dragged up the strip did not scroll"
    assert cluster(page) == was, "a finger on its way down the board stepped the clusters"
    page.close()


def test_a_press_of_an_arrow_is_not_held_in_case_a_second_one_is_coming(browser, site):
    """Why the stepper felt slow on a phone, and it was not the drawing.

    Left at `touch-action: auto` iOS holds every tap for a third of a second in case it turns out
    to be the first half of a double tap to zoom. Any narrower value says there is no such gesture
    here and the press goes through when the finger lifts.

    This one is asserted as a property rather than acted out, because the browser the tests run in
    does not have the bug to act out: Chrome dropped the wait for any page whose viewport is
    `width=device-width`, so a tap here is prompt either way and only the property says why.

    `touch-action` does not inherit, so the arrows compute `auto` and it means nothing: what a
    gesture is allowed to do is the intersection of the values down the chain it was hit through,
    and the strip is on that chain. Which is what the second assertion is for -- the strip's value
    covers the arrows only for as long as the arrows are inside it.
    """
    page = touch_page(browser, site)
    assert page.eval_on_selector(".pager", "e => getComputedStyle(e).touchAction") != "auto"
    assert page.eval_on_selector_all(".pager .arrow", "es => es.length") == 2, (
        "the arrows are outside the strip whose gesture rules cover them"
    )
    page.close()


def test_the_arrow_does_not_stay_lit_after_the_press_is_over(browser, site):
    """A phone has no leave event to give. iOS holds `:hover` on whatever was last touched until
    something else is, so the arrow that stepped the board stayed lit afterwards and read as the
    state of the board rather than as a press that had already happened.

    Both states are forced through CDP rather than acted out: the sticky half of `:hover` is a
    phone's behaviour and not a headless browser's, so a tap here would look right either way.
    """
    for mobile, hover_lights in ((False, True), (True, False)):
        page = touch_page(browser, site) if mobile else browser.new_page(viewport=BOARD)
        if not mobile:
            page.goto(site)
            page.wait_for_selector('.wall [data-j="999"]')
        cdp = page.context.new_cdp_session(page)
        cdp.send("DOM.enable")
        cdp.send("CSS.enable")
        root = cdp.send("DOM.getDocument")["root"]["nodeId"]
        node = cdp.send(
            "DOM.querySelector", {"nodeId": root, "selector": ".pager .arrow + .at + .arrow"}
        )["nodeId"]
        accent = page.evaluate(
            "() => getComputedStyle(document.documentElement).getPropertyValue('--accent').trim()"
        )
        lit = lambda: page.eval_on_selector(  # noqa: E731
            ".pager .arrow:last-of-type", "e => getComputedStyle(e).color"
        )
        as_hex = page.evaluate(
            """a => { const d = document.createElement('i'); d.style.color = a;
                      document.body.append(d); const c = getComputedStyle(d).color;
                      d.remove(); return c; }""",
            accent,
        )
        for state, wanted in (("hover", hover_lights), ("active", True)):
            cdp.send("CSS.forcePseudoState", {"nodeId": node, "forcedPseudoClasses": [state]})
            page.wait_for_timeout(60)
            assert (lit() == as_hex) is wanted, f"is_mobile={mobile} :{state} drew {lit()}"
            cdp.send("CSS.forcePseudoState", {"nodeId": node, "forcedPseudoClasses": []})
        page.close()


def test_a_mouse_dragged_across_the_strip_does_not_step(page):
    """A mouse has the arrows under it and two keys beside them. A board where dragging the corner
    of the figure swept through the clusters would be answering a gesture nobody made."""
    box = page.locator(".pager").bounding_box()
    y = box["y"] + box["height"] / 2
    page.mouse.move(box["x"] + box["width"] - 60, y)
    page.mouse.down()
    for i in range(1, 8):
        page.mouse.move(box["x"] + box["width"] - 60 - i * 30, y)
        page.wait_for_timeout(20)
    page.mouse.up()
    page.wait_for_timeout(250)
    assert cluster(page) == "1"
