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
    asked for in the event, which is five of them."""
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
    the same test as the rail: a fine pointer is the nearest thing a page can ask about a
    keyboard."""
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
