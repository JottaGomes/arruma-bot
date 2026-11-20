#!/usr/bin/env python3
"""Plays the "Arruma a Loja" shelf-sorting game of the Continente app over adb.

Each shelf holds three products. Three identical products on one shelf clear it, and
the products stored behind them (drawn darker) move to the front. The bot repeatedly:
captures the screen, finds the shelves, groups identical products by comparing their
pixels (no product catalogue is needed because identical products are drawn with
identical pixels), chooses one drag with a greedy strategy and performs it.

Usage:
    .venv/bin/python bot.py debug     # analyse one screenshot, write debug.png, no input
    .venv/bin/python bot.py step      # make five drags, saving and checking the board around each
    .venv/bin/python bot.py play      # play until 50% completion, then hand over to manual play
    .venv/bin/python bot.py full      # play the whole level (still stops when stuck)
    .venv/bin/python bot.py coop      # play the whole level, leaving the right column to a person
    .venv/bin/python bot.py top       # play only the nine fixed shelves, leaving the belt to a person
"""

import io
import os
import random
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

# adb serial or ip:port of the phone. When unset, adb uses its only connected device.
DEVICE = os.environ.get("ARRUMA_DEVICE")
# Package of the Continente app. Every drag is preceded by a focus check so that a phone
# call or notification that takes over the screen never receives the drags.
APP_PACKAGE = "pt.continente.continentetelemovel"

# Horizontal centres of the nine slots (three shelves of three slots) on a 1080 px wide
# screen, measured from screenshots of the game.
SLOT_X = [140, 228, 314, 456, 541, 626, 770, 856, 942]
# Column used to find the shelves: it crosses the grey left frame of the first column of
# shelves, which is the only grey there; everything else at that x is beige background.
FRAME_PROBE_X = 70
SLOT_HALF_WIDTH = 34
# Region of a slot used to identify its front product, as fractions of the shelf height and a
# half width in px. It is kept low and narrow because products stored behind (level 23) peek
# out above and beside the front one and would make copies of a product look different.
CROP_TOP, CROP_BOTTOM = 0.65, 0.93
CROP_HALF_WIDTH = 26
# Pixels of the grey veil over hidden products (little colour, not bright) are blanked before
# products are compared, for the same reason.
VEIL_MAX_CHROMA, VEIL_MAX_BRIGHTNESS = 45, 175

# A crop is empty when fewer than this share of its pixels differ from the white shelf.
EMPTY_FRACTION = 0.12
# Products stored behind the front row are drawn darkened. A crop whose brightest pixels
# stay below this value holds only such a product, so its front slot is free.
# Upper brightness of a hidden product: level 23 hides products under a lighter veil (about
# 120) than earlier levels (about 70), so the chroma check below is what tells them apart.
DARK_BRIGHTNESS = 160
# Hidden products under the veil keep little colour (chroma about 30); a dark front product
# such as the red cone tree of level 23 keeps about 80.
DARK_MAX_CHROMA = 50
# Strongest colour (99th percentile chroma) a hidden product can show: about 40 at most on
# levels 21-23, against about 160 on the red label of a black soy sauce bottle.
HIDDEN_MAX_PEAK_CHROMA = 60
# Colour spread (max - min channel, 0-255) below which a slot shows only a grey ghost.
GHOST_MAX_CHROMA = 25


# Label for a slot holding a product that is not identified.
UNKNOWN = -1


def adb(*args, binary=False):
    target = ["-s", DEVICE] if DEVICE else []
    out = subprocess.run(["adb", *target, *args], capture_output=True, check=True)
    return out.stdout if binary else out.stdout.decode()


def capture():
    """Screenshot of the phone. img.info["t"] holds the moment it was taken (the midpoint of
    the adb call), which positions the moving belt shelves at drag time."""
    start = time.time()
    data = adb("exec-out", "screencap", "-p", binary=True)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.info["t"] = (start + time.time()) / 2
    return img


def app_in_focus():
    # mCurrentFocus is the window receiving input; mFocusedApp stays on the game even while
    # the lock screen or an incoming call sits on top, so only the former is checked.
    focus = adb("shell", "dumpsys", "window", "|", "grep", "mCurrentFocus")
    return APP_PACKAGE in focus.lower()


def is_fixed_row(px, top, bottom):
    """True when the inner left walls of the row sit where fixed shelves have them."""
    runs = wall_runs(px, top, bottom)
    return all(any(abs(r[1] - edge) < 12 for r in runs) for edge in FIXED_INNER_WALLS)


# Inner edges (x) of the left walls of the three fixed shelves in a row.
FIXED_INNER_WALLS = (88, 403, 718)


def find_shelf_rows(px):
    """Returns (top, bottom) pixel rows of each row of shelves, top to bottom."""
    column = px[:, FRAME_PROBE_X, :].astype(int)
    spread = column.max(axis=1) - column.min(axis=1)
    grey = (spread < 18) & (column[:, 0] > 120) & (column[:, 0] < 225)
    rows, start = [], None
    for y, is_grey in enumerate(grey):
        if is_grey and start is None:
            start = y
        elif not is_grey and start is not None:
            # Shorter grey runs are icons or text. A belt shelf whose wall happens to cross the
            # probe column also gives a long run; fixed rows are told apart by their inner walls,
            # which always sit at the same three positions.
            # Rows already measured as belts are never fixed, even in a frame where their shelves
            # line up with the fixed positions for a moment.
            known_belt = any(abs(start - top) < 15 for top in BELT_SPEEDS)
            if y - start > 200 and not known_belt and is_fixed_row(px, start, y):
                rows.append((start, y))
            start = None
    return rows


# Some levels add a belt: a row of shelves under the fixed ones that scrolls sideways at a
# constant speed. Belt shelves hold products like any other shelf and can be dragged from and
# to while they are fully on screen. Belt shelves have the same size as fixed ones; the slot
# centres sit at these offsets from the inner edge of the shelf's left wall.
SLOT_OFFSETS = (53, 139, 224)
SHELF_INNER_WIDTH = 275
# Belt speed in px/s, positive to the right; measured at the start of play().
BELT_SPEED = 113.0
# Measured speed of each belt row, keyed by the row's top pixel, for levels with several belts.
BELT_SPEEDS = {}
# A belt shelf is only used when it stays fully on screen for this long after the screenshot,
# which covers deciding, the adb latency and the drag itself.
BELT_SAFE_SECONDS = 1.6
# Vertical gap between rows of shelves.
ROW_GAP = 38


def wall_runs(px, top, bottom):
    """Inner left edges of the shelves in one row, from the grey side walls at 30 % height."""
    row = px[top + int((bottom - top) * 0.3)].astype(int)
    spread = row.max(axis=1) - row.min(axis=1)
    grey = (spread < 18) & (row[:, 0] > 140) & (row[:, 0] < 245)
    runs, start = [], None
    for x, is_grey in enumerate(grey):
        if is_grey and start is None:
            start = x
        elif not is_grey and start is not None:
            if x - start >= 4:
                runs.append((start, x))
            start = None
    if start is not None:
        runs.append((start, len(grey)))
    return runs


def shelf_geometry(px, captured_at):
    """One entry per shelf: dict(top, bottom, xs (slot centres), belt, t)."""
    rows = find_shelf_rows(px)
    shelves = []
    for top, bottom in rows:
        for column in range(3):
            shelves.append(dict(top=top, bottom=bottom, belt=False, t=captured_at,
                                xs=[SLOT_X[column * 3 + k] for k in range(3)]))
    # Belt rows sit under the last fixed row (levels 22 and others) or between two fixed rows,
    # where the gap is tall enough to hold whole rows (level 26, two belts running in opposite
    # directions). The wall check below rejects a row of fixed shelves.
    belt_rows = []
    if len(rows) >= 2:
        height = rows[-1][1] - rows[-1][0]
        for (_, upper_bottom), (lower_top, _) in zip(rows, rows[1:]):
            top = upper_bottom + ROW_GAP
            while top + height <= lower_top:
                belt_rows.append((top, top + height))
                top += height + ROW_GAP
        top = rows[-1][1] + ROW_GAP
        if top + height < px.shape[0]:
            belt_rows.append((top, top + height))
    for top, bottom in belt_rows:
        runs = wall_runs(px, top, bottom)
        fixed_walls = all(any(abs(r[1] - edge) < 12 for r in runs) for edge in (88, 403, 718))
        if len(runs) < 2 or fixed_walls:
            continue
        speed = BELT_SPEEDS.get(top, BELT_SPEED)
        for _, inner_left in runs:
            right = inner_left + SHELF_INNER_WIDTH
            travel = speed * BELT_SAFE_SECONDS
            if inner_left + min(travel, 0) < 5 or right + max(travel, 0) > px.shape[1] - 5:
                continue
            shelves.append(dict(top=top, bottom=bottom, belt=True, t=captured_at, speed=speed,
                                xs=[inner_left + o for o in SLOT_OFFSETS]))
    return shelves


def slot_crop(px, shelf, k):
    """Lower part of the product area of one slot, where only the front product is drawn."""
    top, bottom = shelf["top"], shelf["bottom"]
    height = bottom - top
    y0, y1 = top + int(height * CROP_TOP), top + int(height * CROP_BOTTOM)
    x = shelf["xs"][k]
    return px[y0:y1, x - CROP_HALF_WIDTH:x + CROP_HALF_WIDTH, :]


def occupancy_crop(px, shelf, k):
    """Wider and taller slot region used only to decide whether the front slot is free. The
    identity crop above is too low for that: it reaches the grey shelf floor, whose bright
    pixels make a darkened hidden product look like a front product."""
    top, bottom = shelf["top"], shelf["bottom"]
    height = bottom - top
    y0, y1 = top + int(height * 0.55), top + int(height * 0.92)
    x = shelf["xs"][k]
    return px[y0:y1, x - SLOT_HALF_WIDTH:x + SLOT_HALF_WIDTH, :]


def describe(crop):
    """Returns None for a free slot, otherwise a small float array that identifies the product."""
    c = crop.astype(int)
    not_white = (c.min(axis=2) < 225)
    if not_white.mean() < EMPTY_FRACTION:
        return None
    brightness = c.max(axis=2)[not_white]
    chroma = (c.max(axis=2) - c.min(axis=2))[not_white]
    # A dark slot is a hidden product only when its colour is also washed out by the grey
    # veil; dark but saturated sprites (the dark red cone trees of level 23) are front products.
    # Black front products (the soy sauce bottles of level 25) are as dark and colourless as a
    # hidden one, but their label keeps a few strongly coloured pixels; under the veil none do.
    if np.percentile(brightness, 90) < DARK_BRIGHTNESS and np.percentile(chroma, 90) < DARK_MAX_CHROMA \
            and np.percentile(chroma, 99) < HIDDEN_MAX_PEAK_CHROMA:
        return None
    # Some hidden products show through a free slot as a pale grey ghost instead of a dark
    # silhouette; colourless pixels throughout mean the slot holds no front product.
    chroma = (c.max(axis=2) - c.min(axis=2))[not_white]
    if np.percentile(chroma, 90) < GHOST_MAX_CHROMA:
        return None
    front = crop.astype(int).copy()
    front[((front.max(axis=2) - front.min(axis=2)) < VEIL_MAX_CHROMA) & (front.max(axis=2) < VEIL_MAX_BRIGHTNESS)] = 255
    small = Image.fromarray(front.astype(np.uint8)).resize((10, 14), Image.BILINEAR)
    return np.asarray(small, dtype=float)


# Products stored behind the front row are drawn as copies of the same sprites under a
# dark grey veil (about 85 % opaque, grey level about 47). Blending with a grey scales the
# channel differences and adds the same amount to every channel, so the hue survives while
# the saturation collapses. A hidden product is therefore recognised by the hue histogram of
# its dark pixels, using chroma (max - min channel) instead of saturation to pick and weight
# the pixels that still carry colour.
HUE_BINS = 18
HIDDEN_MAX_V = 0.45   # brightest value (0-1) a darkened pixel can have
HIDDEN_MIN_V = 0.12   # darker pixels are black outlines and carry no reliable hue
MIN_SATURATION = 0.35
MIN_HIDDEN_CHROMA = 10 / 255
MIN_HIDDEN_PIXELS = 250
MIN_HUE_SIMILARITY = 0.7


def hsv(crop):
    c = crop.astype(float) / 255.0
    v = c.max(axis=2)
    span = v - c.min(axis=2)
    s = np.where(v > 0, span / np.maximum(v, 1e-6), 0)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    safe = np.maximum(span, 1e-6)
    h = np.where(v == r, ((g - b) / safe) % 6, np.where(v == g, (b - r) / safe + 2, (r - g) / safe + 4)) / 6.0
    return h, s, v


def hue_histogram(h, weight, mask):
    hist = np.bincount((h[mask] * HUE_BINS).astype(int) % HUE_BINS, weights=weight[mask], minlength=HUE_BINS)
    total = hist.sum()
    return hist / total if total else None


def front_hue(crop):
    """Hue histogram of a front product, from the part of the slot only it covers."""
    h, s, v = hsv(crop)
    not_white = crop.astype(int).min(axis=2) < 225
    return hue_histogram(h, s * v, not_white & (s > MIN_SATURATION) & (v > HIDDEN_MIN_V))


def hidden_hue(px, shelf, k, has_front):
    """Hue histogram of the darkened product behind one slot, or None when too little shows.

    With a front product present only the strip above it is used, because there the hidden
    product peeks out and nothing else is drawn; with a free slot the whole height is used.
    """
    top, bottom = shelf["top"], shelf["bottom"]
    height = bottom - top
    x = shelf["xs"][k]
    area = px[top + int(height * 0.08):top + int(height * 0.92), x - SLOT_HALF_WIDTH - 6:x + SLOT_HALF_WIDTH + 6, :]
    h, s, v = hsv(area)
    not_white = area.astype(int).min(axis=2) < 225
    if has_front:
        bright = not_white & (v > 0.55)
        rows_with_front = np.where(bright.mean(axis=1) > 0.3)[0]
        if len(rows_with_front) == 0:
            return None
        cutoff = max(rows_with_front[0] - 4, 0)
        h, s, v, not_white = h[:cutoff], s[:cutoff], v[:cutoff], not_white[:cutoff]
    chroma = s * v
    mask = not_white & (v < HIDDEN_MAX_V) & (v > HIDDEN_MIN_V) & (chroma > MIN_HIDDEN_CHROMA)
    if mask.sum() < MIN_HIDDEN_PIXELS:
        return None
    return hue_histogram(h, chroma, mask)


def match_hidden(hist, front_hues, hidden_hues, next_label):
    """Labels one hidden product, returning (label, next_label).

    A hidden product is first compared with the front products. A product type that is not
    in front anywhere still matters when several copies hide on different shelves, so an
    unmatched histogram is compared with earlier unmatched ones and given a new shared label.
    """
    best, best_label = 0.0, UNKNOWN
    for label, ref in list(front_hues.items()) + list(hidden_hues.items()):
        similarity = np.minimum(hist, ref).sum()
        if similarity > best:
            best, best_label = similarity, label
    if best >= MIN_HUE_SIMILARITY:
        return best_label, next_label
    hidden_hues[next_label] = hist
    return next_label, next_label + 1


# Two crops show the same product when both their colour histograms and their downscaled
# pixels agree. Pixels alone confused products of one shape that differ only in label colour
# (wine bottles); colours alone confused products of one colour scheme. Each crop takes the
# closest matching prototype, not the first one under the thresholds. On level 26 copies of
# one product measure at most 0.09 (colour) and 7.2 (shape), while pizza boxes and meal trays
# measure 0.34 and 22, so the limits sit well below the latter.
SAME_COLOUR_THRESHOLD = 0.2
SAME_SHAPE_THRESHOLD = 15.0


def colour_histogram(crop):
    """Share of the non-white pixels in each of 512 coarse RGB bins."""
    mask = crop.astype(int).min(axis=2) < 225
    q = (crop[mask] // 32).astype(int)
    hist = np.bincount(q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2], minlength=512).astype(float)
    return hist / max(hist.sum(), 1)


def read_board(img, with_hidden=False):
    """Returns (rows, board), plus the hidden layer when with_hidden is set.

    board[shelf] is a list of three labels, None for a free slot. hidden[shelf] lists the
    labels guessed for the products behind each slot (UNKNOWN when unrecognised), which come
    forward once that shelf clears or is emptied.
    """
    px = np.asarray(img)
    rows = shelf_geometry(px, img.info.get("t", time.time()))
    if SKIP_BELT:
        rows = [g for g in rows if not g["belt"]]
    if TOP_HALF_ONLY:
        rows = top_half(rows)
    global FIXED_SHELVES
    FIXED_SHELVES = sum(not g["belt"] for g in rows)
    prototypes, board, front_hues = [], [], {}
    for geometry in rows:
        if True:
            shelf = []
            for k in range(3):
                if describe(occupancy_crop(px, geometry, k)) is None:
                    shelf.append(None)
                    continue
                feature = describe(slot_crop(px, geometry, k))
                if feature is None:
                    feature = describe(occupancy_crop(px, geometry, k))
                colours = colour_histogram(slot_crop(px, geometry, k))
                label, best = None, None
                for i, (proto, proto_colours) in enumerate(prototypes):
                    colour_distance = 1 - np.minimum(colours, proto_colours).sum()
                    if colour_distance < SAME_COLOUR_THRESHOLD \
                            and np.abs(proto - feature).mean() < SAME_SHAPE_THRESHOLD \
                            and (best is None or colour_distance < best):
                        label, best = i, colour_distance
                if label is None:
                    prototypes.append((feature, colours))
                    label = len(prototypes) - 1
                    hue = front_hue(occupancy_crop(px, geometry, k))
                    if hue is not None:
                        front_hues[label] = hue
                shelf.append(label)
            board.append(shelf)
    # Three identical products on one shelf clear at once, so a settled screen never shows
    # them; a read that does has merged different products into one label. Plans built on
    # that label move the same products back and forth, so it is treated as unidentified.
    merged = {shelf[0] for shelf in board
              if shelf[0] is not None and shelf[0] != UNKNOWN and shelf.count(shelf[0]) == 3}
    if merged:
        board = [[UNKNOWN if v in merged else v for v in shelf] for shelf in board]
    if not with_hidden:
        return rows, board
    hidden, hidden_hues, next_label = [], {}, len(prototypes)
    for s, shelf in enumerate(board):
        guesses = []
        for k in range(3):
            if rows[s]["belt"]:
                guesses.append(UNKNOWN)
                continue
            hist = hidden_hue(px, rows[s], k, shelf[k] is not None)
            if hist is None:
                guesses.append(UNKNOWN)
                continue
            label, next_label = match_hidden(hist, front_hues, hidden_hues, next_label)
            guesses.append(label)
        hidden.append(guesses)
    return rows, board, hidden


def slot_point(rows, shelf, k, at=None):
    """Screen point of one slot; belt slots are moved to where the belt will be at `at`."""
    geometry = rows[shelf]
    y = geometry["top"] + int((geometry["bottom"] - geometry["top"]) * 0.70)
    x = geometry["xs"][k]
    if geometry["belt"] and at is not None:
        x += geometry.get("speed", BELT_SPEED) * (at - geometry["t"])
    return int(x), y


# Plan scores. A cleared triple is the goal; emptying a shelf is the next best thing because
# it brings the hidden products forward, which is the only source of new matches once the
# visible ones run out.
CLEAR_GAIN = 100
# Extra score for a triple cleared on a fixed shelf, and on a belt shelf. Products left on the
# belt keep scrolling past and cost extra drags to fetch later, so clearing them first while
# they are on screen keeps the end of the level short; the fixed shelves stay reachable.
FIXED_CLEAR_BONUS = 0
BELT_CLEAR_BONUS = 40
REVEAL_GAIN = 45
SECOND_MOVE_WEIGHT = 0.9
# Drags in a row without a cleared or emptied shelf after which the bot stops and hands over.
HANDOVER_AFTER = 10
# Same limit in 'full' mode, where the bot keeps playing on its own.
HANDOVER_AFTER_FULL = 40
FIRST_MOVES_EXPANDED = 25
# Drags looked ahead per decision (the drag itself included) and the replies searched at each
# deeper level.
SEARCH_DEPTH = 3
# Pause after a drag that clears or empties a shelf before checking that the screen is still;
# the logged 'reveal wait' times show whether it matches the length of the animation.
REVEAL_SETTLE = 0.8
# Pause between a drag that clears nothing and the next capture: long enough for the dropped
# product to settle in its slot.
PAUSE_AFTER_DRAG = 0.2
# Whether the bot keeps playing other shelves while a fixed shelf runs its clear animation,
# instead of waiting for the whole screen to settle (about 2 s per clear, close to half of
# the level clock on level 26).
PLAY_THROUGH_REVEALS = True
# Number of empty, exhausted fixed shelves from which the bot stops searching ahead and
# assembles triples one at a time (endgame_move). Zero disables it.
ENDGAME_EMPTY_SHELVES = 2
# Level 26 budget: 7 min 40 s and 290 drags. time_to_spare compares the two.
LEVEL_SECONDS = 460
LEVEL_DRAGS = 290
# The bot's pace in drags, measured on level 26 with play-through reveals.
SECONDS_PER_DRAG = 1.3
# Seconds kept in reserve before the bot stops waiting for the belt.
SPARE_MARGIN = 10
# Longest wait for the belt since the last endgame drag; past it the bot takes what it has.
MAX_BELT_WAIT = 4.0
# Whether play_through also covers clears on belt shelves, which are then tracked by position
# rather than by index (see belt_animating in play). Off: on level 26 it saved time but the
# unverified belt clears cost moves (76% in 268 drags against 72% in 212 with it off).
PLAY_THROUGH_BELT = False
# Seconds a fixed shelf is left alone after it clears or empties.
REVEAL_ANIMATION = 2.2
# Score taken off a drag that parks a product on a belt shelf without clearing it. Zero
# (disabled): on level 26 a penalty of 25 together with BELT_PATIENCE reached 36% against 62%
# without them.
BELT_DROP_PENALTY = 0
# Whether position_value ignores belt shelves; see the note in position_value.
SCORE_FIXED_ONLY = False
# On belt levels, when the best drag clears nothing and improves the position by less than
# this, the bot reads the screen again (up to BELT_PATIENCE times) in case the belt brings a
# better one: drags are the scarcer budget on those levels.
WAIT_IF_GAIN_BELOW = 15
# Zero (disabled): waiting for the belt saved drags but ran the level out of time.
BELT_PATIENCE = 0
# Fewer shelves than this on screen means the level is over: levels have two or three fixed
# rows of three shelves, plus belt shelves on some.
MIN_SHELVES = 6
# Number of fixed (non-belt) shelves on the current level, updated on every read: nine on
# levels with three fixed rows, six on levels such as 22 with two.
FIXED_SHELVES = 9
# Some levels (22) have a basket under the shelves holding spare products. Dragging the basket
# onto a completely empty fixed shelf fills it; the basket can be used BASKET_USES times.
BASKET_POINT = (540, 1836)
BASKET_RING_POINT = (540, 1700)
BASKET_USES = 7


def basket_on_screen(img):
    """True when the red basket inside its yellow ring is drawn at the bottom of the screen."""
    px = np.asarray(img).astype(int)
    r, g, b = px[BASKET_POINT[1], BASKET_POINT[0]]
    ring = px[BASKET_RING_POINT[1], BASKET_RING_POINT[0]]
    red = r > 200 and g < 60 and b < 60
    yellow = ring[0] > 200 and 150 < ring[1] < 220 and ring[2] < 80
    return red and yellow
# Whether the level has a conveyor belt of incoming products; level 21 has none.
LEVEL_HAS_BELT = False
# Seconds spent per phase, filled and printed only in "feed" mode.
TIMINGS = {}
SECOND_MOVES_EXPANDED = 12
# When set, drags whose outcome is predictable are played without a fresh screenshot.
PREDICT_BETWEEN_READS = False


def apply_move(board, move, exhausted=frozenset(), hidden=None):
    """Returns (new_board, gain) after one drag, following the game rules.

    `exhausted` holds shelves seen empty with nothing left behind them: emptying or clearing
    such a shelf reveals nothing, so it stays free and earns no reveal bonus. `hidden` holds
    the guessed products behind each shelf; a shelf that clears or empties is refilled with
    them, so plans can favour reveals that bring forward the third copy of a pair. Slots
    without a guess are filled with UNKNOWN, a product that pairs with nothing.
    """
    def revealed(shelf):
        return list(hidden[shelf]) if hidden else [UNKNOWN] * 3

    s, k, d, dk = move
    label = board[s][k]
    after = [list(shelf) for shelf in board]
    after[s][k], after[d][dk] = None, label
    gain = 0
    if after[d].count(label) == 3:
        after[d] = [None] * 3 if d in exhausted else revealed(d)
        gain += CLEAR_GAIN
        gain += FIXED_CLEAR_BONUS if d < FIXED_SHELVES else BELT_CLEAR_BONUS
    if all(v is None for v in after[s]) and s not in exhausted:
        after[s] = revealed(s)
        gain += REVEAL_GAIN
    return after, gain


def legal_moves(board):
    for s, shelf in enumerate(board):
        for k, label in enumerate(shelf):
            if label is None or label == UNKNOWN:
                continue
            for d, target in enumerate(board):
                # Shelves past the nine fixed ones are on the belt; in 'belt-feed' play they
                # are only ever a source of products, never a place to drop one.
                if BELT_FEED and d >= 9:
                    continue
                if FIXED_TARGETS_ONLY and d >= FIXED_SHELVES:
                    continue
                if TOP_ROW_TARGETS_ONLY and d >= 3:
                    continue
                if d != s and None in target:
                    yield s, k, d, target.index(None)


def endgame_move(board, rows, exhausted, last_drags=()):
    """First drag of the cheapest triple to assemble, or None when no triple can be assembled
    from the products on screen.

    Used once ENDGAME_EMPTY_SHELVES fixed shelves are empty with nothing left behind them. The
    lookahead planner then sees many near-equal drags and wastes the move budget shuffling;
    here every drag brings a product to the shelf where its triple is being built. A
    destination must hold nothing but copies of that product, and the plan costs one drag
    per missing copy. Sources on the belt come first (they scroll away otherwise), then the
    fixed shelf with the fewest products, which brings its hidden products forward soonest.
    """
    places = {}
    for shelf, slots in enumerate(board):
        for slot, label in enumerate(slots):
            if label is not None and label != UNKNOWN:
                places.setdefault(label, []).append((shelf, slot))
    # Products the last drags put down stay where they are unless they complete a triple;
    # moving them again is how the bot ended up carrying one product back and forth.
    placed = {(m[2], m[3]) for m in last_drags}

    def pure(shelf, label):
        return all(v in (None, label) for v in board[shelf])

    candidates = []
    for d, geometry in enumerate(rows):
        if None not in board[d]:
            continue
        held = {v for v in board[d] if v is not None}
        # Once the fixed shelves run dry most products sit on the belt, often as a pair with
        # a free slot: completing the triple there saves carrying all three copies to a fixed
        # shelf. An empty belt shelf is not a starting point, since it scrolls off screen.
        if geometry["belt"] and not held:
            continue
        if len(held) > 1 or UNKNOWN in held:
            continue
        for label in held or places:
            have = board[d].count(label)
            need = 3 - have
            # Build on the shelf already holding the most copies; starting a triple
            # beside a shelf that holds as many only moves the copies from one to the other.
            if any(pure(o, label) and board[o].count(label) > have
                   or pure(o, label) and board[o].count(label) == have and have and o < d
                   for o in range(len(board)) if o != d):
                continue
            sources = [p for p in places.get(label, []) if p[0] != d
                       and (p not in placed or need == 1)]
            if len(sources) < need:
                continue
            # Ties go to a belt shelf, whose products otherwise scroll past again and again,
            # then to a shelf that still hides products, so the triple also reveals some.
            cost = (need, not geometry["belt"], d in exhausted)

            def source_rank(p):
                return (not rows[p[0]]["belt"], sum(v is not None for v in board[p[0]]))
            s, k = min(sources, key=source_rank)
            candidates.append((cost, (s, k, d, board[d].index(None))))
    # Among plans of equal cost, the first drag that leaves the most one-drag triples behind
    # goes first: with few products left, one drag often sets up two triples.
    best = None
    if candidates:
        cheapest = min(c[0][0] for c in candidates)
        ranked = [(cost[0], -one_drag_triples(drag_result(board, move)), cost[1:], move)
                  for cost, move in candidates if cost[0] == cheapest]
        need, _, rest, move = min(ranked, key=lambda r: r[:3])
        best = ((need,) + rest, move)
    # A full shelf holding a pair and one odd product is two drags from a triple: move the odd
    # one out, then bring the third copy. That beats building on an empty shelf (three drags).
    if best is None or best[0][0] > 2:
        free = [(o, board[o].index(None)) for o in range(len(board)) if None in board[o]]
        for d, slots in enumerate(board):
            if None in slots or UNKNOWN in slots:
                continue
            for label in set(slots):
                if slots.count(label) != 2 or not any(p[0] != d for p in places.get(label, [])):
                    continue
                k = next(i for i, v in enumerate(slots) if v != label)
                odd = slots[k]
                targets = [t for t in free if t[0] != d]
                if not targets:
                    continue
                # Park the odd product where it pairs, or failing that on a fixed shelf.
                o, slot = max(targets, key=lambda t: (board[t[0]].count(odd), not rows[t[0]]["belt"]))
                cost = (2, not rows[d]["belt"], d in exhausted)
                if best is None or cost < best[0]:
                    best = (cost, (d, k, o, slot))
    global ENDGAME_LAST_COST
    ENDGAME_LAST_COST = best[0][0] if best else None
    return best[1] if best else None


def drag_result(board, move):
    """Board after a drag, without the products a clear or an emptied shelf would reveal."""
    s, k, d, dk = move
    after = [list(shelf) for shelf in board]
    after[s][k], after[d][dk] = None, board[s][k]
    return after


def one_drag_triples(board):
    """Shelves holding a pair, a free slot and nothing else whose third copy is on screen."""
    count = 0
    for d, slots in enumerate(board):
        held = {v for v in slots if v is not None}
        if None not in slots or len(held) != 1 or UNKNOWN in held:
            continue
        label = held.pop()
        if slots.count(label) == 2 and any(label in other for o, other in enumerate(board) if o != d):
            count += 1
    return count


# Set once play() enters the endgame (see ENDGAME_EMPTY_SHELVES).
ENDGAME_ACTIVE = False

# Drags the cheapest endgame plan needed at the last endgame_move call (None: no plan).
ENDGAME_LAST_COST = None


def time_to_spare(drags):
    """True while the level clock holds more than the remaining drags need at the bot's pace.

    Assumes the bot started with the level (a bot restarted mid-level underestimates its
    drags and so waits too readily). Waiting for the belt then costs nothing that matters,
    while every drag saved is one more chance at a triple before the move budget runs out.
    """
    # When a person takes over at STOP_AT, the time left is theirs, and they play slower.
    # Before WAIT_FROM_PROGRESS the board is crowded enough that a drag nearly always helps.
    if STOP_AT is not None or LAST_PROGRESS < WAIT_FROM_PROGRESS:
        return False
    left_seconds = LEVEL_SECONDS - (time.time() - PLAY_STARTED)
    left_drags = LEVEL_DRAGS - drags
    return left_seconds > left_drags * SECONDS_PER_DRAG + SPARE_MARGIN


def make_room(board, rows):
    """Last-resort drag when the planner finds nothing useful: moves a product from a full
    fixed shelf down to a free slot on a belt shelf, so the fixed shelves get room to play.
    Prefers the odd product of a shelf (the one without a copy beside it) and a belt shelf
    that already holds that product. Returns None when no belt shelf has a free slot."""
    belt = [d for d, g in enumerate(rows) if g["belt"] and None in board[d]]
    fixed = [s for s, g in enumerate(rows) if not g["belt"] and None not in board[s]]
    best = None
    for s in fixed:
        for k, label in enumerate(board[s]):
            if label is None or label == UNKNOWN:
                continue
            for d in belt:
                score = (board[s].count(label) == 1, board[d].count(label))
                if best is None or score > best[0]:
                    best = (score, (s, k, d, board[d].index(None)))
    return best[1] if best else None


def position_value(board):
    """Heuristic worth of a board: pairs that can become triples, and nearly empty shelves."""
    counts = {}
    for shelf in board:
        for v in shelf:
            if v is not None and v != UNKNOWN:
                counts[v] = counts.get(v, 0) + 1
    value = 0
    # Scoring only the fixed shelves (belt shelves leave the screen within seconds) looked
    # right but did worse on level 26 (46% against 62%), so it is off by default.
    fixed = board[:FIXED_SHELVES] if SCORE_FIXED_ONLY else board
    for shelf in fixed:
        known = [v for v in shelf if v is not None and v != UNKNOWN]
        free = shelf.count(None)
        for label in set(known):
            if known.count(label) == 2:
                completable = counts[label] >= 3
                if free:
                    value += 30 if completable else 8
                else:
                    value += 12 if completable else 3
        if free == 2 and len(known) == 1:
            value += 10  # one more drag empties it and reveals the products behind
        if free == 3:
            value += 6   # an empty shelf is room to park products while unblocking others
    # With fewer than two free slots on the whole board most drags become impossible and
    # the bot can lock itself out, so filling the last free slots is heavily penalised.
    total_free = sum(shelf.count(None) for shelf in fixed)
    if total_free < 2:
        value -= 60 * (2 - total_free)
    return value


def belt_drop_penalty(board, move, gain):
    """Cost of putting a product on a belt shelf without clearing it: the product leaves the
    screen and has to be fetched again later, which costs drags."""
    on_belt = len(board) > FIXED_SHELVES and move[2] >= FIXED_SHELVES
    return BELT_DROP_PENALTY if on_belt and gain < CLEAR_GAIN else 0


def pointless(board, move, exhausted):
    """True for a drag that only relocates a product: from a shelf holding nothing else, with
    nothing hidden behind it, to an empty shelf. The board is no closer to a triple and a drag
    is gone; waiting for the belt or using a help is better."""
    s, k, d, _ = move
    label = board[s][k]
    return (all(v is None for v in board[d])
            and all(v in (None, label) for v in board[s])
            and s in exhausted)


def choose_move(board, history, exhausted=frozenset(), hidden=None):
    """Picks the next drag, or None.

    `history` holds the last drags as (src_shelf, src_slot, dst_shelf, dst_slot). Two kinds of
    drag are refused unless they complete a triple, because they shuffle products back and
    forth instead of uncovering new ones: moving a product that one of the last four drags
    placed, and any drag between the same two shelves as one of the last two drags (in either
    direction).
    """
    baseline = position_value(board)
    recent = {(m[2], m[3]) for m in history[-4:]}
    recent_pairs = {frozenset((m[0], m[2])) for m in history[-2:]}
    # Shelf pairs (from, to) of the last three drags: sending a product back to the shelf a
    # recent drag took it from undoes that drag.
    backwards = {(m[2], m[0]) for m in history[-3:]}

    def churn(move, gain):
        return gain < CLEAR_GAIN and ((move[0], move[1]) in recent
                                       or frozenset((move[0], move[2])) in recent_pairs
                                       or (move[0], move[2]) in backwards)

    best, best_move = None, None
    first = []
    # Belt-feed rule: with BELT_FEED_ABOVE or more free slots on the fixed shelves the bot
    # refills them from the belt, keeping two free slots to manoeuvre with; otherwise it only
    # takes from the belt to complete a triple. Once the belt is empty only fixed moves remain.
    free_fixed = sum(shelf.count(None) for shelf in board[:9])
    belt_moves = BELT_FEED and any(m[0] >= 9 for m in legal_moves(board))
    # Fixed-shelf moves only give way to the belt when a pull would actually pair up.
    useful_pull = belt_moves and any(
        m[0] >= 9 and board[m[2]].count(board[m[0]][m[1]]) > 0 for m in legal_moves(board))
    for move in legal_moves(board):
        after, gain = apply_move(board, move, exhausted, hidden)
        if churn(move, gain) or pointless(board, move, exhausted):
            continue
        # In the endgame a product left on a belt shelf only scrolls away and has to be
        # fetched again: of the drags that did so on level 26, almost none led to a triple.
        if ENDGAME_ACTIVE and move[2] >= FIXED_SHELVES and gain < CLEAR_GAIN:
            continue
        if belt_moves:
            from_belt = move[0] >= 9
            if from_belt:
                # Only pull a product that already has a copy on the destination shelf: it
                # forms a pair or completes a triple there. Anything else just fills a free
                # slot with a product that later costs extra drags to move again.
                label = board[move[0]][move[1]]
                if board[move[2]].count(label) == 0:
                    continue
                if free_fixed < BELT_FEED_ABOVE and gain < CLEAR_GAIN:
                    continue
            elif free_fixed >= BELT_FEED_ABOVE and useful_pull:
                continue
        first.append((gain + position_value(after) - belt_drop_penalty(board, move, gain), after, gain, move))
    first.sort(key=lambda c: c[0], reverse=True)
    def best_continuation(state, depth):
        """Best score reachable from state in `depth` more drags (heuristic value at the end).

        Only the SECOND_MOVES_EXPANDED most promising replies are searched at the deeper
        levels, which keeps a three-drag search well under a second per decision.
        """
        value = position_value(state)
        if depth == 0:
            return value
        replies = []
        for reply in legal_moves(state):
            nxt, g = apply_move(state, reply, exhausted, hidden)
            replies.append((g + position_value(nxt), nxt, g))
        replies.sort(key=lambda r: r[0], reverse=True)
        best_value = value
        for _, nxt, g in replies[:SECOND_MOVES_EXPANDED]:
            best_value = max(best_value, g + SECOND_MOVE_WEIGHT * best_continuation(nxt, depth - 1))
        return best_value

    for _, after, gain, move in first[:FIRST_MOVES_EXPANDED]:
        total = gain + SECOND_MOVE_WEIGHT * best_continuation(after, SEARCH_DEPTH - 1) \
            - belt_drop_penalty(board, move, gain)
        if best is None or total > best:
            best, best_move = total, move

    # Two drags cannot reach any improvement: nothing visible can be matched soon, so dig.
    # Empty the shelf with the fewest known products, sending each one where it pairs if
    # possible, so the hidden products behind it come forward.
    if best is None or best <= baseline + 5:
        def known(shelf):
            return [v for v in shelf if v is not None and v != UNKNOWN]
        shelves = sorted((a for a in range(len(board))
                          if known(board[a]) and UNKNOWN not in board[a] and a not in exhausted),
                         key=lambda a: len(known(board[a])))
        for a in shelves:
            options = []
            for k, label in enumerate(board[a]):
                if label is None or (a, k) in recent:
                    continue
                for d, target in enumerate(board):
                    if d == a or None not in target or churn((a, k, d, 0), 0):
                        continue
                    pairs = target.count(label)
                    fill = 3 - target.count(None)
                    # Prefer a shelf that already holds this product, then the fullest one,
                    # so free room stays together on as few shelves as possible.
                    options.append(((pairs, fill), (a, k, d, target.index(None))))
            if options:
                return max(options)[1]
    return best_move


# Seconds between issuing an adb swipe and the finger touching down on the phone.
ADB_LATENCY = 0.25


def drag(rows, move):
    """Drags one product. A belt slot is aimed at where the belt will be when the finger lands
    (source) or lifts (destination)."""
    s, k, d, dk = move
    duration = random.randint(140, 180) / 1000
    touch = time.time() + ADB_LATENCY
    x0, y0 = slot_point(rows, s, k, at=touch)
    x1, y1 = slot_point(rows, d, dk, at=touch + duration)
    adb("shell", "input", "swipe", str(x0), str(y0), str(x1), str(y1), str(int(duration * 1000)))


def debug(source=None, path="debug.png"):
    """Reads one board (a saved screenshot, or a fresh capture) and draws what was recognised:
    the front label above each slot in red and the guessed hidden label in blue."""
    img = Image.open(source).convert("RGB") if source else capture()
    if not source:
        img.save("last.png")
    rows, board, hidden = read_board(img, with_hidden=True)
    print("shelf rows:", rows)
    for i, (shelf, behind) in enumerate(zip(board, hidden)):
        print(i + 1, "front", shelf, "behind", ["?" if v == UNKNOWN else v for v in behind])
    print("next move:", choose_move(board, [], frozenset(), hidden))
    draw = ImageDraw.Draw(img)
    for s, shelf in enumerate(board):
        for k, label in enumerate(shelf):
            x, y = slot_point(rows, s, k)
            draw.text((x - 10, y - 60), "-" if label is None else str(label), fill=(255, 0, 0))
            guess = hidden[s][k]
            draw.text((x - 10, y - 200), "?" if guess == UNKNOWN else str(guess), fill=(0, 0, 255))
    img.save(path)


# Row of the level progress bar and the x range it spans on a 1080 px wide screen. The red
# share of that range is the level completion that the game prints next to the bar.
PROGRESS_Y, PROGRESS_X0, PROGRESS_X1 = 325, 211, 918
# Completion at which the bot stops and hands the rest of the level over to manual play.
HANDOVER_PROGRESS = 0.50
# Set by the 'full' mode: play the whole level instead of handing over at HANDOVER_PROGRESS.
KEEP_PLAYING = False
# Optional level progress (0-1) at which 'full' mode stops so a person can finish the level,
# given as a percentage after the mode: `bot.py full 90`.
STOP_AT = None


# Level progress (0 to 1) at the last should_hand_over check.
LAST_PROGRESS = 0.0


def should_hand_over(img):
    global LAST_PROGRESS
    level = progress(img)
    LAST_PROGRESS = level
    return (not KEEP_PLAYING and level >= HANDOVER_PROGRESS) or (STOP_AT is not None and level >= STOP_AT)
# Fixed shelves (0-based, row by row) left to a person playing alongside the bot; set by the
# 'coop' mode to the right-hand column. The belt stays shared.
RESERVED_SHELVES = frozenset()
# Set by the 'top' mode: the person plays the belt, so the bot ignores belt shelves entirely.
SKIP_BELT = False
# Set by the 'half' mode: a person plays the lower half of the board (the bottom fixed row
# and the belt above it) at the same time, so the bot neither reads nor touches those shelves.
TOP_HALF_ONLY = False


def top_half(rows):
    """Shelves above the middle of the board, the middle being halfway between the tops of
    the first and last fixed rows (on level 26 this keeps the top fixed row and the top belt)."""
    tops = [g["top"] for g in rows if not g["belt"]]
    if not tops:
        return rows
    middle = (min(tops) + max(tops)) / 2
    return [g for g in rows if g["top"] < middle]


# Set by the 'half' mode. The move counter is shared with the person playing alongside, so
# the bot only spends a drag when it clears a triple, empties a shelf, or sets up a triple
# that the very next drag can clear; otherwise it waits for the belt or the person to change
# the board.
STRICT_CLOSING = False
# Set by the 'half' mode: products are only ever dropped on fixed shelves. Belt shelves are a
# source of products, so every triple the bot completes is on a fixed shelf.
FIXED_TARGETS_ONLY = False
# Set by the 'cima' mode: products are only dropped on the top row of fixed shelves (the
# first three shelves); they may be taken from any shelf, belts and the bottom row included.
TOP_ROW_TARGETS_ONLY = False
# Set by the 'fixas' mode: the only drags allowed are those that complete a triple at once.
CLEARS_ONLY = False
# Seconds without a clear or emptied shelf after which one ordinary drag is allowed anyway.
STRICT_MAX_WAIT = 20


def closes_soon(board, move, exhausted, hidden):
    """True when the drag clears or empties a shelf, or leaves a triple one drag away."""
    after, gain = apply_move(board, move, exhausted, hidden)
    if CLEARS_ONLY:
        return gain >= CLEAR_GAIN
    if gain >= CLEAR_GAIN or (gain >= REVEAL_GAIN and move[0] < FIXED_SHELVES):
        return True
    return any(apply_move(after, reply, exhausted, hidden)[1] >= CLEAR_GAIN
               for reply in legal_moves(after))


def closing_move(board, exhausted, hidden):
    """Best drag that passes closes_soon: immediate gains first, then triple set-ups."""
    best = None
    for move in legal_moves(board):
        if not closes_soon(board, move, exhausted, hidden):
            continue
        gain = apply_move(board, move, exhausted, hidden)[1]
        if best is None or gain > best[0]:
            best = (gain, move)
    return best[1] if best else None


# Set by the 'feed' mode: the belt is only a source, used to refill the fixed shelves whenever
# they have BELT_FEED_ABOVE or more free slots.
BELT_FEED = False
BELT_FEED_ABOVE = 3


def progress(img):
    row = np.asarray(img)[PROGRESS_Y, PROGRESS_X0:PROGRESS_X1 + 1].astype(int)
    red = (row[:, 0] > 180) & (row[:, 1] < 90) & (row[:, 2] < 90)
    return red.mean()


def board_signature(img):
    """Coarse greyscale thumbnail of the shelf area, used to tell when animations have ended."""
    # Only the fixed shelves are compared, as a belt never stops moving; in 'coop' mode only
    # the two left columns, as the person's moves on the right would look like animations.
    right = 700 if RESERVED_SHELVES else 1080
    # Only the bands of the fixed rows are compared, wherever the level places them: belts can
    # sit under the fixed rows (level 22) or between them (level 26).
    rows = find_shelf_rows(np.asarray(img)) or [(560, 1640)]
    bands = [np.asarray(img.crop((0, top, right, bottom)).convert("L").resize((36, 18)), dtype=float)
             for top, bottom in rows]
    return np.concatenate(bands)


def wait_until_still(timeout=3.0):
    """Captures until two consecutive frames match, so the board is read after the animations."""
    previous = board_signature(capture())
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(0.2)
        img = capture()
        current = board_signature(img)
        # Frames that found a different number of fixed rows are still mid-animation.
        if current.shape == previous.shape and np.abs(current - previous).mean() < 1.0:
            return img
        previous = current
    return capture()


def show(move):
    s, k, d, dk = move
    return f"shelf {s + 1} slot {k + 1} -> shelf {d + 1} slot {dk + 1}"


def outcome(before, after, move):
    """Classifies what one drag did, comparing the boards read before and after it."""
    s, k, d, dk = move
    # Belt shelves are renumbered on every read, so a drag touching one cannot be checked.
    if s >= FIXED_SHELVES or d >= FIXED_SHELVES or d >= len(after):
        return "ok"
    if before == after:
        return "nothing changed"
    # A cleared shelf is entirely free afterwards, so a free destination is only a failure
    # when the rest of that shelf still holds products.
    if after[d][dk] is None and any(v is not None for v in after[d]):
        return "destination still free"
    return "ok"


def measure_belt(attempts=6):
    """Fills BELT_SPEEDS with the speed of every belt row, from pairs of screenshots.

    A row can be missed on one pair (no shelf of it fully on screen in both frames), and an
    unmeasured row falls back to BELT_SPEED, which on level 26 is the other belt's speed in
    the opposite direction: every drag to or from that row then misses. So pairs are taken
    until every belt row seen has its own speed.
    """
    global BELT_SPEED
    seen = set()
    for _ in range(attempts):
        first = capture()
        time.sleep(0.4)
        second = capture()
        rows1 = [g for g in shelf_geometry(np.asarray(first), first.info["t"]) if g["belt"]]
        rows2 = [g for g in shelf_geometry(np.asarray(second), second.info["t"]) if g["belt"]]
        seen |= {g["top"] for g in rows1} | {g["top"] for g in rows2}
        dt = second.info["t"] - first.info["t"]
        for top in sorted({g["top"] for g in rows1} - set(BELT_SPEEDS)):
            shifts = [b["xs"][0] - a["xs"][0] for a in rows1 if a["top"] == top
                      for b in rows2 if b["top"] == top]
            if not shifts or dt <= 0:
                continue
            # Shelves are 315 px apart, so the true shift is the smallest one in magnitude.
            shift = min(shifts, key=abs)
            if abs(shift) < 300:
                BELT_SPEEDS[top] = shift / dt
                BELT_SPEED = shift / dt
                print(f"belt at y={top}: {BELT_SPEEDS[top]:.0f} px/s")
        if not seen or seen <= set(BELT_SPEEDS):
            return
    missing = sorted(seen - set(BELT_SPEEDS))
    if missing:
        print(f"belt rows without a measured speed: {missing}")


# Help buttons under the board: x of each button centre, at HELP_Y. A button is red while
# its help is unused and turns beige once used; HELP_RIM_OFFSET samples the red rim, since
# the centre holds a white icon.
HELP_BUTTONS = {"hammer": 263, "freeze": 539, "shuffle": 813}
HELP_Y = 2160
HELP_RIM_OFFSET = 60
USE_HELPS = True
USE_FREEZE = True
# ARRUMA_HAMMER=0 leaves the hammer to the person who finishes the level.
USE_HAMMER = os.environ.get("ARRUMA_HAMMER", "1") != "0"
# Drags in a row without a clear or reveal after which the hammer, then the shuffle, is used.
HAMMER_AFTER_STALE = 6
SHUFFLE_AFTER_STALE = 10
# Seconds into the level before the bot may use the hammer: the full opening board often has
# no free slot for a moment, which only looks like being stuck.
HELPS_NOT_BEFORE = 90
# The game's shuffle groups identical products: on a level 26 board it turned one shelf with
# a pair and no one-drag triple into five of each. It pays most on a crowded board, so it is
# used once the fixed shelves hold this many products and no drag completes a triple.
SHUFFLE_WHEN_PRODUCTS = 12


def crowded_without_triple(board):
    products = sum(v is not None for shelf in board[:FIXED_SHELVES] for v in shelf)
    return products >= SHUFFLE_WHEN_PRODUCTS and not any(
        apply_move(board, m)[1] >= CLEAR_GAIN for m in legal_moves(board))


# Seconds for the hammer or shuffle animation to finish before the board is read again.
HELP_ANIMATION = 2.8


def help_available(img, name):
    r, g, b = img.getpixel((HELP_BUTTONS[name] - HELP_RIM_OFFSET, HELP_Y))[:3]
    return r > 180 and g < 60 and b < 60


def use_help(name):
    adb("shell", "input", "tap", str(HELP_BUTTONS[name]), str(HELP_Y))
    print(f"{time.time() - PLAY_STARTED:6.1f}s help used: {name}")
    time.sleep(HELP_ANIMATION)


# Belt shelves are this far apart, centre to centre.
BELT_SPACING = 315
# Progress from which the bot waits for a remembered belt shelf, and the longest such wait.
MEMORY_FROM_PROGRESS = 0.50
MEMORY_WAIT_MAX = 3.0
# Progress from which the bot waits blindly for the belt when it has time to spare.
WAIT_FROM_PROGRESS = 0.70
# A belt shelf counts as reachable once its centre is inside this band of the screen.
SCREEN_BAND = (150, 930)


class BeltMemory:
    """Remembers what each belt shelf carries after it leaves the screen.

    A shelf keeps the same offset x - speed * t while it moves, so that offset divided by the
    shelf spacing is a stable number for it. The belt is a loop: when a shelf seen earlier
    comes back with the same products, the difference between the two numbers is the number
    of shelves in the loop, and from then on every remembered shelf's next appearance can be
    predicted. Two sightings must agree on that number before it is trusted, since two
    different shelves may carry the same products.
    """

    def __init__(self, t0):
        self.t0 = t0
        self.shelves = {}  # (row top, shelf number) -> (labels, time last seen)
        self.period = {}   # row top -> shelves in the loop, once known
        self.votes = {}    # (row top, candidate period) -> sightings

    def number(self, g):
        offset = g["xs"][1] - g.get("speed", BELT_SPEED) * (g["t"] - self.t0)
        return round(offset / BELT_SPACING)

    def key(self, top, n):
        period = self.period.get(top)
        return (top, n % period if period else n)

    def update(self, rows, board):
        now = time.time()
        for g, slots in zip(rows, board):
            if not g["belt"]:
                continue
            top, n, labels = g["top"], self.number(g), tuple(slots)
            known = [v for v in labels if v not in (None, UNKNOWN)]
            if top not in self.period and len(known) >= 2:
                for (other_top, m), (old, seen) in list(self.shelves.items()):
                    if other_top == top and m != n and old == labels and now - seen > 3:
                        candidate = abs(n - m)
                        if candidate < 4:
                            continue
                        votes = self.votes.get((top, candidate), 0) + 1
                        self.votes[(top, candidate)] = votes
                        if votes >= 2:
                            self.period[top] = candidate
                            print(f"{now - PLAY_STARTED:6.1f}s belt at y={top}: {candidate} shelves per loop")
                            merged = {}
                            for (t2, m2), value in self.shelves.items():
                                k = self.key(t2, m2)
                                if k not in merged or value[1] > merged[k][1]:
                                    merged[k] = value
                            self.shelves = merged
                        break
            self.shelves[self.key(top, n)] = (labels, now)

    def arrival(self, label):
        """Seconds until a remembered belt shelf holding `label` reaches the screen band, or
        None when no such shelf is known on a belt whose loop length is known."""
        now = time.time()
        best = None
        for (top, c), (labels, _) in self.shelves.items():
            period = self.period.get(top)
            speed = BELT_SPEEDS.get(top)
            if label not in labels or not period or not speed:
                continue
            loop = period * BELT_SPACING
            x = (c * BELT_SPACING + speed * (now - self.t0)) % loop
            if SCREEN_BAND[0] <= x <= SCREEN_BAND[1]:
                distance = 0.0
            elif speed > 0:
                distance = (SCREEN_BAND[0] - x) % loop
            else:
                distance = (x - SCREEN_BAND[1]) % loop
            eta = distance / abs(speed)
            best = eta if best is None else min(best, eta)
        return best


def waiting_pairs(board):
    """Products with a shelf holding a pair of them, a free slot and nothing else, and no third
    copy on screen: the shelves the belt memory can complete."""
    labels = set()
    for d, slots in enumerate(board):
        held = {v for v in slots if v is not None}
        if None not in slots or len(held) != 1 or UNKNOWN in held:
            continue
        label = next(iter(held))
        if slots.count(label) == 2 and not any(label in other for o, other in enumerate(board) if o != d):
            labels.add(label)
    return labels


# Start of the current play() call; move log lines show seconds since then.
PLAY_STARTED = time.time()


def play(step_mode=False):
    """Plays until the level ends. In step mode it makes at most five drags, saving the board
    before and after each one in shots/ and printing how the drag was interpreted."""
    global FIXED_SHELVES
    recent, failures, drags, exhausted, stale = [], 0, 0, set(), 0
    basket_left, basket_misses = BASKET_USES, 0
    # Every drag made, never pruned: `recent` loses entries when a drag is expected to
    # reveal products, so the loop guard needs its own record.
    trail = []
    patience = BELT_PATIENCE
    rows, board = None, None
    idle_waits = 0
    last_progress = time.time()
    # Fixed shelf index -> time at which its clear animation ends ('play through' reveals).
    animating = {}
    memory = BeltMemory(time.time())
    # (row top, belt shelf number, slot, time) of the last product dropped on a belt shelf.
    placed_on_belt = None
    last_drag_at = time.time()
    # Help buttons already pressed (or found used) in this level.
    helps_used = set()
    # Belt shelves running a clear animation: (row top, centre x, capture time, speed, end).
    belt_animating = []
    endgame_logged = False
    # Time of the last endgame drag; the bot waits for the belt for up to MAX_BELT_WAIT seconds
    # after it before falling back to the lookahead planner.
    endgame_since = time.time()
    # Set once any belt shelf has been seen: belt shelves are often only partly on screen,
    # and the bot must keep waiting for them instead of reading that as "no move".
    belt_seen = False
    measure_belt()
    while True:
        if board is None:
            if not app_in_focus():
                print("Continente app is not in the foreground; stopping.")
                return
            img = wait_until_still()
            for _ in range(10):
                # A dialog, a notification or a finger mid-drag can hide the shelves for a
                # moment; only a lasting absence (about five seconds) means there is no level.
                if len(find_shelf_rows(np.asarray(img))) >= 2:
                    break
                time.sleep(0.5)
                img = capture()
            if should_hand_over(img):
                img.save("shots/handover.png")
                print("HANDOVER: level at %d%%; screenshot saved to shots/handover.png" % round(progress(img) * 100))
                return
            rows, board, hidden = read_board(img, with_hidden=True)
            # A frame taken mid-animation can miss shelves; only a few misses in a row mean
            # the level is over.
            for _ in range(6):
                if len(rows) >= MIN_SHELVES:
                    break
                time.sleep(0.5)
                img = capture()
                rows, board, hidden = read_board(img, with_hidden=True)
            if len(rows) < MIN_SHELVES:
                print("No shelves on screen (level over?); stopping.")
                return
            if step_mode:
                img.save(f"shots/step{drags}_before.png")
                print("board before:", board)
        empty_fixed = [i for i, g in enumerate(rows)
                       if not g["belt"] and all(v is None for v in board[i])]
        if empty_fixed and basket_left > 0 and img is not None and not isinstance(img, tuple) \
                and basket_on_screen(img):
            target = empty_fixed[0]
            x1, y1 = slot_point(rows, target, 1)
            adb("shell", "input", "swipe", str(BASKET_POINT[0]), str(BASKET_POINT[1]),
                str(x1), str(y1), str(random.randint(230, 330)))
            # The refilled products slide in; reading earlier sees the shelf still empty.
            time.sleep(1.2)
            img = wait_until_still()
            rows, board, hidden = read_board(img, with_hidden=True)
            if target < len(board) and any(v is not None for v in board[target]):
                basket_left -= 1
                basket_misses = 0
                exhausted.discard(target)
                print(f"{time.time() - PLAY_STARTED:6.1f}s basket -> shelf {target + 1}: ok ({basket_left} left)")
            else:
                # Two drags with no effect mean the basket is already empty.
                basket_misses += 1
                print(f"{time.time() - PLAY_STARTED:6.1f}s basket -> shelf {target + 1}: no effect")
                if basket_misses >= 2:
                    basket_left = 0
            continue
        if RESERVED_SHELVES:
            # Shelves a person is playing at the same time: shown to the planner as full of
            # unidentified products, so they are never a source or a destination.
            board = [[UNKNOWN] * 3 if i in RESERVED_SHELVES else shelf for i, shelf in enumerate(board)]
        now = time.time()
        memory.update(rows, board)
        busy = {x for x, until in animating.items() if until > now and x < len(board)}
        belt_animating = [a for a in belt_animating if a[4] > now]
        for i, g in enumerate(rows):
            if g["belt"] and i < len(board) and any(
                    top == g["top"] and abs(x0 + speed * (g["t"] - t0) - g["xs"][1]) < 100
                    for top, x0, t0, speed, _ in belt_animating):
                busy.add(i)
        if busy:
            board = [[UNKNOWN] * 3 if i in busy else shelf for i, shelf in enumerate(board)]
        move = None
        # Which rule chose the drag, for the move log: endgame plan, lookahead search or
        # make_room.
        source = "search"
        # Once on, endgame play stays on: assembling a triple fills the empty shelves that
        # triggered it.
        if ENDGAME_EMPTY_SHELVES and endgame_logged or ENDGAME_EMPTY_SHELVES and sum(i in exhausted and all(v is None for v in board[i])
                                         for i in range(FIXED_SHELVES)) >= ENDGAME_EMPTY_SHELVES:
            move = endgame_move(board, rows, exhausted, trail[-2:])
            # With time to spare the bot also waits rather than take a plan of three drags,
            # since the belt often brings a pair that needs only one.
            costly = move is not None and ENDGAME_LAST_COST is not None and ENDGAME_LAST_COST >= 3
            if (move is None or costly) and any(g["belt"] for g in rows) \
                    and time.time() - endgame_since < MAX_BELT_WAIT and time_to_spare(drags):
                # The missing copy is usually on a belt shelf still off screen; waiting for it
                # costs only time, which is no longer the scarce budget at this stage.
                time.sleep(0.5)
                board = None
                continue
            if move is not None:
                endgame_since = time.time()
                source = "plan"
            if move is not None and not endgame_logged:
                endgame_logged = True
                global ENDGAME_ACTIVE
                ENDGAME_ACTIVE = True
                print(f"{time.time() - PLAY_STARTED:6.1f}s endgame: assembling triples")
        if move is None:
            move = choose_move(board, recent, exhausted, hidden)
        if move is None and recent:
            # Every drag is barred by the anti-shuffle rules; relaxing them is better than
            # giving up with time left. The last drag stays in the history, otherwise the
            # relaxed choice is often that drag undone and the bot ping-pongs between two
            # shelves.
            move = choose_move(board, recent[-1:], exhausted, hidden)
        # A product just dropped on a belt shelf is not picked up again unless that completes a
        # triple: belt drags are left out of `recent` (belt shelves are renumbered on every
        # read), so the anti-shuffle rules alone let the bot carry it straight back.
        if move is not None and placed_on_belt and time.time() - placed_on_belt[3] < 5:
            top, number, slot, _ = placed_on_belt
            g = rows[move[0]] if move[0] < len(rows) else None
            if g and g["belt"] and g["top"] == top and memory.number(g) == number \
                    and move[1] == slot and apply_move(board, move, exhausted, hidden)[1] < CLEAR_GAIN:
                board = [list(shelf) for shelf in board]
                board[move[0]][move[1]] = UNKNOWN
                move = choose_move(board, recent, exhausted, hidden)
        # Loop guard: when the last drags already went three times between the same two
        # shelves, the product about to move again is frozen and another drag is chosen. This
        # happens when the read predicts a triple the game does not see (two similar sprites
        # read as one product), which also lets the move slip past the anti-shuffle rules.
        for _ in range(4):
            if move is None:
                break
            pair = frozenset((move[0], move[2]))
            if sum(frozenset((m[0], m[2])) == pair for m in trail[-6:]) < 3:
                break
            board = [list(shelf) for shelf in board]
            board[move[0]][move[1]] = UNKNOWN
            move = choose_move(board, recent, exhausted, hidden)
        if STRICT_CLOSING and (move is None or not closes_soon(board, move, exhausted, hidden)):
            move = closing_move(board, exhausted, hidden)
            if move is None and time.time() - last_progress < STRICT_MAX_WAIT:
                time.sleep(0.4)
                board = None
                continue
            if move is None:
                # Nothing closes for STRICT_MAX_WAIT seconds: one ordinary drag unblocks the
                # board rather than waiting out the level clock.
                move = choose_move(board, recent, exhausted, hidden)
        if move is not None and belt_seen and patience > 0:
            after, gain = apply_move(board, move, exhausted, hidden)
            if gain < CLEAR_GAIN and gain + position_value(after) - position_value(board) < WAIT_IF_GAIN_BELOW:
                patience -= 1
                time.sleep(0.3)
                board = None
                continue
        # Belt memory: a pair waiting for its third copy, which a remembered belt shelf brings
        # within MEMORY_WAIT_MAX seconds, is worth waiting for rather than spending drags.
        if LAST_PROGRESS >= MEMORY_FROM_PROGRESS and time.time() - last_drag_at < MEMORY_WAIT_MAX + 0.5 \
                and (move is None or apply_move(board, move, exhausted, hidden)[1] < CLEAR_GAIN):
            etas = [memory.arrival(label) for label in waiting_pairs(board)]
            etas = [e for e in etas if e is not None and e <= MEMORY_WAIT_MAX]
            if etas:
                time.sleep(min(0.5, min(etas) + 0.1))
                board = None
                continue
        # Helps: each one-use button saves drags, the scarce budget. The hammer clears a
        # triple of its own choosing; the shuffle deals every product again. Both go to a
        # board where the bot is stuck or only a costly plan is left; the freeze is kept for
        # the person who takes over, when the clock matters most.
        stuck = move is None or stale >= HAMMER_AFTER_STALE \
            or endgame_logged and ENDGAME_LAST_COST is not None and ENDGAME_LAST_COST >= 3
        # The freeze stops the clock for about 10 s whenever it is used, so it is used at once:
        # every second the bot saves is left for the person who finishes the level.
        if USE_HELPS and USE_FREEZE and "freeze" not in helps_used:
            helps_used.add("freeze")
            shot = capture()
            if help_available(shot, "freeze"):
                adb("shell", "input", "tap", str(HELP_BUTTONS["freeze"]), str(HELP_Y))
                print(f"{time.time() - PLAY_STARTED:6.1f}s help used: freeze")
        for name in ("hammer", "shuffle"):
            # Early on the board is full and a missing move only means waiting for the belt.
            if not USE_HELPS or name in helps_used or name == "hammer" and not USE_HAMMER \
                    or name == "hammer" and time.time() - PLAY_STARTED < HELPS_NOT_BEFORE:
                continue
            if name == "shuffle" and not (move is None or stale >= SHUFFLE_AFTER_STALE
                                          or crowded_without_triple(board)):
                continue
            if name == "hammer" and not stuck:
                continue
            shot = capture()
            helps_used.add(name)
            if help_available(shot, name):
                use_help(name)
                stale = 0
                if name == "shuffle":
                    # Every shelf now holds different products.
                    exhausted, recent, trail = set(), [], []
                    animating.clear()
                board = None
                break
        if board is None:
            continue
        if move is None and not FIXED_TARGETS_ONLY:
            move = make_room(board, rows)
            source = "room"
        belt_seen = belt_seen or any(g["belt"] for g in rows)
        if move is None and belt_seen and idle_waits < 20:
            # With every visible shelf full the belt is the only way out; it brings other
            # shelves, often with room, within a few seconds.
            idle_waits += 1
            time.sleep(0.5)
            board = None
            continue
        if move is None:
            print("No useful move found; stopping.")
            return
        idle_waits = 0
        # Signs that the greedy search is stuck: putting a product back on the shelf it came
        # from, or a run of drags that neither clears nor uncovers anything. The bot then
        # hands the game over to a human (or an assistant reading the saved screenshot).
        bounced = any(m[2] == move[0] and m[0] == move[2] for m in recent[-3:]) \
            and not apply_move(board, move, exhausted, hidden)[1] >= CLEAR_GAIN
        # In 'full' mode nobody is waiting to take over and the loop guard already breaks
        # back-and-forth drags, so only a long run without progress stops the bot; on belt
        # levels tidying up often takes more than HANDOVER_AFTER drags before a triple.
        if KEEP_PLAYING:
            bounced = False
        if bounced or stale >= (HANDOVER_AFTER_FULL if KEEP_PLAYING else HANDOVER_AFTER):
            capture().save("shots/handover.png")
            print("HANDOVER: stuck (%s); screenshot saved to shots/handover.png"
                  % ("product sent back" if bounced else f"{stale} drags without progress"))
            return
        drag(rows, move)
        last_drag_at = time.time()
        g = rows[move[2]]
        placed_on_belt = (g["top"], memory.number(g), move[3], time.time()) if g["belt"] else None
        drags += 1
        recent.append(move)
        trail.append(move)
        patience = BELT_PATIENCE
        s, k, d, dk = move
        label = board[s][k]
        predicted = [list(shelf) for shelf in board]
        predicted[s][k], predicted[d][dk] = None, label
        # Products behind a shelf only come forward when that shelf clears (three identical
        # products) or is emptied, so other drags have a predictable result. Playing on
        # predictions was faster, but a single drag the game dropped made every following
        # predicted drag wasted. The move budget, not the clock, is what runs out first, so
        # the board is read again after every drag unless PREDICT_BETWEEN_READS is set.
        reveals = predicted[d].count(label) == 3 or all(v is None for v in predicted[s])
        stale = 0 if reveals else stale + 1
        if reveals:
            last_progress = time.time()
        # A belt keeps moving and brings new shelves in, so a board with a belt is always
        # read again before the next drag.
        # Belt shelves may be filtered out of `rows` ('top' mode, or all off-screen), so the
        # level-wide flag decides whether the board can be predicted between reads.
        has_belt = LEVEL_HAS_BELT or any(g["belt"] for g in rows)
        if PREDICT_BETWEEN_READS and not has_belt and not reveals and not step_mode and drags % 6 != 0:
            print(f"{show(move)}: predicted")
            board = predicted
            time.sleep(random.uniform(0.3, 0.45))
            continue
        # Shelves whose clear or reveal animation is still running after this drag.
        revealing = [x for x in (s, d) if x == d and predicted[d].count(label) == 3
                     or x == s and all(v is None for v in predicted[s])]
        play_through = reveals and PLAY_THROUGH_REVEALS and (
            PLAY_THROUGH_BELT or all(x < FIXED_SHELVES for x in revealing))
        if not reveals or play_through:
            # Nothing animates on the fixed shelves after a plain drag, so one short pause
            # replaces the wait for two identical frames on every level. A clear on a fixed
            # shelf only animates that shelf, so the bot keeps playing elsewhere and leaves
            # the shelf alone until REVEAL_ANIMATION has passed.
            time.sleep(PAUSE_AFTER_DRAG)
            img = capture()
            for x in revealing if play_through else ():
                if x < FIXED_SHELVES:
                    animating[x] = time.time() + REVEAL_ANIMATION
                else:
                    # Belt shelves are renumbered on every read, so the animating one is
                    # remembered by its row and its centre, which moves at the row's speed.
                    g = rows[x]
                    belt_animating.append((g["top"], g["xs"][1], g["t"],
                                           g.get("speed", BELT_SPEED), time.time() + REVEAL_ANIMATION))
        else:
            # A clear or an emptied shelf animates; capturing during the animation only
            # wastes captures (about 0.65 s each), so the bot first waits REVEAL_SETTLE.
            started = time.time()
            time.sleep(REVEAL_SETTLE)
            img = wait_until_still()
            print(f"{time.time() - PLAY_STARTED:6.1f}s reveal wait {time.time() - started:.1f}s")
        if should_hand_over(img):
            img.save("shots/handover.png")
            print("HANDOVER: level at %d%%; screenshot saved to shots/handover.png" % round(progress(img) * 100))
            return
        after_rows, after, after_hidden = read_board(img, with_hidden=True)
        for _ in range(6):
            # A product being dragged by hand (in 'coop' mode) or a passing animation can hide
            # a shelf frame for a moment; only a lasting absence means the level has ended.
            if len(after_rows) >= MIN_SHELVES:
                break
            time.sleep(0.5)
            img = capture()
            after_rows, after, after_hidden = read_board(img, with_hidden=True)
        if len(after_rows) < MIN_SHELVES:
            print("No shelves on screen (level over?); stopping.")
            return
        # A shelf caught mid-animation cannot confirm the drag, so it counts as done.
        result = "ok (animating)" if play_through else outcome(board, after, move)
        print(f"{time.time() - PLAY_STARTED:6.1f}s {show(move)}: {result} [{source}]")
        if step_mode:
            img.save(f"shots/step{drags - 1}_after.png")
            print("board after: ", after)
            if drags >= 5:
                return
        # A drag that leaves the board unchanged or the target empty means the bot has the
        # wrong idea of the board; repeating such drags only burns the move budget.
        failures = failures + 1 if not result.startswith("ok") else 0
        if failures >= 3:
            print("Three drags in a row had no effect; stopping.")
            return
        rows, board, hidden = after_rows, after, after_hidden
        FIXED_SHELVES = sum(not g["belt"] for g in rows)
        # Belt shelves get new indices on every read, so remembered drags and exhausted marks
        # only stay meaningful for the fixed shelves.
        recent = [m for m in recent if m[0] < FIXED_SHELVES and m[2] < FIXED_SHELVES]
        exhausted = {x for x in exhausted if x < FIXED_SHELVES}
        # A fixed shelf still empty once its clear animation is over has nothing behind it.
        for shelf, until in list(animating.items()):
            if until <= time.time() and shelf < len(board):
                if all(v is None for v in board[shelf]):
                    exhausted.add(shelf)
                del animating[shelf]
        if reveals:
            for shelf in (s, d):
                if shelf < FIXED_SHELVES and not play_through and all(v is None for v in board[shelf]):
                    exhausted.add(shelf)
            # The slots of a shelf that revealed products now hold different ones, so drags
            # into it no longer describe what is there.
            renewed = {x for x in (s, d) if x not in exhausted}
            recent = [m for m in recent if m[2] not in renewed]
        # A notification or the system bar can briefly take the focus; only a second failed
        # check a moment later means the app was really left.
        if drags % 10 == 0 and not app_in_focus() and (time.sleep(1.5) or not app_in_focus()):
            print("Continente app is not in the foreground; stopping.")
            return


if __name__ == "__main__":
    # `kill -USR1 <pid>` prints every thread's stack, to see where a silent bot is waiting.
    import faulthandler, signal
    faulthandler.register(signal.SIGUSR1)
    mode = sys.argv[1] if len(sys.argv) > 1 else "debug"
    if mode == "debug":
        debug(sys.argv[2] if len(sys.argv) > 2 else None)
    elif mode == "full":
        KEEP_PLAYING = True
        if len(sys.argv) > 2:
            STOP_AT = float(sys.argv[2]) / 100
        play()
    elif mode == "feed":
        KEEP_PLAYING = True
        BELT_FEED = True
        started = time.time()
        try:
            play()
        finally:
            print("timings (s):", {k: round(v, 1) for k, v in TIMINGS.items()}, "total", round(time.time() - started, 1))
    elif mode == "top":
        KEEP_PLAYING = True
        SKIP_BELT = True
        play()
    elif mode == "half":
        KEEP_PLAYING = True
        TOP_HALF_ONLY = True
        STRICT_CLOSING = True
        FIXED_TARGETS_ONLY = True
        # Only the top fixed row (three shelves) is guaranteed to be on screen in this mode.
        MIN_SHELVES = 3
        if len(sys.argv) > 2:
            STOP_AT = float(sys.argv[2]) / 100
        play()
    elif mode == "cima":
        # The bot completes triples on the top fixed row, fetching products from anywhere,
        # and keeps still when no drag clears or sets up a triple there.
        KEEP_PLAYING = True
        TOP_ROW_TARGETS_ONLY = True
        STRICT_CLOSING = True
        STRICT_MAX_WAIT = float("inf")
        if len(sys.argv) > 2:
            STOP_AT = float(sys.argv[2]) / 100
        play()
    elif mode == "fixas":
        # The bot only plays the fixed shelves, and only drags that complete a triple there;
        # otherwise it keeps still. The person plays the belts and does all the tidying.
        KEEP_PLAYING = True
        SKIP_BELT = True
        STRICT_CLOSING = True
        CLEARS_ONLY = True
        STRICT_MAX_WAIT = float("inf")
        if len(sys.argv) > 2:
            STOP_AT = float(sys.argv[2]) / 100
        play()
    elif mode == "coop":
        KEEP_PLAYING = True
        RESERVED_SHELVES = frozenset({2, 5, 8})
        play()
    else:
        {"play": play, "step": lambda: play(step_mode=True)}[mode]()
