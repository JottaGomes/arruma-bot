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

# A crop is empty when fewer than this share of its pixels differ from the white shelf.
EMPTY_FRACTION = 0.12
# Products stored behind the front row are drawn darkened. A crop whose brightest pixels
# stay below this value holds only such a product, so its front slot is free.
DARK_BRIGHTNESS = 115
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
            if y - start > 200:  # shorter grey runs are icons or text, not a shelf frame
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
    if len(rows) == 3:
        top = rows[-1][1] + ROW_GAP
        bottom = top + (rows[-1][1] - rows[-1][0])
        if bottom < px.shape[0]:
            runs = wall_runs(px, top, bottom)
            fixed_walls = all(any(abs(r[1] - edge) < 12 for r in runs) for edge in (88, 403, 718))
            if len(runs) >= 2 and not fixed_walls:
                for _, inner_left in runs:
                    right = inner_left + SHELF_INNER_WIDTH
                    travel = BELT_SPEED * BELT_SAFE_SECONDS
                    if inner_left + min(travel, 0) < 5 or right + max(travel, 0) > px.shape[1] - 5:
                        continue
                    shelves.append(dict(top=top, bottom=bottom, belt=True, t=captured_at,
                                        xs=[inner_left + o for o in SLOT_OFFSETS]))
    return shelves


def slot_crop(px, shelf, k):
    """Lower part of the product area of one slot, where only the front product is drawn."""
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
    if np.percentile(brightness, 90) < DARK_BRIGHTNESS:
        return None
    # Some hidden products show through a free slot as a pale grey ghost instead of a dark
    # silhouette; colourless pixels throughout mean the slot holds no front product.
    chroma = (c.max(axis=2) - c.min(axis=2))[not_white]
    if np.percentile(chroma, 90) < GHOST_MAX_CHROMA:
        return None
    small = Image.fromarray(crop.astype(np.uint8)).resize((10, 14), Image.BILINEAR)
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
# closest matching prototype, not the first one under the thresholds.
SAME_COLOUR_THRESHOLD = 0.35
SAME_SHAPE_THRESHOLD = 40.0


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
    prototypes, board, front_hues = [], [], {}
    for geometry in rows:
        if True:
            shelf = []
            for k in range(3):
                feature = describe(slot_crop(px, geometry, k))
                if feature is None:
                    shelf.append(None)
                    continue
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
                    hue = front_hue(slot_crop(px, geometry, k))
                    if hue is not None:
                        front_hues[label] = hue
                shelf.append(label)
            board.append(shelf)
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
        x += BELT_SPEED * (at - geometry["t"])
    return int(x), y


# Plan scores. A cleared triple is the goal; emptying a shelf is the next best thing because
# it brings the hidden products forward, which is the only source of new matches once the
# visible ones run out.
CLEAR_GAIN = 100
REVEAL_GAIN = 45
SECOND_MOVE_WEIGHT = 0.9
# Drags in a row without a cleared or emptied shelf after which the bot stops and hands over.
HANDOVER_AFTER = 10
FIRST_MOVES_EXPANDED = 25
# Drags looked ahead per decision (the drag itself included) and the replies searched at each
# deeper level.
SEARCH_DEPTH = 3
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
                if d != s and None in target:
                    yield s, k, d, target.index(None)


def position_value(board):
    """Heuristic worth of a board: pairs that can become triples, and nearly empty shelves."""
    counts = {}
    for shelf in board:
        for v in shelf:
            if v is not None and v != UNKNOWN:
                counts[v] = counts.get(v, 0) + 1
    value = 0
    for shelf in board:
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
    total_free = sum(shelf.count(None) for shelf in board)
    if total_free < 2:
        value -= 60 * (2 - total_free)
    return value


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
        if churn(move, gain):
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
        first.append((gain + position_value(after), after, gain, move))
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
        total = gain + SECOND_MOVE_WEIGHT * best_continuation(after, SEARCH_DEPTH - 1)
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
# Fixed shelves (0-based, row by row) left to a person playing alongside the bot; set by the
# 'coop' mode to the right-hand column. The belt stays shared.
RESERVED_SHELVES = frozenset()
# Set by the 'top' mode: the person plays the belt, so the bot ignores belt shelves entirely.
SKIP_BELT = False
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
    return np.asarray(img.crop((0, 560, right, 1640)).convert("L").resize((36, 54)), dtype=float)


def wait_until_still(timeout=3.0):
    """Captures until two consecutive frames match, so the board is read after the animations."""
    previous = board_signature(capture())
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(0.2)
        img = capture()
        current = board_signature(img)
        if np.abs(current - previous).mean() < 1.0:
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
    if s >= 9 or d >= 9 or d >= len(after):
        return "ok"
    if before == after:
        return "nothing changed"
    # A cleared shelf is entirely free afterwards, so a free destination is only a failure
    # when the rest of that shelf still holds products.
    if after[d][dk] is None and any(v is not None for v in after[d]):
        return "destination still free"
    return "ok"


def measure_belt():
    """Updates BELT_SPEED from two screenshots, when the level has a belt."""
    global BELT_SPEED
    first = capture()
    time.sleep(0.4)
    second = capture()
    rows1 = [g for g in shelf_geometry(np.asarray(first), first.info["t"]) if g["belt"]]
    rows2 = [g for g in shelf_geometry(np.asarray(second), second.info["t"]) if g["belt"]]
    if not rows1 or not rows2:
        return
    dt = second.info["t"] - first.info["t"]
    shifts = [b["xs"][0] - a["xs"][0] for a in rows1 for b in rows2]
    # Shelves are 315 px apart, so the true shift is the smallest one in magnitude.
    shift = min(shifts, key=abs)
    if abs(shift) < 300 and dt > 0:
        BELT_SPEED = shift / dt
        print(f"belt speed {BELT_SPEED:.0f} px/s")


def play(step_mode=False):
    """Plays until the level ends. In step mode it makes at most five drags, saving the board
    before and after each one in shots/ and printing how the drag was interpreted."""
    recent, failures, drags, exhausted, stale = [], 0, 0, set(), 0
    rows, board = None, None
    idle_waits = 0
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
                if len(find_shelf_rows(np.asarray(img))) >= 3:
                    break
                time.sleep(0.5)
                img = capture()
            if progress(img) >= HANDOVER_PROGRESS and not KEEP_PLAYING:
                img.save("shots/handover.png")
                print("HANDOVER: level at %d%%; screenshot saved to shots/handover.png" % round(progress(img) * 100))
                return
            rows, board, hidden = read_board(img, with_hidden=True)
            if len(rows) < 9:
                print("No shelves on screen (level over?); stopping.")
                return
            if step_mode:
                img.save(f"shots/step{drags}_before.png")
                print("board before:", board)
        if RESERVED_SHELVES:
            # Shelves a person is playing at the same time: shown to the planner as full of
            # unidentified products, so they are never a source or a destination.
            board = [[UNKNOWN] * 3 if i in RESERVED_SHELVES else shelf for i, shelf in enumerate(board)]
        move = choose_move(board, recent, exhausted, hidden)
        if move is None and recent:
            # Every drag is barred by the anti-shuffle rules; dropping the history once is
            # better than giving up with time left.
            move = choose_move(board, [], exhausted, hidden)
        if move is None and any(g["belt"] for g in rows) and idle_waits < 20:
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
        if bounced or stale >= HANDOVER_AFTER:
            capture().save("shots/handover.png")
            print("HANDOVER: stuck (%s); screenshot saved to shots/handover.png"
                  % ("product sent back" if bounced else f"{stale} drags without progress"))
            return
        drag(rows, move)
        drags += 1
        recent.append(move)
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
        # A belt keeps moving and brings new shelves in, so a board with a belt is always
        # read again before the next drag.
        # Belt shelves may be filtered out of `rows` ('top' mode, or all off-screen), so the
        # level-wide flag decides whether the board can be predicted between reads.
        has_belt = LEVEL_HAS_BELT
        if PREDICT_BETWEEN_READS and not has_belt and not reveals and not step_mode and drags % 6 != 0:
            print(f"{show(move)}: predicted")
            board = predicted
            time.sleep(random.uniform(0.3, 0.45))
            continue
        if has_belt and not reveals:
            # Nothing animates on the fixed shelves after a plain drag, so one short pause
            # replaces the wait for two identical frames. Levels with a belt have no move limit
            # and are lost on time, which makes this the main speed-up there.
            time.sleep(0.35)
            img = capture()
        else:
            img = wait_until_still()
        if progress(img) >= HANDOVER_PROGRESS and not KEEP_PLAYING:
            img.save("shots/handover.png")
            print("HANDOVER: level at %d%%; screenshot saved to shots/handover.png" % round(progress(img) * 100))
            return
        after_rows, after, after_hidden = read_board(img, with_hidden=True)
        for _ in range(6):
            # A product being dragged by hand (in 'coop' mode) or a passing animation can hide
            # a shelf frame for a moment; only a lasting absence means the level has ended.
            if len(after_rows) >= 9:
                break
            time.sleep(0.5)
            img = capture()
            after_rows, after, after_hidden = read_board(img, with_hidden=True)
        if len(after_rows) < 9:
            print("No shelves on screen (level over?); stopping.")
            return
        result = outcome(board, after, move)
        print(f"{show(move)}: {result}")
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
        # Belt shelves get new indices on every read, so remembered drags and exhausted marks
        # only stay meaningful for the nine fixed shelves.
        recent = [m for m in recent if m[0] < 9 and m[2] < 9]
        exhausted = {x for x in exhausted if x < 9}
        if reveals:
            for shelf in (s, d):
                if shelf < 9 and all(v is None for v in board[shelf]):
                    exhausted.add(shelf)
            # The slots of a shelf that revealed products now hold different ones, so drags
            # into it no longer describe what is there.
            renewed = {x for x in (s, d) if x not in exhausted}
            recent = [m for m in recent if m[2] not in renewed]
        if drags % 10 == 0 and not app_in_focus():
            print("Continente app is not in the foreground; stopping.")
            return


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "debug"
    if mode == "debug":
        debug(sys.argv[2] if len(sys.argv) > 2 else None)
    elif mode == "full":
        KEEP_PLAYING = True
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
    elif mode == "coop":
        KEEP_PLAYING = True
        RESERVED_SHELVES = frozenset({2, 5, 8})
        play()
    else:
        {"play": play, "step": lambda: play(step_mode=True)}[mode]()
