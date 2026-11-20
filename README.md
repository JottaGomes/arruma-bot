# arruma-bot

Plays **Arruma a Loja**, the shelf-sorting mini-game in the Continente app, on an Android phone over adb.

Each shelf holds three products. Three identical products on one shelf clear it, and the products stored behind (drawn darker) move to the front. The bot repeats the same loop:

1. Captures the screen.
2. Finds the shelves and reads the front and hidden products, grouping identical ones by their pixels. No product catalogue is needed.
3. Picks the next drag with a short look-ahead search.
4. Performs the drag as a fast `input swipe` onto a free slot.

## Requirements

- Python 3 with the packages in `requirements.txt`
- `adb`, with the phone connected by USB or wireless debugging

## Usage

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export ARRUMA_DEVICE=<adb serial or ip:port>   # optional when only one device is connected
.venv/bin/python bot.py debug   # read one screenshot, write debug.png, no input
.venv/bin/python bot.py step    # five drags, checking the board around each
.venv/bin/python bot.py play    # play and stop to hand over once progress stalls
.venv/bin/python bot.py full    # play until the level ends
.venv/bin/python bot.py full 85 # play until 85% progress, then stop so a person can finish
```

`ARRUMA_HAMMER=0` leaves the hammer help to the person who finishes the level.

Open the level and wait until the shelves are on screen before starting.

## Notes

- Products that look dark are hidden products behind a **free** front slot, not products of their own. Dragging onto a full shelf does nothing.
- Only a fast swipe (about 230–330 ms) onto a free slot registers as a move. `input draganddrop`, slow swipes and taps on products are ignored by the game.
- Screen positions were measured on a 1080x2340 screen. On level 21 the shelf frame sits at x=70 (`FRAME_PROBE_X`).
- Levels with conveyor belts (22, 26): the speed of each belt row is measured at start, and drags onto a belt shelf aim where it will be when the finger lands. Belt shelves are remembered by position after they scroll away; once one comes back, the loop length is known and the bot waits for a missing copy that is about to reappear instead of spending drags.
- Endgame: once fixed shelves run empty, the bot assembles triples one at a time, cheapest first (a pair with a free slot before a pair with an odd product, before an empty shelf), and never leaves a product on a belt shelf unless that completes a triple.
- Helps under the board (hammer, freeze, shuffle) are used once each: freeze at the start, shuffle on a crowded board without a one-drag triple, hammer when stuck. The game's shuffle groups identical products, so it pays most early.
- On level 26 the move budget (290), not the clock, decides the result: the bot alone reached 98%.
