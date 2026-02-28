"""
Pinata Wins - Passive Slot Watcher Agent
=========================================
Single-file agent that:
  1. Opens game URL in browser
  2. Clicks GET STARTED on splash screen
  3. Detects the game board on screen
  4. Reads all 15 symbols (3x5 grid) using GPT-4o Vision
  5. Watches for spins -> re-reads board after each spin stops

Usage:
  python watcher.py              # full flow: open browser + watch
  python watcher.py --watch      # attach to already-open game

Requirements:
  pip install opencv-python numpy pyautogui pillow requests
"""

import os
import sys
import time
import json
import base64
import logging
import platform
import re
import numpy as np
import cv2
import pyautogui
import requests
from datetime import datetime, timezone
from PIL import ImageGrab

# ── Paths ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR  = os.path.join(BASE_DIR, "logs")
ASSETS   = os.path.join(BASE_DIR, "assets")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "watcher.log"),
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
)

# ── Config ────────────────────────────────────────────────
GRID_ROWS = 3
GRID_COLS = 5

# Spin detection
MOTION_THRESHOLD   = 18.0   # mean pixel diff to call "moving"
SPIN_START_FRAMES  = 3      # consecutive moving frames to confirm spin
SPIN_STOP_FRAMES   = 8      # consecutive calm frames to confirm stop
MIN_SPIN_DURATION  = 1.0    # ignore "spins" shorter than this (sec)
WATCH_FPS          = 10     # capture rate during spin
IDLE_FPS           = 3      # capture rate when idle

# Image sizing for API (keep small for speed)
API_IMAGE_WIDTH    = 512    # resize images to this width before sending
JPEG_QUALITY       = 65     # JPEG quality for API uploads

SYMBOLS = [
    "scatter", "multiplier", "wild", "girl", "skull", "sombrero",
    "shawarma_taco", "maracas_pair", "chili_pepper",
    "A_letter", "K_letter", "Q_letter", "J_letter",
]


# ═══════════════════════════════════════════════════════════
#  API KEY
# ═══════════════════════════════════════════════════════════

def load_openai_key():
    env_path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8-sig") as f:
            for line in f:
                if line.strip().startswith("OPENAI_API_KEY="):
                    return line.strip().split("=", 1)[1].strip().strip("'\"")
    return os.environ.get("OPENAI_API_KEY", "")


# ═══════════════════════════════════════════════════════════
#  SCREEN CAPTURE
# ═══════════════════════════════════════════════════════════

def screenshot_full():
    """Full screen -> BGR numpy array."""
    return cv2.cvtColor(np.array(ImageGrab.grab()), cv2.COLOR_RGB2BGR)


def screenshot_region(x, y, w, h):
    """Region -> (gray, bgr) numpy arrays."""
    img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), bgr


def img_to_b64_jpeg(bgr, max_width=API_IMAGE_WIDTH):
    """Resize + JPEG encode -> base64 string (small for API)."""
    h, w = bgr.shape[:2]
    if w > max_width:
        scale = max_width / w
        bgr = cv2.resize(bgr, (max_width, int(h * scale)))
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return base64.b64encode(buf).decode()


# ═══════════════════════════════════════════════════════════
#  GPT-4o VISION CALLS
# ═══════════════════════════════════════════════════════════

def gpt4o_call(api_key, prompt, bgr_img, max_tokens=300, model="gpt-4o-mini"):
    """Send image + prompt to GPT-4o, return response text or None."""
    b64 = img_to_b64_jpeg(bgr_img)
    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": "low",
                    }},
                ]}],
                "max_tokens": max_tokens,
                "temperature": 0.1,
            },
            timeout=30,
        )
        if r.status_code != 200:
            logging.error(f"GPT-4o HTTP {r.status_code}: {r.text[:200]}")
            return None
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logging.error(f"GPT-4o exception: {e}")
        return None


def detect_screen_state(api_key, bgr_img):
    """Ask GPT-4o what's on screen: 'splash', 'game', or 'other'."""
    prompt = (
        "What is shown in this screenshot? Reply with ONLY one word:\n"
        "- 'splash' if it shows a game loading, start, or promotional screen\n"
        "- 'game' if a slot game board with a grid of symbol icons is visible\n"
        "- 'other' if it is neither"
    )
    resp = gpt4o_call(api_key, prompt, bgr_img, max_tokens=10)
    if resp:
        word = resp.strip().lower().strip(".'\"")
        if "game" in word:
            return "game"
        if "splash" in word or "start" in word or "load" in word:
            return "splash"
    return "other"


GRID_PROMPT = (
    "This image shows a Mexican fiesta themed game with a 3 rows x 5 columns grid of icons. "
    "Identify each icon in the grid reading left-to-right, top-to-bottom.\n\n"
    "Possible icons (use exact names):\n"
    "scatter (pinata), multiplier (xN badge), wild (Wild badge), "
    "girl (girl character), skull (decorated skull), sombrero (hat), "
    "shawarma_taco (taco), maracas_pair (maracas), chili_pepper (pepper), "
    "A_letter, K_letter, Q_letter, J_letter\n\n"
    "Return ONLY valid JSON:\n"
    '{"grid":[["r1c1","r1c2","r1c3","r1c4","r1c5"],'
    '["r2c1","r2c2","r2c3","r2c4","r2c5"],'
    '["r3c1","r3c2","r3c3","r3c4","r3c5"]]}'
)


def read_grid_symbols(api_key, bgr_img):
    """Send grid image to GPT-4o, get 3x5 symbol names back.
    Returns list[3][5] of symbol names, or None.
    """
    # Try gpt-4o first (more accurate), fallback to mini
    for model in ["gpt-4o", "gpt-4o-mini"]:
        resp = gpt4o_call(api_key, GRID_PROMPT, bgr_img, max_tokens=300, model=model)
        if not resp:
            continue

        text = resp.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            logging.warning(f"[{model}] No JSON: {text[:100]}")
            continue

        try:
            data = json.loads(text[start:end])
        except json.JSONDecodeError:
            logging.warning(f"[{model}] JSON error: {text[:100]}")
            continue

        grid = data.get("grid", [])
        if len(grid) != 3 or any(len(row) != 5 for row in grid):
            logging.warning(f"[{model}] Bad grid: {[len(r) for r in grid]}")
            continue

        return [[normalize_sym(s) for s in row] for row in grid]

    return None


def normalize_sym(name):
    name = str(name).strip().lower()
    aliases = {
        "taco": "shawarma_taco", "shawarma": "shawarma_taco",
        "maracas": "maracas_pair", "maraca": "maracas_pair",
        "chili": "chili_pepper", "pepper": "chili_pepper",
        "a": "A_letter", "k": "K_letter", "q": "Q_letter", "j": "J_letter",
        "a_letter": "A_letter", "k_letter": "K_letter",
        "q_letter": "Q_letter", "j_letter": "J_letter",
        "pinata": "scatter", "free_spin": "scatter",
    }
    if name in aliases:
        return aliases[name]
    for valid in SYMBOLS:
        if name == valid.lower():
            return valid
    for valid in SYMBOLS:
        if valid.lower() in name or name in valid.lower():
            return valid
    return name


# ═══════════════════════════════════════════════════════════
#  BOARD DETECTION
# ═══════════════════════════════════════════════════════════

def find_game_viewport():
    """Find the game area on screen via color saturation analysis.
    Returns (x, y, w, h) or None.
    """
    screen = screenshot_full()
    h, w = screen.shape[:2]

    hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] > 30) & (hsv[:, :, 2] > 40)).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    biggest = max(contours, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(biggest)

    if bw < w * 0.20 or bh < h * 0.20:
        return None

    return (bx, by, bw, bh)


def estimate_grid_from_viewport(vx, vy, vw, vh):
    """Estimate the 3x5 grid region within the game viewport.
    Grid is roughly center of the game, skipping top UI and bottom buttons.
    """
    gx = int(vx + vw * 0.15)
    gy = int(vy + vh * 0.18)
    gw = int(vw * 0.70)
    gh = int(vh * 0.55)
    return (gx, gy, gw, gh)


# ═══════════════════════════════════════════════════════════
#  SPLASH SCREEN HANDLING
# ═══════════════════════════════════════════════════════════

def click_get_started(api_key):
    """Find and click the GET STARTED button on splash screen."""
    btn_path = os.path.join(ASSETS, "game_start_button.png")
    screen = screenshot_full()

    # Template matching
    if os.path.exists(btn_path):
        btn_img = cv2.imread(btn_path)
        if btn_img is not None:
            gray_scr = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            gray_btn = cv2.cvtColor(btn_img, cv2.COLOR_BGR2GRAY)
            sh, sw = gray_scr.shape

            for scale in [0.08, 0.10, 0.12, 0.15, 0.20, 0.25]:
                tw = int(gray_btn.shape[1] * scale)
                th = int(gray_btn.shape[0] * scale)
                if tw >= sw or th >= sh or tw < 20 or th < 20:
                    continue
                resized = cv2.resize(gray_btn, (tw, th))
                res = cv2.matchTemplate(gray_scr, resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > 0.5:
                    cx = max_loc[0] + tw // 2
                    cy = max_loc[1] + th // 2
                    print(f"    Found button (conf={max_val:.2f}), clicking ({cx},{cy})")
                    pyautogui.click(cx, cy)
                    return True

    # GPT-4o fallback
    print("    Asking GPT-4o to find start button...")
    prompt = (
        "Find the GET STARTED or PLAY button in this game screen. "
        "Reply with ONLY the pixel coordinates as: x,y"
    )
    resp = gpt4o_call(api_key, prompt, screen, max_tokens=30)
    if resp:
        nums = re.findall(r'\d+', resp)
        if len(nums) >= 2:
            h, w = screen.shape[:2]
            sw = min(API_IMAGE_WIDTH, w)
            scale_factor = w / sw
            rx = int(int(nums[0]) * scale_factor)
            ry = int(int(nums[1]) * scale_factor)
            if 0 < rx < w and 0 < ry < h:
                print(f"    GPT-4o says button at ({rx},{ry}), clicking...")
                pyautogui.click(rx, ry)
                return True

    # Fallback: click center-bottom
    h, w = screen.shape[:2]
    pyautogui.click(w // 2, int(h * 0.75))
    print("    Clicked center-bottom as fallback")
    return True


def wait_for_game_board(api_key, timeout=90):
    """Wait until game board with symbol grid is visible.
    Handles splash screen automatically.
    Returns viewport (x,y,w,h) or None.
    """
    start = time.time()
    splash_clicked = False
    last_state = ""

    while time.time() - start < timeout:
        viewport = find_game_viewport()
        if not viewport:
            print("    No game area found, waiting...")
            time.sleep(3)
            continue

        vx, vy, vw, vh = viewport
        _, board_bgr = screenshot_region(vx, vy, vw, vh)
        state = detect_screen_state(api_key, board_bgr)

        if state != last_state:
            print(f"    Screen state: {state}")
            last_state = state

        if state == "game":
            print(f"    Game board detected!")
            return viewport

        if state == "splash" and not splash_clicked:
            print("    Splash screen - clicking GET STARTED...")
            click_get_started(api_key)
            splash_clicked = True
            time.sleep(5)
            continue

        time.sleep(3)

    return None


# ═══════════════════════════════════════════════════════════
#  DISPLAY & LOGGING
# ═══════════════════════════════════════════════════════════

def print_grid(grid):
    if not grid:
        print("    (no grid)")
        return
    cw = 16
    sep = "-" * (cw * GRID_COLS + GRID_COLS + 1)
    print(f"    {sep}")
    for ri, row in enumerate(grid):
        cells = " | ".join(f"{s:^{cw-2}}" for s in row)
        print(f"    | {cells} |")
        if ri < len(grid) - 1:
            print(f"    {sep}")
    print(f"    {sep}")


def log_spin(num, before, after, duration):
    entry = {
        "spin": num,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_sec": round(duration, 2),
        "before": before,
        "after": after,
    }
    path = os.path.join(LOG_DIR, "spin_log.json")
    data = []
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            pass
    data.append(entry)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ═══════════════════════════════════════════════════════════
#  SPIN MONITORING
# ═══════════════════════════════════════════════════════════

def compute_motion(prev_gray, curr_gray):
    if prev_gray is None or prev_gray.shape != curr_gray.shape:
        return 0.0
    return float(np.mean(cv2.absdiff(prev_gray, curr_gray)))


def read_board(grid_region, api_key):
    """Capture grid area, identify symbols via GPT-4o."""
    x, y, w, h = grid_region
    _, bgr = screenshot_region(x, y, w, h)
    cv2.imwrite(os.path.join(LOG_DIR, "debug_grid.png"), bgr)

    print("    Sending to GPT-4o...", end=" ", flush=True)
    t0 = time.time()
    grid = read_grid_symbols(api_key, bgr)
    elapsed = time.time() - t0

    if grid:
        print(f"OK ({elapsed:.1f}s)")
    else:
        print(f"FAILED ({elapsed:.1f}s)")
    return grid


def watch_for_spins(grid_region, api_key):
    """Main loop: IDLE -> SPINNING -> IDLE with motion detection."""
    print("\n" + "=" * 55)
    print("  SPIN MONITOR - watching for slot spins")
    print("  Press Ctrl+C to stop")
    print("=" * 55)

    x, y, w, h = grid_region

    # Initial read
    print("\n  Reading initial board...")
    current_grid = read_board(grid_region, api_key)
    if current_grid:
        print("\n  CURRENT BOARD:")
        print_grid(current_grid)

    state = "IDLE"
    spin_count = 0
    motion_frames = 0
    calm_frames = 0
    spin_start = 0
    pre_spin_grid = None
    prev_gray = None

    print(f"\n  Watching... (spin the slot manually)\n")

    while True:
        try:
            fps = IDLE_FPS if state == "IDLE" else WATCH_FPS
            time.sleep(1.0 / fps)

            gray, _ = screenshot_region(x, y, w, h)
            motion = compute_motion(prev_gray, gray)
            prev_gray = gray

            # ── IDLE: wait for spin ──
            if state == "IDLE":
                if motion > MOTION_THRESHOLD:
                    motion_frames += 1
                    if motion_frames >= SPIN_START_FRAMES:
                        state = "SPINNING"
                        spin_start = time.time()
                        pre_spin_grid = current_grid
                        calm_frames = 0
                        print(f"  >>> SPIN DETECTED (motion={motion:.1f})")
                else:
                    motion_frames = 0

            # ── SPINNING: wait for stop ──
            elif state == "SPINNING":
                if motion < MOTION_THRESHOLD:
                    calm_frames += 1
                    if calm_frames >= SPIN_STOP_FRAMES:
                        duration = time.time() - spin_start

                        if duration < MIN_SPIN_DURATION:
                            print(f"  >>> False alarm ({duration:.1f}s), ignoring")
                            state = "IDLE"
                            motion_frames = 0
                            calm_frames = 0
                            continue

                        spin_count += 1
                        print(f"\n  >>> SPIN #{spin_count} STOPPED ({duration:.1f}s)")

                        time.sleep(0.5)

                        print("  Reading new board...")
                        new_grid = read_board(grid_region, api_key)

                        if new_grid:
                            current_grid = new_grid
                            print(f"\n  {'='*55}")
                            print(f"  SPIN #{spin_count} RESULT ({duration:.1f}s)")
                            print(f"  {'='*55}")

                            if pre_spin_grid:
                                print("\n  BEFORE:")
                                print_grid(pre_spin_grid)

                            print("\n  AFTER:")
                            print_grid(current_grid)

                            if pre_spin_grid:
                                changes = []
                                for r in range(GRID_ROWS):
                                    for c in range(GRID_COLS):
                                        if pre_spin_grid[r][c] != current_grid[r][c]:
                                            changes.append(
                                                f"    R{r+1}C{c+1}: {pre_spin_grid[r][c]} -> {current_grid[r][c]}"
                                            )
                                if changes:
                                    print(f"\n  CHANGES ({len(changes)}):")
                                    for ch in changes:
                                        print(ch)
                                else:
                                    print("\n  No changes detected.")

                            log_spin(spin_count, pre_spin_grid, current_grid, duration)
                        else:
                            print("  Could not read board after spin")

                        print(f"\n  Watching for next spin...\n")
                        state = "IDLE"
                        motion_frames = 0
                        calm_frames = 0
                else:
                    calm_frames = 0

                    elapsed = time.time() - spin_start
                    if int(elapsed * WATCH_FPS) % (WATCH_FPS * 2) == 0 and elapsed > 0.5:
                        print(f"    spinning... {elapsed:.0f}s")

        except KeyboardInterrupt:
            print(f"\n\n  Stopped. Total spins: {spin_count}")
            break


# ═══════════════════════════════════════════════════════════
#  BROWSER
# ═══════════════════════════════════════════════════════════

def load_game_url():
    path = os.path.join(BASE_DIR, "ruleset", "tactic.json")
    try:
        with open(path, encoding="utf-8-sig") as f:
            return json.load(f).get("url", "")
    except Exception:
        return ""


def open_browser(url):
    print(f"  Opening: {url[:80]}...")
    try:
        if platform.system() == "Windows":
            os.startfile(url)
        else:
            import subprocess
            subprocess.Popen(["xdg-open", url])
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main(skip_browser=False):
    print("=" * 55)
    print("  PINATA WINS - Slot Watcher Agent")
    print("=" * 55)
    print()

    # 1. API key
    api_key = load_openai_key()
    if not api_key:
        print("ERROR: No OPENAI_API_KEY in .env")
        return
    print(f"  OpenAI key: OK")

    # 2. Browser
    if not skip_browser:
        url = load_game_url()
        if not url:
            print("ERROR: No URL in ruleset/tactic.json")
            return
        if not open_browser(url):
            return
        print("  Waiting for page to load...")
        time.sleep(8)

    # 3. Find game board (handles splash)
    print("\n  Looking for game board...")
    viewport = wait_for_game_board(api_key, timeout=90)
    if not viewport:
        print("ERROR: Game board not found")
        cv2.imwrite(os.path.join(LOG_DIR, "debug_fullscreen.png"), screenshot_full())
        return

    vx, vy, vw, vh = viewport
    print(f"  Viewport: ({vx},{vy}) {vw}x{vh}")

    # 4. Grid region
    grid_region = estimate_grid_from_viewport(vx, vy, vw, vh)
    gx, gy, gw, gh = grid_region
    print(f"  Grid: ({gx},{gy}) {gw}x{gh}  cells: {gw//GRID_COLS}x{gh//GRID_ROWS}")

    if gw // GRID_COLS < 30 or gh // GRID_ROWS < 30:
        print("ERROR: Cells too small")
        return

    # 5. Watch
    watch_for_spins(grid_region, api_key)


if __name__ == "__main__":
    if "--watch" in sys.argv:
        main(skip_browser=True)
    else:
        main(skip_browser=False)
