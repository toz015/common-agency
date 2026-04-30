"""Response comparison figure for HH-RLHF (3 objectives: help / harm / humor).

Adapts the safety-alignment response_comparison plot for the Helpful Assistant
case study. For a fixed prompt and preference vector α = (α_help, α_harm, α_humor),
shows the four methods' responses side-by-side with all three reward scores in
the header. Highlights three categories of content:

    Blue    = helpful (substantive music advice for the case-study prompt)
    Orange  = humor (playful style markers: emojis, banter, casual tokens)
    Purple  = harmless qualifiers (hedges, disclaimers, "not necessarily" etc.)

Outputs: code/evaluation/plots/HH-RLHF/case_study/response_comparison_<uid>_alpha<h>_<s>_<u>.png
"""
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results" / "HH-RLHF"
OUT_DIR = BASE / "plots" / "HH-RLHF" / "case_study"

# Register an emoji font once so unicode emoji glyphs render as outlines
# instead of producing missing-glyph warnings.
EMOJI_FONT_PATH = Path("/home/toz015/.local/share/fonts/TwitterColorEmoji-SVGinOT.ttf")
EMOJI_FAMILY = "DejaVu Sans Mono"
if EMOJI_FONT_PATH.is_file():
    try:
        font_manager.fontManager.addfont(str(EMOJI_FONT_PATH))
        EMOJI_FAMILY_LIST = ["DejaVu Sans Mono", "Twitter Color Emoji"]
    except Exception:
        EMOJI_FAMILY_LIST = ["DejaVu Sans Mono"]
else:
    EMOJI_FAMILY_LIST = ["DejaVu Sans Mono"]


# --------- pattern lists for the music-classes case study ---------

HELPFUL_PATTERNS = [
    r"expose\w* (?:them|babies|babies to)", r"exposure to",
    r"variety of music", r"different styles", r"different kinds of music",
    r"\bclassical\b", r"\bjazz\b", r"\bpop\b",
    r"singing to (?:them|your baby)", r"playing instruments?",
    r"music education", r"musical develop\w+",
    r"\brhythm\w*", r"\bmelod\w+",
    r"play music for", r"natural way",
    r"learn(?:ing)? (?:music|new things|best|through)",
    r"develop\w* (?:a |their |an )?(?:connection|appreciation)",
    r"good way (?:to|for)", r"best way to introduce",
    r"learn new things", r"comfortable", r"feel comfortable",
    r"routine", r"\bstructure\w*",
    r"every week", r"\bsocial experience\b",
    r"important to keep in mind",
]

HUMOR_PATTERNS = [
    r":(?:lol|grin|smile|sweat|haha|wink|upside|hmm|cool|party|smirk|heart|sad|music|sparkle):",
    r"[\U0001F300-\U0001F9FF]",
    r"[☀-➿]",
    r"\bHa+ha+\b", r"\blol\b", r"\bLOL\b", r"\blmao\b",
    r"\byou know what\??", r"\bI mean\b", r"\bjust saying\b",
    r"\byou know\b", r"\bI'?d say\b",
    r"\btotally\b", r"\bsick\b",
    r"\b(?:90|80|70|20|10)s\??\b",
    r"got luck\w*", r"sick beat",
    r"\?{2,}", r"!{2,}", r"!\?", r"\?!",
    r"\bI'?m a\b.*?expert", r"not a big deal",
    r"\bdo you\?", r"\bdon'?t you\?",
]

HARMLESS_PATTERNS = [
    r"\bhowever\b", r"\balthough\b", r"\bthat said\b",
    r"\bkeep in mind\b",
    r"not (?:a |the )?(?:good|best) way",
    r"\bnot necessarily\b", r"\bI'?m not sure\b", r"\bI don'?t think\b",
    r"\bIt'?s not (?:like|the|a)\b",
    r"\bbe careful\b", r"\bcaut\w+",
    r"\bnot (?:safe|recommended|great)\b",
]


def _find_matches(text, patterns):
    used = [False] * len(text)
    for p in patterns:
        for m in re.finditer(p, text, re.IGNORECASE):
            for i in range(m.start(), m.end()):
                used[i] = True
    return used


def annotate_response(text, max_len=800):
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0] + " ..."

    helpful = _find_matches(text, HELPFUL_PATTERNS)
    humor = _find_matches(text, HUMOR_PATTERNS)
    harmless = _find_matches(text, HARMLESS_PATTERNS)

    # Resolution priority on overlapping matches: helpful > humor > harmless
    n = len(text)
    cat = ["normal"] * n
    for i in range(n):
        if helpful[i]:
            cat[i] = "helpful"
        elif humor[i]:
            cat[i] = "humor"
        elif harmless[i]:
            cat[i] = "harmless"

    segments = []
    i = 0
    while i < n:
        j = i
        while j < n and cat[j] == cat[i]:
            j += 1
        segments.append((cat[i], text[i:j]))
        i = j
    return segments


def render_segments(ax, segments, fontsize=12):
    colors = {
        "normal":   "#333333",
        "helpful":  "#0055AA",
        "humor":    "#CC4400",
        "harmless": "#7B1FA2",
    }
    bgs = {
        "normal":   None,
        "helpful":  "#D6EAFF",
        "humor":    "#FFE4CC",
        "harmless": "#E8DCFB",
    }
    weights = {
        "normal":   "normal",
        "helpful":  "bold",
        "humor":    "bold",
        "harmless": "bold",
    }

    fig = ax.get_figure()
    renderer = fig.canvas.get_renderer()

    x_start, x_end = 0.03, 0.97
    line_height = 0.073
    x, y = x_start, 0.86

    for seg_type, seg_text in segments:
        words = seg_text.split(" ")
        for wi, word in enumerate(words):
            word_display = word + " " if wi < len(words) - 1 else word
            if not word_display.strip():
                t = ax.text(0, 0, word_display, fontsize=fontsize, family=EMOJI_FAMILY_LIST,
                            transform=ax.transAxes)
                bb_ax = t.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())
                x += bb_ax.width
                t.remove()
                continue

            t = ax.text(0, 0, word_display, fontsize=fontsize, family=EMOJI_FAMILY_LIST,
                        fontweight=weights[seg_type], transform=ax.transAxes)
            bb_ax = t.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())
            word_width = bb_ax.width
            t.remove()

            if x + word_width > x_end and x > x_start + 0.01:
                x = x_start
                y -= line_height
                if y < 0.0:
                    return

            props = dict(fontsize=fontsize, family=EMOJI_FAMILY_LIST,
                         color=colors[seg_type], fontweight=weights[seg_type],
                         transform=ax.transAxes, va="top")
            if bgs[seg_type]:
                props["bbox"] = dict(boxstyle="round,pad=0.06",
                                     facecolor=bgs[seg_type],
                                     edgecolor="none", alpha=0.85)
            ax.text(x, y, word_display, **props)
            x += word_width


def load_row(method, alphas, uid):
    h, s, u = alphas
    name = f"{method}_{h}help_{s}harm_{u}humor"
    p = RESULTS_DIR / name / "reward_result.json"
    if not p.is_file():
        return None
    for r in json.load(open(p)):
        if r["uid"] == uid:
            return r
    return None


def plot_comparison(uid, alphas, prompt_text):
    h, s, u = alphas
    method_order = ["EPEC_PARM", "EPEC_GenARM", "GenARM", "PARM"]
    display = {"EPEC_PARM": "EPEC+PARM", "EPEC_GenARM": "EPEC+GenARM",
               "GenARM": "GenARM", "PARM": "PARM"}
    header_colors = {"EPEC_PARM": "forestgreen", "EPEC_GenARM": "crimson",
                     "GenARM": "steelblue", "PARM": "purple"}

    responses = {m: load_row(m, alphas, uid) for m in method_order}
    missing = [m for m, r in responses.items() if r is None]
    if missing:
        print(f"  SKIP α=({h},{s},{u}): missing {missing}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(20, 11))
    disp_prompt = prompt_text if len(prompt_text) <= 95 else prompt_text[:92] + "..."
    fig.text(0.5, 0.99, f'Prompt: "{disp_prompt}"',
             fontsize=15, ha="center", va="top")

    legend_parts = [
        (f"Preference α = ({h}, {s}, {u})  [help, harm, humor]    |    ", "#333333"),
        ("Blue", "#0055AA"), (" = helpful    |    ", "#333333"),
        ("Orange", "#CC4400"), (" = humor    |    ", "#333333"),
        ("Purple", "#7B1FA2"), (" = harmless qualifiers", "#333333"),
    ]
    fig_renderer = fig.canvas.get_renderer()
    total_w = 0.0
    for txt, _ in legend_parts:
        t = fig.text(0, 0, txt, fontsize=13, fontweight="bold")
        total_w += t.get_window_extent(fig_renderer).transformed(fig.transFigure.inverted()).width
        t.remove()
    x_cur = 0.5 - total_w / 2
    for txt, col in legend_parts:
        t = fig.text(x_cur, 0.96, txt, fontsize=13, fontweight="bold",
                     color=col, va="top")
        x_cur += t.get_window_extent(fig_renderer).transformed(fig.transFigure.inverted()).width

    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for idx, m in enumerate(method_order):
        r, c = positions[idx]
        ax = axes[r][c]
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

        row = responses[m]
        ax.text(0.02, 0.97, display[m], ha="left", va="top",
                fontsize=14, fontweight="bold",
                color=header_colors[m], transform=ax.transAxes)
        ax.text(0.98, 0.97,
                f"Help: {row['help_score']:+.2f}    "
                f"Harm: {row['harm_score']:+.2f}    "
                f"Humor: {row['humor_score']:.3f}",
                ha="right", va="top", fontsize=11, color="#555555",
                family=EMOJI_FAMILY_LIST, transform=ax.transAxes)

        rect = FancyBboxPatch((0.005, 0.005), 0.99, 0.99,
                              boxstyle="round,pad=0.008",
                              facecolor=header_colors[m], alpha=0.04,
                              edgecolor=header_colors[m], linewidth=2,
                              transform=ax.transAxes)
        ax.add_patch(rect)

        segments = annotate_response(row["response"], max_len=800)
        render_segments(ax, segments, fontsize=12)

    plt.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.01,
                        wspace=0.05, hspace=0.06)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"response_comparison_{uid}_alpha{h}_{s}_{u}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    prompt = (
        'Are music classes good for babies? (multi-turn dialogue about Gymboree-style '
        'baby music classes; Human: "Well repeating songs is good to some extent.")'
    )
    cases = [
        ("hh_eval55", ("1.0", "0.0", "0.0"), prompt),     # help-corner
        ("hh_eval55", ("0.0", "0.0", "1.0"), prompt),     # humor-corner
        ("hh_eval55", ("0.33", "0.33", "0.34"), prompt),  # balanced centroid
    ]
    for uid, a, p in cases:
        plot_comparison(uid, a, p)


if __name__ == "__main__":
    main()
