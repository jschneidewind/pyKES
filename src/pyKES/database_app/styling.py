"""
Visual theme for the database application.

Deliberately small. The colours and base theme come from Streamlit's own
configuration in ``.streamlit/config.toml``; this module adds only what
configuration cannot express — the typeface and a few spacing and weight
corrections — through one stylesheet applied at the top of every page. No
custom components, no markup injected around widgets, nothing that a Streamlit
upgrade can silently break.
"""

import streamlit as st


# =============================================================================
# Palette
# =============================================================================

# Green carries meaning here rather than decoration: it marks the active page,
# the primary action, and healthy status. Everything else stays neutral so the
# green is worth noticing.
ACCENT = "#22c55e"
ACCENT_SOFT = "rgba(34, 197, 94, 0.12)"
SURFACE_BORDER = "rgba(148, 163, 184, 0.16)"

# Grotesque with a large x-height, legible at the small sizes a dense metadata
# table needs. Loaded from Google Fonts with a system stack behind it, so the
# page still renders correctly when the font host is unreachable — which, on an
# air-gapped instrument network, it will be.
FONT_URL = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
FONT_STACK = ('"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", '
              'Roboto, sans-serif')

THEME_STYLESHEET = f"""
<style>
@import url('{FONT_URL}');

/* Set the typeface on the containers and let it inherit. Deliberately not on
   [class*="st-"]: that also matches Streamlit's Material icon spans, whose
   glyphs are ligatures of their own font, and overriding it renders expanders
   with the literal text "keyboard_arrow_right" instead of a caret. */
html, body, button, input, select, textarea,
[data-testid="stAppViewContainer"],
[data-testid="stSidebar"],
[data-testid="stHeader"] {{
    font-family: {FONT_STACK};
}}

/* Headings tighter and heavier than the default, so a page reads as a
   dashboard rather than as a document. */
h1 {{ font-size: 1.65rem; font-weight: 700; letter-spacing: -0.02em; }}
h2 {{ font-size: 1.2rem;  font-weight: 600; letter-spacing: -0.01em; }}
h3 {{ font-size: 1.0rem;  font-weight: 600; }}

/* Panels: a hairline and a radius are enough to group content without the
   heavy card shadows that make a dense page noisy. Padding is deliberately not
   set on the expander — it shifts the widget's own caret out of place. */
div[data-testid="stMetric"] {{
    border: 1px solid {SURFACE_BORDER};
    border-radius: 10px;
    padding: 0.5rem 0.75rem;
}}
div[data-testid="stExpander"] {{
    border: 1px solid {SURFACE_BORDER};
    border-radius: 10px;
}}
div[data-testid="stMetricValue"] {{ font-weight: 600; letter-spacing: -0.02em; }}

/* The filter groups in the sidebar are named after the entity they describe, so
   their labels are headings and are sized like them rather than like body text. */
[data-testid="stSidebar"] div[data-testid="stExpander"] summary p {{
    font-size: 0.95rem;
    font-weight: 600;
}}

/* The active page in the sidebar is the one thing worth an accent. */
div[data-testid="stSidebarNav"] a[aria-current="page"] {{
    background: {ACCENT_SOFT};
    border-radius: 8px;
    font-weight: 600;
}}

/* Numbers in tables line up when they share a width. */
div[data-testid="stDataFrame"] {{ font-variant-numeric: tabular-nums; }}

button[kind="primary"] {{ font-weight: 600; }}
</style>
"""


def apply_theme() -> None:
    """
    Apply the stylesheet to the current page.

    Called once at the top of every page, before anything is drawn.

    Returns
    -------
    None : None
    """
    st.markdown(THEME_STYLESHEET, unsafe_allow_html=True)
