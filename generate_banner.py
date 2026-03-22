#!/usr/bin/env python3
"""Generate a professional banner image for the Fine-Tuning vs RAG project."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path


def draw_rounded_rect(ax, x, y, w, h, color, alpha=1.0, radius=0.02, linewidth=0, edgecolor=None):
    """Draw a rounded rectangle."""
    fancy = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=color, alpha=alpha,
        edgecolor=edgecolor or color,
        linewidth=linewidth
    )
    ax.add_patch(fancy)
    return fancy


def create_banner():
    fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('#0a0e27')

    # --- Background gradient effect using overlapping rectangles ---
    for i in range(50):
        alpha = 0.015
        y_pos = i / 50
        color = plt.cm.cool(y_pos * 0.3 + 0.1)
        ax.axhspan(y_pos, y_pos + 0.02, color=color, alpha=alpha)

    # --- Subtle grid pattern ---
    for x in np.arange(0, 1.01, 0.05):
        ax.axvline(x=x, color='#1a2455', alpha=0.3, linewidth=0.3)
    for y in np.arange(0, 1.01, 0.05):
        ax.axhline(y=y, color='#1a2455', alpha=0.3, linewidth=0.3)

    # --- Decorative data points (scatter) ---
    np.random.seed(42)
    n_points = 80
    x_pts = np.random.uniform(0.02, 0.98, n_points)
    y_pts = np.random.uniform(0.02, 0.98, n_points)
    sizes = np.random.uniform(2, 15, n_points)
    colors_scatter = np.random.choice(['#00d4ff', '#7c4dff', '#ff6b9d', '#00e676'], n_points)
    alphas = np.random.uniform(0.05, 0.2, n_points)
    for xp, yp, s, c, a in zip(x_pts, y_pts, sizes, colors_scatter, alphas):
        ax.scatter(xp, yp, s=s, color=c, alpha=a, zorder=1)

    # --- Neural network connection lines (decorative) ---
    np.random.seed(7)
    for _ in range(25):
        x1, y1 = np.random.uniform(0.0, 0.3), np.random.uniform(0.0, 1.0)
        x2, y2 = np.random.uniform(0.7, 1.0), np.random.uniform(0.0, 1.0)
        ax.plot([x1, x2], [y1, y2], color='#1a3a6a', alpha=0.08, linewidth=0.5, zorder=1)

    # --- Main title area (frosted glass effect) ---
    draw_rounded_rect(ax, 0.05, 0.58, 0.90, 0.36, '#0d1440', alpha=0.75, radius=0.02,
                      linewidth=1.5, edgecolor='#2a4a8a')

    # --- Title ---
    ax.text(0.50, 0.855, "FINE-TUNING vs RAG", fontsize=32, fontweight='bold',
            color='#ffffff', ha='center', va='center', zorder=10,
            fontfamily='sans-serif',
            bbox=dict(boxstyle='round,pad=0.01', facecolor='none', edgecolor='none'))

    # --- Subtitle line ---
    ax.text(0.50, 0.785, "A Data-Driven Benchmark for LLM Specialization",
            fontsize=14, color='#8ecae6', ha='center', va='center', zorder=10,
            fontfamily='sans-serif', style='italic')

    # --- Divider line ---
    ax.plot([0.15, 0.85], [0.745, 0.745], color='#3a6ea5', alpha=0.6, linewidth=1.5, zorder=10)

    # --- Tagline ---
    ax.text(0.50, 0.705, "253 Experiments  \u2022  7 Models  \u2022  6 Benchmark Suites",
            fontsize=11, color='#b8d4e8', ha='center', va='center', zorder=10,
            fontfamily='sans-serif')

    # --- Key insight ---
    ax.text(0.50, 0.645, '"Fine-tuning teaches skills. RAG provides knowledge. Hybrid wins."',
            fontsize=10.5, color='#ffd166', ha='center', va='center', zorder=10,
            fontfamily='sans-serif', style='italic')

    # --- Three pillar cards ---
    card_y = 0.28
    card_h = 0.24
    card_w = 0.26
    gap = 0.035

    cards = [
        {
            'x': 0.05 + gap,
            'color': '#1a237e',
            'border': '#5c6bc0',
            'icon': '\u2699',  # gear
            'title': 'Fine-Tuning',
            'subtitle': 'Teaches Skills',
            'stats': ['85% Sentiment', '70% Numerical', '4x Faster'],
            'stat_color': '#7986cb'
        },
        {
            'x': 0.05 + gap + card_w + gap,
            'color': '#004d40',
            'border': '#26a69a',
            'icon': '\U0001F50D',  # magnifying glass
            'title': 'RAG',
            'subtitle': 'Provides Knowledge',
            'stats': ['87% Aligned Data', '3.8/5 Faithful', '12 Documents'],
            'stat_color': '#80cbc4'
        },
        {
            'x': 0.05 + gap + 2 * (card_w + gap),
            'color': '#4a148c',
            'border': '#ab47bc',
            'icon': '\u26A1',  # lightning
            'title': 'Hybrid',
            'subtitle': 'Best of Both',
            'stats': ['93% Accuracy', 'Skills + Knowledge', 'Production Ready'],
            'stat_color': '#ce93d8'
        }
    ]

    for card in cards:
        # Card background
        draw_rounded_rect(ax, card['x'], card_y, card_w, card_h,
                          card['color'], alpha=0.7, radius=0.015,
                          linewidth=1.2, edgecolor=card['border'])

        # Card title
        ax.text(card['x'] + card_w / 2, card_y + card_h - 0.04,
                card['title'], fontsize=13, fontweight='bold',
                color='#ffffff', ha='center', va='center', zorder=10)

        # Card subtitle
        ax.text(card['x'] + card_w / 2, card_y + card_h - 0.075,
                card['subtitle'], fontsize=9, color=card['stat_color'],
                ha='center', va='center', zorder=10, style='italic')

        # Divider in card
        ax.plot([card['x'] + 0.03, card['x'] + card_w - 0.03],
                [card_y + card_h - 0.095, card_y + card_h - 0.095],
                color=card['border'], alpha=0.4, linewidth=0.8, zorder=10)

        # Stats
        for i, stat in enumerate(card['stats']):
            ax.text(card['x'] + card_w / 2, card_y + card_h - 0.135 - i * 0.035,
                    f"\u2022 {stat}", fontsize=8, color=card['stat_color'],
                    ha='center', va='center', zorder=10)

    # --- Bottom bar ---
    draw_rounded_rect(ax, 0.05, 0.04, 0.90, 0.08, '#0d1440', alpha=0.6, radius=0.015,
                      linewidth=1, edgecolor='#2a4a8a')

    # Tech stack
    techs = "Ollama  \u2022  ChromaDB  \u2022  HuggingFace  \u2022  GPT-4o Judge  \u2022  Streamlit  \u2022  Docker"
    ax.text(0.50, 0.08, techs,
            fontsize=8, color='#6b88a8', ha='center', va='center', zorder=10,
            fontfamily='sans-serif')

    # --- Decorative corner accents ---
    # Top-left
    ax.plot([0.02, 0.02, 0.07], [0.93, 0.97, 0.97], color='#00d4ff', alpha=0.5, linewidth=2, zorder=10)
    # Top-right
    ax.plot([0.93, 0.98, 0.98], [0.97, 0.97, 0.93], color='#00d4ff', alpha=0.5, linewidth=2, zorder=10)
    # Bottom-left
    ax.plot([0.02, 0.02, 0.07], [0.07, 0.03, 0.03], color='#7c4dff', alpha=0.5, linewidth=2, zorder=10)
    # Bottom-right
    ax.plot([0.93, 0.98, 0.98], [0.03, 0.03, 0.07], color='#7c4dff', alpha=0.5, linewidth=2, zorder=10)

    # --- Floating accent shapes ---
    # Small diamond shapes
    for dx, dy, c in [(0.92, 0.88, '#ff6b9d'), (0.08, 0.88, '#00e676'),
                       (0.92, 0.15, '#00d4ff'), (0.08, 0.15, '#ffd166')]:
        diamond = np.array([[dx, dy + 0.012], [dx + 0.008, dy],
                            [dx, dy - 0.012], [dx - 0.008, dy]])
        ax.add_patch(plt.Polygon(diamond, closed=True, facecolor=c, alpha=0.25, zorder=5))

    # --- Save ---
    output_path = Path(__file__).parent / 'project_banner.png'
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.02,
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Banner saved to: {output_path}")
    return output_path


if __name__ == '__main__':
    create_banner()
