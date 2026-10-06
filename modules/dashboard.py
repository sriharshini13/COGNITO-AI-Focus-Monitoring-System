import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
from modules.session_logger import SessionLogger
from datetime import datetime, timedelta

# ── Color palette ────────────────────────────────────────────────
BG       = '#1e1e2e'
CARD     = '#313244'
ACCENT   = '#89b4fa'
GREEN    = '#a6e3a1'
ORANGE   = '#fab387'
RED      = '#f38ba8'
WHITE    = '#cdd6f4'
PURPLE   = '#cba6f7'

def show_dashboard(mode='week'):
    """
    mode: 'today' or 'week'
    """
    logger   = SessionLogger()
    sessions = logger.get_today() if mode == 'today' \
               else logger.get_this_week()
    all_time = logger.get_all_sessions()

    if not sessions:
        print(f"No sessions found for {mode}!")
        return

    summary = logger.get_stats_summary(sessions)

    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    fig.suptitle(
        f"COGNITO Dashboard — "
        f"{'Today' if mode=='today' else 'Last 7 Days'}  "
        f"({len(sessions)} sessions)",
        color=WHITE, fontsize=16, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.55, wspace=0.4)

    # ── 1. Productivity trend line ───────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor(CARD)
    labels = [f"{s['date']}\n{s['time'][:5]}" for s in sessions]
    prod   = [s['avg_productivity'] for s in sessions]
    x      = range(len(prod))
    ax1.plot(x, prod, color=ACCENT, linewidth=2, marker='o',
             markersize=6)
    ax1.fill_between(x, prod, alpha=0.2, color=ACCENT)
    ax1.axhline(y=70, color=GREEN,  linestyle='--',
                linewidth=1, alpha=0.7, label='Good (70)')
    ax1.axhline(y=40, color=RED,    linestyle='--',
                linewidth=1, alpha=0.7, label='Low (40)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, color=WHITE, fontsize=7)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('Score', color=WHITE, fontsize=9)
    ax1.set_title('Productivity Score Trend',
                  color=WHITE, fontsize=11)
    ax1.tick_params(colors=WHITE)
    ax1.legend(fontsize=8, labelcolor=WHITE,
               facecolor=CARD, edgecolor=ACCENT)
    for xi, yi in zip(x, prod):
        ax1.annotate(f'{yi}', (xi, yi),
                     textcoords="offset points",
                     xytext=(0, 8), ha='center',
                     color=WHITE, fontsize=8)

    # ── 2. Summary stat cards ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(BG)
    ax2.axis('off')

    cards = [
        ('Sessions',       str(summary['total_sessions']),      ACCENT),
        ('Avg Score',      f"{summary['avg_productivity']}/100", GREEN),
        ('Avg Distractions', str(summary['avg_distractions']),  ORANGE),
        ('Avg High Focus', f"{summary['avg_high_focus_pct']}%", PURPLE),
    ]
    for i, (label, value, color) in enumerate(cards):
        y_pos = 0.85 - i * 0.22
        ax2.add_patch(mpatches.FancyBboxPatch(
            (0.05, y_pos - 0.08), 0.9, 0.18,
            boxstyle="round,pad=0.02",
            facecolor=CARD, edgecolor=color, linewidth=1.5,
            transform=ax2.transAxes))
        ax2.text(0.5, y_pos + 0.04, value,
                 transform=ax2.transAxes,
                 fontsize=14, color=color,
                 ha='center', va='center',
                 fontweight='bold')
        ax2.text(0.5, y_pos - 0.04, label,
                 transform=ax2.transAxes,
                 fontsize=8, color=WHITE,
                 ha='center', va='center')
    ax2.set_title('Quick Stats', color=WHITE, fontsize=11)

    # ── 3. Focus distribution stacked bar ───────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.set_facecolor(CARD)
    highs   = [s['focus_high_pct']   for s in sessions]
    mediums = [s['focus_medium_pct'] for s in sessions]
    lows    = [s['focus_low_pct']    for s in sessions]
    x       = range(len(sessions))
    ax3.bar(x, highs,   color=GREEN,  label='High',   alpha=0.9)
    ax3.bar(x, mediums, color=ORANGE, label='Medium',
            bottom=highs, alpha=0.9)
    ax3.bar(x, lows,    color=RED,    label='Low',
            bottom=[h+m for h,m in zip(highs,mediums)], alpha=0.9)
    ax3.set_xticks(x)
    ax3.set_xticklabels(
        [s['time'][:5] for s in sessions],
        color=WHITE, fontsize=8)
    ax3.set_ylabel('%', color=WHITE, fontsize=9)
    ax3.set_title('Focus Distribution Per Session',
                  color=WHITE, fontsize=11)
    ax3.tick_params(colors=WHITE)
    ax3.legend(fontsize=8, labelcolor=WHITE,
               facecolor=CARD, edgecolor=ACCENT,
               loc='upper right')
    ax3.set_ylim(0, 110)

    # ── 4. Emotion pie (aggregated) ──────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor(BG)
    e_keys   = ['Engaged','Bored','Confused','Frustrated']
    e_colors = [GREEN, ACCENT, ORANGE, RED]
    e_avgs   = [
        sum(s[f'emotion_{k.lower()}_pct'] for s in sessions)/len(sessions)
        for k in e_keys
    ]
    non_zero = [(v,l,c) for v,l,c in zip(e_avgs,e_keys,e_colors) if v>0]
    if non_zero:
        vals, lbls, cols = zip(*non_zero)
        ax4.pie(vals, labels=lbls, colors=cols,
                autopct='%1.1f%%',
                textprops={'color': WHITE, 'fontsize': 8})
    ax4.set_title('Avg Emotion Mix', color=WHITE, fontsize=11)

    # ── 5. Distraction trend ─────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    ax5.set_facecolor(CARD)
    distracts = [s['total_distractions'] for s in sessions]
    x         = range(len(sessions))
    bars      = ax5.bar(x, distracts, color=ORANGE, alpha=0.85)
    ax5.set_xticks(x)
    ax5.set_xticklabels(
        [s['time'][:5] for s in sessions],
        color=WHITE, fontsize=8)
    ax5.set_ylabel('Count', color=WHITE, fontsize=9)
    ax5.set_title('Distractions Per Session',
                  color=WHITE, fontsize=11)
    ax5.tick_params(colors=WHITE)
    for bar, val in zip(bars, distracts):
        ax5.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+0.1,
                 str(val), ha='center',
                 color=WHITE, fontsize=9)

    # ── 6. All time grades ───────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.set_facecolor(CARD)
    grade_counts = {'A':0,'B':0,'C':0,'D':0,'F':0}
    for s in all_time:
        g = s.get('grade','F')
        if g in grade_counts:
            grade_counts[g] += 1
    g_colors = [GREEN, ACCENT, ORANGE, RED, '#6c6f85']
    g_vals   = list(grade_counts.values())
    g_lbls   = list(grade_counts.keys())
    non_zero = [(v,l,c) for v,l,c
                in zip(g_vals,g_lbls,g_colors) if v>0]
    if non_zero:
        vals, lbls, cols = zip(*non_zero)
        ax6.pie(vals, labels=lbls, colors=cols,
                autopct='%1.0f%%',
                textprops={'color': WHITE, 'fontsize': 9})
    ax6.set_title('All-Time Grade Distribution',
                  color=WHITE, fontsize=11)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path      = f'session_logs/dashboard_{mode}_{timestamp}.png'
    plt.savefig(path, dpi=120,
                bbox_inches='tight', facecolor=BG)
    print(f"Dashboard saved to {path}")
    plt.show()