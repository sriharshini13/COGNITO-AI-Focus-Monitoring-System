import json
import os
import webbrowser
from datetime import datetime
from modules.analytics_engine import AnalyticsEngine
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

engine = AnalyticsEngine()
report = engine.full_report()
rec    = report['recommendations']
ana    = report['analytics']

# ════════════════════════════════════════════════════════════════
#                    TERMINAL REPORT
# ════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("         COGNITO — FULL ANALYTICS REPORT")
print("="*55)

# Personal bests
bests = ana['bests']
if bests:
    print(f"\n🏆 PERSONAL BESTS")
    print(f"  Best Session    : {bests['best_session']['date']} "
          f"— {bests['best_session']['score']}/100 "
          f"(Grade {bests['best_session']['grade']})")
    print(f"  Longest Session : {bests['longest_session']['date']} "
          f"— {bests['longest_session']['mins']} mins")
    print(f"  Current Streak  : {bests['current_streak']} days 🔥")
    print(f"  Total Sessions  : {bests['total_sessions']}")
    print(f"  Total Study Hrs : {bests['total_study_hrs']} hrs")

# Daily summary
daily = ana['daily']
print(f"\n📅 DAILY SUMMARY — {daily['date']}")
if daily['today']:
    t = daily['today']
    print(f"  Today    → {t['sessions']} sessions | "
          f"Avg Score: {t['avg_prod']} | "
          f"Focus: {t['avg_focus']}% | "
          f"Dist: {t['avg_dist']}")
else:
    print("  Today    → No sessions yet")
if daily['yesterday']:
    y = daily['yesterday']
    print(f"  Yesterday → {y['sessions']} sessions | "
          f"Avg Score: {y['avg_prod']} | "
          f"Focus: {y['avg_focus']}% | "
          f"Dist: {y['avg_dist']}")

# Monthly
monthly = ana['monthly']
if monthly:
    print(f"\n📆 MONTHLY REPORT — {monthly['month']}")
    print(f"  Total Sessions  : {monthly['total_sessions']}")
    print(f"  Avg Productivity: {monthly['avg_prod']}/100")
    print(f"  Total Study Hrs : {monthly['total_study_hrs']} hrs")
    print(f"  Best Grade      : {monthly['best_grade']}")

# Best vs Worst
bvw = ana['best_vs_worst']
if bvw:
    print(f"\n⚡ BEST vs WORST SESSION")
    b = bvw['best'];  w = bvw['worst']
    print(f"  {'Metric':<22} {'BEST':>10} {'WORST':>10}")
    print(f"  {'-'*42}")
    print(f"  {'Date':<22} {b['date']:>10} {w['date']:>10}")
    print(f"  {'Productivity':<22} {b['productivity']:>10} "
          f"{w['productivity']:>10}")
    print(f"  {'Focus High %':<22} {b['focus_high']:>10} "
          f"{w['focus_high']:>10}")
    print(f"  {'Engaged %':<22} {b['engaged']:>10} "
          f"{w['engaged']:>10}")
    print(f"  {'Distractions':<22} {b['distractions']:>10} "
          f"{w['distractions']:>10}")
    print(f"  {'Duration (mins)':<22} {b['duration_mins']:>10} "
          f"{w['duration_mins']:>10}")
    print(f"  Gap between best & worst: {bvw['difference']} pts")

# Recommendations
print(f"\n💡 RECOMMENDATIONS")
bst = rec['best_study_time']
if bst:
    print(f"  Best Study Time : {bst['best_hour_str']} "
          f"(avg {bst['avg_score']}/100)")

brk = rec['break_interval']
print(f"  Break Every     : {brk['recommended_mins']} mins — "
      f"{brk['tip']}")

dem = rec['distraction_emotion']
print(f"  Watch Out For   : {dem['worst_emotion']} — "
      f"{dem['tip']}")

pred = rec['predicted_today']
print(f"  Today's Prediction: {pred['predicted']}/100 "
      f"({pred['confidence']} confidence)")

tips = rec['weekly_tips']
print(f"\n📈 WEEKLY TIPS")
for tip in tips['tips']:
    print(f"  • {tip}")

# ════════════════════════════════════════════════════════════════
#                    MATPLOTLIB CHARTS
# ════════════════════════════════════════════════════════════════
BG    = '#1e1e2e'; CARD  = '#313244'
ACCENT= '#89b4fa'; GREEN = '#a6e3a1'
ORANGE= '#fab387'; RED   = '#f38ba8'
WHITE = '#cdd6f4'; PURPLE= '#cba6f7'

fig = plt.figure(figsize=(18,12), facecolor=BG)
fig.suptitle('COGNITO — Analytics & Recommendations Dashboard',
             color=WHITE, fontsize=16, fontweight='bold')
gs  = gridspec.GridSpec(3, 3, figure=fig,
                        hspace=0.55, wspace=0.4)

# ── 1. Weekly productivity trend ────────────────────────────────
ax1 = fig.add_subplot(gs[0,:2])
ax1.set_facecolor(CARD)
weekly = ana['weekly']
dates  = list(weekly.keys())
prods  = [weekly[d]['avg_prod'] if weekly[d] else 0
          for d in dates]
colors = [GREEN if p>=70 else ORANGE if p>=50 else RED
          for p in prods]
bars   = ax1.bar(range(len(dates)), prods,
                 color=colors, alpha=0.85)
ax1.set_xticks(range(len(dates)))
ax1.set_xticklabels(
    [d[5:] for d in dates], color=WHITE, fontsize=8)
ax1.set_ylim(0,110)
ax1.set_ylabel('Productivity', color=WHITE)
ax1.set_title('7-Day Productivity Trend',
              color=WHITE, fontsize=11)
ax1.tick_params(colors=WHITE)
ax1.axhline(70, color=GREEN, linestyle='--',
            alpha=0.5, linewidth=1)
for bar, val in zip(bars, prods):
    if val > 0:
        ax1.text(bar.get_x()+bar.get_width()/2,
                 val+1, str(val),
                 ha='center', color=WHITE, fontsize=8)

# ── 2. Best study hours ──────────────────────────────────────────
ax2 = fig.add_subplot(gs[0,2])
ax2.set_facecolor(CARD)
if bst and bst['all_hours']:
    hours  = sorted(bst['all_hours'].keys())
    scores = [bst['all_hours'][h] for h in hours]
    hcols  = [ACCENT if h==bst['best_hour']
              else '#45475a' for h in hours]
    ax2.barh([f"{h:02d}:00" for h in hours],
             scores, color=hcols)
    ax2.set_xlabel('Avg Score', color=WHITE, fontsize=8)
    ax2.set_title('Best Study Hours',
                  color=WHITE, fontsize=11)
    ax2.tick_params(colors=WHITE)
    ax2.set_xlim(0,105)

# ── 3. Best vs Worst comparison ──────────────────────────────────
ax3 = fig.add_subplot(gs[1,:2])
ax3.set_facecolor(CARD)
if bvw:
    metrics   = ['Productivity','Focus High%',
                 'Engaged%','Distractions']
    best_vals = [bvw['best']['productivity'],
                 bvw['best']['focus_high'],
                 bvw['best']['engaged'],
                 bvw['best']['distractions']]
    worst_vals= [bvw['worst']['productivity'],
                 bvw['worst']['focus_high'],
                 bvw['worst']['engaged'],
                 bvw['worst']['distractions']]
    x = np.arange(len(metrics))
    ax3.bar(x-0.2, best_vals,  0.4,
            label='Best',  color=GREEN,  alpha=0.85)
    ax3.bar(x+0.2, worst_vals, 0.4,
            label='Worst', color=RED,    alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, color=WHITE, fontsize=9)
    ax3.set_title('Best vs Worst Session',
                  color=WHITE, fontsize=11)
    ax3.tick_params(colors=WHITE)
    ax3.legend(labelcolor=WHITE, facecolor=CARD)

# ── 4. Emotion distraction impact ───────────────────────────────
ax4 = fig.add_subplot(gs[1,2])
ax4.set_facecolor(CARD)
dem    = rec['distraction_emotion']
emots  = list(dem['impact_scores'].keys())
impacts= list(dem['impact_scores'].values())
ecols  = [RED if e==dem['worst_emotion']
          else '#45475a' for e in emots]
ax4.bar(emots, impacts, color=ecols, alpha=0.85)
ax4.set_title('Emotion → Distraction Impact',
              color=WHITE, fontsize=11)
ax4.set_ylabel('Impact Score', color=WHITE, fontsize=8)
ax4.tick_params(colors=WHITE)

# ── 5. Monthly weekly breakdown ─────────────────────────────────
ax5 = fig.add_subplot(gs[2,:2])
ax5.set_facecolor(CARD)
if monthly and monthly['weekly_breakdown']:
    wb    = monthly['weekly_breakdown']
    wkeys = list(wb.keys())
    wvals = [wb[k]['avg_prod'] for k in wkeys]
    wsess = [wb[k]['sessions'] for k in wkeys]
    ax5.plot(wkeys, wvals, color=ACCENT,
             linewidth=2, marker='o', markersize=8)
    ax5.fill_between(range(len(wkeys)),
                     wvals, alpha=0.2, color=ACCENT)
    ax5.set_xticks(range(len(wkeys)))
    ax5.set_xticklabels(wkeys, color=WHITE, fontsize=9)
    ax5.set_ylim(0,110)
    ax5.set_title(f"Monthly Breakdown — "
                  f"{monthly['month']}",
                  color=WHITE, fontsize=11)
    ax5.tick_params(colors=WHITE)
    for i,(k,v) in enumerate(zip(wkeys,wvals)):
        ax5.annotate(f'{v}\n({wsess[i]} sess)',
                     (i,v), textcoords="offset points",
                     xytext=(0,10), ha='center',
                     color=WHITE, fontsize=8)

# ── 6. Personal bests card ───────────────────────────────────────
ax6 = fig.add_subplot(gs[2,2])
ax6.set_facecolor(BG)
ax6.axis('off')
if bests:
    cards = [
        ('🏆 Best Score',
         f"{bests['best_session']['score']}/100", GREEN),
        ('🔥 Streak',
         f"{bests['current_streak']} days", ORANGE),
        ('📚 Total Sessions',
         str(bests['total_sessions']),   ACCENT),
        ('⏱️ Study Hours',
         f"{bests['total_study_hrs']}h", PURPLE),
    ]
    import matplotlib.patches as mpatches
    for i,(label,value,color) in enumerate(cards):
        y = 0.82 - i*0.22
        ax6.add_patch(mpatches.FancyBboxPatch(
            (0.05, y-0.08), 0.9, 0.18,
            boxstyle="round,pad=0.02",
            facecolor=CARD, edgecolor=color,
            linewidth=1.5,
            transform=ax6.transAxes))
        ax6.text(0.5, y+0.04, value,
                 transform=ax6.transAxes,
                 fontsize=13, color=color,
                 ha='center', va='center',
                 fontweight='bold')
        ax6.text(0.5, y-0.04, label,
                 transform=ax6.transAxes,
                 fontsize=8, color=WHITE,
                 ha='center', va='center')
ax6.set_title('Personal Bests',
              color=WHITE, fontsize=11)

os.makedirs('reports', exist_ok=True)
chart_path = 'reports/analytics_charts.png'
plt.savefig(chart_path, dpi=120,
            bbox_inches='tight', facecolor=BG)
print(f"\nCharts saved to {chart_path}")
plt.show()

# ════════════════════════════════════════════════════════════════
#                    HTML REPORT
# ════════════════════════════════════════════════════════════════
weekly_rows = ""
for date, data in weekly.items():
    if data:
        weekly_rows += f"""
        <tr>
            <td>{date}</td>
            <td>{data['sessions']}</td>
            <td>{data['avg_prod']}</td>
            <td>{data['avg_focus']}%</td>
            <td>{data['avg_dist']}</td>
        </tr>"""

tips_html = "".join(
    f"<li>{t}</li>"
    for t in rec['weekly_tips']['tips'])

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset='UTF-8'>
<title>COGNITO Analytics Report</title>
<style>
  body {{
    font-family: 'Segoe UI', sans-serif;
    background: #1e1e2e; color: #cdd6f4;
    margin: 0; padding: 20px;
  }}
  h1 {{ color: #89b4fa; text-align: center; }}
  h2 {{ color: #89b4fa; border-bottom: 1px solid #313244;
        padding-bottom: 6px; }}
  .grid {{ display: grid;
           grid-template-columns: repeat(4, 1fr);
           gap: 16px; margin: 20px 0; }}
  .card {{
    background: #313244; border-radius: 10px;
    padding: 16px; text-align: center;
    border: 1px solid #45475a;
  }}
  .card .value {{
    font-size: 28px; font-weight: bold;
    color: #a6e3a1;
  }}
  .card .label {{
    font-size: 12px; color: #9399b2;
    margin-top: 4px;
  }}
  table {{
    width: 100%; border-collapse: collapse;
    margin: 12px 0;
  }}
  th {{
    background: #89b4fa; color: #1e1e2e;
    padding: 10px; text-align: left;
  }}
  td {{
    padding: 9px; border-bottom: 1px solid #313244;
  }}
  tr:nth-child(even) {{ background: #313244; }}
  .tip {{
    background: #313244; border-left: 4px solid #89b4fa;
    padding: 10px 14px; margin: 8px 0;
    border-radius: 0 8px 8px 0;
  }}
  .rec-card {{
    background: #313244; border-radius: 10px;
    padding: 14px; margin: 10px 0;
    border: 1px solid #45475a;
  }}
  .rec-card h3 {{ color: #cba6f7; margin: 0 0 6px; }}
  img {{ width: 100%; border-radius: 10px;
         margin: 16px 0; }}
  .badge {{
    display: inline-block; padding: 3px 10px;
    border-radius: 20px; font-size: 12px;
    font-weight: bold;
  }}
  .green  {{ background:#a6e3a1; color:#1e1e2e; }}
  .orange {{ background:#fab387; color:#1e1e2e; }}
  .red    {{ background:#f38ba8; color:#1e1e2e; }}
</style>
</head>
<body>
<h1>🧠 COGNITO Analytics Report</h1>
<p style='text-align:center; color:#9399b2'>
  Generated: {datetime.now().strftime('%d %B %Y  %H:%M')}
</p>

<h2>🏆 Personal Bests</h2>
<div class='grid'>
  <div class='card'>
    <div class='value'>{bests['best_session']['score'] if bests else 'N/A'}</div>
    <div class='label'>Best Session Score</div>
  </div>
  <div class='card'>
    <div class='value'>{bests['current_streak'] if bests else 0}🔥</div>
    <div class='label'>Current Streak (days)</div>
  </div>
  <div class='card'>
    <div class='value'>{bests['total_sessions'] if bests else 0}</div>
    <div class='label'>Total Sessions</div>
  </div>
  <div class='card'>
    <div class='value'>{bests['total_study_hrs'] if bests else 0}h</div>
    <div class='label'>Total Study Hours</div>
  </div>
</div>

<h2>💡 Recommendations</h2>
<div class='rec-card'>
  <h3>⏰ Best Time to Study</h3>
  <p>{bst['best_hour_str'] if bst else 'N/A'} —
     avg score {bst['avg_score'] if bst else 'N/A'}/100</p>
</div>
<div class='rec-card'>
  <h3>☕ Recommended Break Interval</h3>
  <p>Every <b>{brk['recommended_mins']} minutes</b>
     — {brk['tip']}</p>
</div>
<div class='rec-card'>
  <h3>⚠️ Biggest Distraction Trigger</h3>
  <p><b>{dem['worst_emotion']}</b> — {dem['tip']}</p>
</div>
<div class='rec-card'>
  <h3>🔮 Today's Predicted Productivity</h3>
  <p><b>{pred['predicted']}/100</b>
     ({pred['confidence']} confidence)
     — {pred['tip']}</p>
</div>

<h2>📈 Weekly Tips</h2>
<ul>{tips_html}</ul>

<h2>📅 7-Day Trend</h2>
<table>
  <tr>
    <th>Date</th><th>Sessions</th>
    <th>Avg Score</th><th>Focus High</th>
    <th>Distractions</th>
  </tr>
  {weekly_rows}
</table>

<h2>📊 Analytics Charts</h2>
<img src='../reports/analytics_charts.png'
     alt='Analytics Charts'>

<p style='text-align:center; color:#6c6f85; margin-top:30px'>
  COGNITO — Vision-Based Focus Analysis System
</p>
</body>
</html>"""

html_path = 'reports/analytics_report.html'
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"HTML report saved to {html_path}")
webbrowser.open(f"file:///{os.path.abspath(html_path)}")
print("Report opened in browser!")