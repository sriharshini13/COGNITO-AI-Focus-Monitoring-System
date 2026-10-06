from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)
from reportlab.lib.units import cm
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics import renderPDF
import os
from datetime import datetime

# ── Color palette ────────────────────────────────────────────────
DARK_BG    = colors.HexColor('#1e1e2e')
ACCENT     = colors.HexColor('#89b4fa')
GREEN      = colors.HexColor('#a6e3a1')
ORANGE     = colors.HexColor('#fab387')
RED        = colors.HexColor('#f38ba8')
WHITE      = colors.white
LIGHT_GRAY = colors.HexColor('#cdd6f4')
CARD_BG    = colors.HexColor('#313244')

def _bar_chart(values, labels, bar_colors, width=400, height=120):
    """Draw a simple bar chart using ReportLab shapes."""
    d      = Drawing(width, height)
    n      = len(values)
    max_v  = max(values) if max(values) > 0 else 1
    bw     = (width - 60) / n
    gap    = bw * 0.2

    for i, (val, label, col) in enumerate(zip(values, labels, bar_colors)):
        bh = int((val / max_v) * (height - 30))
        bx = 30 + i * bw + gap / 2
        by = 20

        # Bar
        r = Rect(bx, by, bw - gap, bh,
                 fillColor=col, strokeColor=None)
        d.add(r)

        # Value label on top
        d.add(String(bx + (bw - gap) / 2, by + bh + 4,
                     f'{val:.1f}%',
                     fontSize=7, fillColor=WHITE,
                     textAnchor='middle'))

        # X label
        d.add(String(bx + (bw - gap) / 2, 4,
                     label[:8],
                     fontSize=7, fillColor=LIGHT_GRAY,
                     textAnchor='middle'))
    return d

def generate_pdf(focus_stats, emotion_stats, productivity_summary,
                 distraction_summary, session_secs,
                 output_path='session_logs/session_report.pdf'):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc  = SimpleDocTemplate(output_path, pagesize=A4,
                              leftMargin=1.5*cm, rightMargin=1.5*cm,
                              topMargin=1.5*cm, bottomMargin=1.5*cm)
    story = []
    styles = getSampleStyleSheet()

    # ── Custom styles ────────────────────────────────────────────
    title_style = ParagraphStyle('Title',
        fontSize=22, textColor=ACCENT,
        spaceAfter=4, fontName='Helvetica-Bold')

    subtitle_style = ParagraphStyle('Subtitle',
        fontSize=10, textColor=LIGHT_GRAY,
        spaceAfter=12)

    section_style = ParagraphStyle('Section',
        fontSize=13, textColor=ACCENT,
        spaceBefore=14, spaceAfter=6,
        fontName='Helvetica-Bold')

    body_style = ParagraphStyle('Body',
        fontSize=10, textColor=WHITE,
        spaceAfter=4, leading=16)

    # ── Header ───────────────────────────────────────────────────
    mins, secs = divmod(int(session_secs), 60)
    now        = datetime.now().strftime("%d %B %Y  %H:%M")

    story.append(Paragraph("COGNITO", title_style))
    story.append(Paragraph(
        "Vision-Based Focus & Emotion Analysis System", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=ACCENT, spaceAfter=10))
    story.append(Paragraph(
        f"Session Date: {now}   |   Duration: {mins:02d}:{secs:02d}",
        body_style))
    story.append(Spacer(1, 0.3*cm))

    # ── Productivity Score Card ──────────────────────────────────
    story.append(Paragraph("Productivity Score", section_style))
    grade       = productivity_summary.get('grade', 'N/A')
    grade_label = productivity_summary.get('grade_label', '')
    avg_score   = productivity_summary.get('average_score', 0)
    peak_score  = productivity_summary.get('peak_score', 0)

    grade_color = GREEN  if avg_score >= 70 else \
                  ORANGE if avg_score >= 40 else RED

    prod_data = [
        ['Metric', 'Value'],
        ['Average Productivity Score', f'{avg_score} / 100'],
        ['Peak Score',                 f'{peak_score} / 100'],
        ['Grade',                      f'{grade} — {grade_label}'],
    ]
    prod_table = Table(prod_data, colWidths=[9*cm, 8*cm])
    prod_table.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,0),  ACCENT),
        ('TEXTCOLOR',   (0,0), (-1,0),  DARK_BG),
        ('FONTNAME',    (0,0), (-1,0),  'Helvetica-Bold'),
        ('BACKGROUND',  (0,1), (-1,-1), CARD_BG),
        ('TEXTCOLOR',   (0,1), (-1,-1), WHITE),
        ('FONTSIZE',    (0,0), (-1,-1), 10),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [CARD_BG, colors.HexColor('#3b3b52')]),
        ('GRID',        (0,0), (-1,-1), 0.5, colors.HexColor('#45475a')),
        ('PADDING',     (0,0), (-1,-1), 8),
        ('ALIGN',       (1,0), (1,-1),  'CENTER'),
    ]))
    story.append(prod_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Focus Distribution ───────────────────────────────────────
    story.append(Paragraph("Focus Distribution", section_style))
    total_f = sum(focus_stats.values()) or 1
    f_data  = [
        ['Focus Level', 'Frames', 'Percentage'],
        ['High Focus',   str(focus_stats.get('High',0)),
         f"{focus_stats.get('High',0)/total_f*100:.1f}%"],
        ['Medium Focus', str(focus_stats.get('Medium',0)),
         f"{focus_stats.get('Medium',0)/total_f*100:.1f}%"],
        ['Low Focus',    str(focus_stats.get('Low',0)),
         f"{focus_stats.get('Low',0)/total_f*100:.1f}%"],
    ]
    f_table = Table(f_data, colWidths=[6*cm, 5*cm, 6*cm])
    f_table.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,0),  ACCENT),
        ('TEXTCOLOR',   (0,0), (-1,0),  DARK_BG),
        ('FONTNAME',    (0,0), (-1,0),  'Helvetica-Bold'),
        ('BACKGROUND',  (0,1), (-1,-1), CARD_BG),
        ('TEXTCOLOR',   (0,1), (0,-1),  WHITE),
        ('TEXTCOLOR',   (1,1), (-1,-1), LIGHT_GRAY),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [CARD_BG, colors.HexColor('#3b3b52')]),
        ('GRID',        (0,0), (-1,-1), 0.5, colors.HexColor('#45475a')),
        ('PADDING',     (0,0), (-1,-1), 8),
        ('ALIGN',       (1,0), (-1,-1), 'CENTER'),
    ]))
    story.append(f_table)
    story.append(Spacer(1, 0.3*cm))

    # Focus bar chart
    f_pcts   = [focus_stats.get(k,0)/total_f*100
                for k in ['High','Medium','Low']]
    f_chart  = _bar_chart(f_pcts, ['High','Medium','Low'],
                          [GREEN, ORANGE, RED])
    story.append(f_chart)
    story.append(Spacer(1, 0.4*cm))

    # ── Emotion Distribution ─────────────────────────────────────
    story.append(Paragraph("Emotion Distribution", section_style))
    total_e   = sum(emotion_stats.values()) or 1
    e_keys    = ['Engaged','Bored','Confused','Frustrated']
    e_colors_map = {
        'Engaged':    GREEN,
        'Bored':      colors.HexColor('#89b4fa'),
        'Confused':   ORANGE,
        'Frustrated': RED
    }
    e_data = [['Emotion', 'Frames', 'Percentage']]
    for k in e_keys:
        e_data.append([k, str(emotion_stats.get(k,0)),
                       f"{emotion_stats.get(k,0)/total_e*100:.1f}%"])

    e_table = Table(e_data, colWidths=[6*cm, 5*cm, 6*cm])
    e_table.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,0),  ACCENT),
        ('TEXTCOLOR',   (0,0), (-1,0),  DARK_BG),
        ('FONTNAME',    (0,0), (-1,0),  'Helvetica-Bold'),
        ('BACKGROUND',  (0,1), (-1,-1), CARD_BG),
        ('TEXTCOLOR',   (0,1), (0,-1),  WHITE),
        ('TEXTCOLOR',   (1,1), (-1,-1), LIGHT_GRAY),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [CARD_BG, colors.HexColor('#3b3b52')]),
        ('GRID',        (0,0), (-1,-1), 0.5, colors.HexColor('#45475a')),
        ('PADDING',     (0,0), (-1,-1), 8),
        ('ALIGN',       (1,0), (-1,-1), 'CENTER'),
    ]))
    story.append(e_table)
    story.append(Spacer(1, 0.3*cm))

    # Emotion bar chart
    e_pcts   = [emotion_stats.get(k,0)/total_e*100 for k in e_keys]
    e_cols   = [e_colors_map[k] for k in e_keys]
    e_chart  = _bar_chart(e_pcts, e_keys, e_cols)
    story.append(e_chart)
    story.append(Spacer(1, 0.4*cm))

    # ── Distraction Summary ──────────────────────────────────────
    story.append(Paragraph("Distraction Analysis", section_style))
    d_data = [
        ['Metric', 'Value'],
        ['Total Distractions',
         str(distraction_summary.get('total_distractions', 0))],
        ['Distractions Per Hour',
         str(distraction_summary.get('distractions_per_hour', 0))],
    ]
    d_table = Table(d_data, colWidths=[9*cm, 8*cm])
    d_table.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,0),  ACCENT),
        ('TEXTCOLOR',   (0,0), (-1,0),  DARK_BG),
        ('FONTNAME',    (0,0), (-1,0),  'Helvetica-Bold'),
        ('BACKGROUND',  (0,1), (-1,-1), CARD_BG),
        ('TEXTCOLOR',   (0,1), (-1,-1), WHITE),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [CARD_BG, colors.HexColor('#3b3b52')]),
        ('GRID',        (0,0), (-1,-1), 0.5, colors.HexColor('#45475a')),
        ('PADDING',     (0,0), (-1,-1), 8),
        ('ALIGN',       (1,0), (1,-1),  'CENTER'),
    ]))
    story.append(d_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Footer ───────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1,
                             color=ACCENT, spaceBefore=10))
    story.append(Paragraph(
        "Generated by COGNITO — Vision-Based Focus Analysis System",
        ParagraphStyle('Footer', fontSize=8,
                       textColor=LIGHT_GRAY, alignment=1)))

    # ── Build PDF ────────────────────────────────────────────────
    doc.build(story)
    print(f"PDF report saved to {output_path}")
    return output_path