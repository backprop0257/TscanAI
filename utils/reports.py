"""CSV and PDF exports of diagnoses (History tab, batch results, single result).

PDFs are built with reportlab from the stored result (the images are the JPEGs the pipeline produced),
so a report shows exactly what the app showed: photo, background removal, Grad-CAM, top-5, advice.
"""
import base64
import csv
import io
import time

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from . import treatment as T

GREEN = colors.HexColor('#15803d')
INK = colors.HexColor('#1f2937')
MUTED = colors.HexColor('#6b7280')
LINE = colors.HexColor('#e5e7eb')
SOFT = colors.HexColor('#f3f4f6')
STATUS_TEXT = {'ok': 'Diagnosed', 'unsure': 'Not sure (low confidence)', 'ood': 'Not accepted'}

CSV_FIELDS = ['date', 'file', 'mode', 'status', 'diagnosis', 'class', 'confidence_pct', 'model', 'models_agree',
              'leaf_focus_score', 'affected_area_pct', 'time_ms', 'message', 'batch_id', 'id']


def _fmt_time(ts):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts or 0))


def _pct(v, nd=1):
    return '' if v is None else f'{float(v) * 100:.{nd}f}'


def csv_bytes(summaries):
    """One row per diagnosis (history summaries, most recent first)."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_FIELDS)
    w.writeheader()
    for s in summaries:
        w.writerow({'date': _fmt_time(s.get('created_at')), 'file': s.get('filename') or '',
                    'mode': s.get('mode') or '', 'status': STATUS_TEXT.get(s.get('status'), s.get('status') or ''),
                    'diagnosis': s.get('title') or '', 'class': s.get('class') or '',
                    'confidence_pct': _pct(s.get('confidence')), 'model': s.get('model') or '',
                    'models_agree': s.get('agree') or '', 'leaf_focus_score': '' if s.get('lfs') is None else f"{s['lfs']:.3f}",
                    'affected_area_pct': _pct(s.get('affected_area')), 'time_ms': s.get('timing_ms') or '',
                    'message': (s.get('message') or '').replace('\n', ' '), 'batch_id': s.get('batch_id') or '',
                    'id': s.get('id')})
    return ('﻿' + buf.getvalue()).encode('utf-8')          # BOM: Excel opens UTF-8 correctly


# ------------------------------------------------------------------------------------------------- PDF
def _styles():
    ss = getSampleStyleSheet()
    return {
        'h1': ParagraphStyle('h1', parent=ss['Heading1'], fontSize=17, textColor=INK, spaceAfter=2),
        'h2': ParagraphStyle('h2', parent=ss['Heading2'], fontSize=12.5, textColor=GREEN, spaceBefore=8, spaceAfter=4),
        'body': ParagraphStyle('body', parent=ss['BodyText'], fontSize=9.5, leading=13, textColor=INK, alignment=TA_LEFT),
        'small': ParagraphStyle('small', parent=ss['BodyText'], fontSize=8, leading=10.5, textColor=MUTED),
        'cell': ParagraphStyle('cell', parent=ss['BodyText'], fontSize=8.5, leading=10.5, textColor=INK),
        'verdict': ParagraphStyle('verdict', parent=ss['Heading1'], fontSize=20, leading=24, textColor=GREEN),
    }


def _esc(s):
    return (str(s if s is not None else '')).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _img(data_url, w_mm, h_mm=None):
    if not data_url or ',' not in str(data_url):
        return None
    try:
        raw = base64.b64decode(data_url.split(',', 1)[1])
        from reportlab.lib.utils import ImageReader
        ir = ImageReader(io.BytesIO(raw))
        iw, ih = ir.getSize()
        w = w_mm * mm
        h = w * ih / max(iw, 1)
        if h_mm and h > h_mm * mm:
            h = h_mm * mm
            w = h * iw / max(ih, 1)
        return Image(io.BytesIO(raw), width=w, height=h)
    except Exception:  # noqa: BLE001 -- a broken image never breaks the report
        return None


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(15 * mm, 10 * mm, 'TomatoLeafAI - decision support only; confirm with an agronomist before treating.')
    canvas.drawRightString(A4[0] - 15 * mm, 10 * mm, f'Page {doc.page}')
    canvas.restoreState()


def _table(rows, widths, header=True, zebra=True):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    st = [('FONT', (0, 0), (-1, -1), 'Helvetica', 8.5), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
          ('LINEBELOW', (0, 0), (-1, -1), 0.4, LINE), ('TOPPADDING', (0, 0), (-1, -1), 3),
          ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]
    if header:
        st += [('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 8.5), ('BACKGROUND', (0, 0), (-1, 0), SOFT)]
    if zebra:
        for i in range(2 if header else 1, len(rows), 2):
            st.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#fafafa')))
    t.setStyle(TableStyle(st))
    return t


def _single_story(item, S):
    r = item.get('result') or {}
    pred = r.get('prediction') or {}
    story = [Paragraph('TomatoLeafAI diagnosis report', S['h1']),
             Paragraph(f"{_esc(item.get('filename') or 'photo')} &nbsp;&middot;&nbsp; {_fmt_time(item.get('created_at'))}"
                       f" &nbsp;&middot;&nbsp; {'all models compared' if item.get('mode') == 'compare' else 'single model'}",
                       S['small']), Spacer(1, 6)]
    accepted = r.get('accepted')
    verdict = r.get('display_name') or ('Not diagnosed' if not accepted else '')
    story.append(Paragraph(_esc(verdict), S['verdict']))
    if accepted:
        cons = r.get('consensus')
        line = (f"Confidence <b>{_pct(r.get('confidence'))}%</b>"
                f"{' (calibrated)' if pred.get('calibrated') else ''} &middot; model <b>{_esc(r.get('model_label'))}</b>")
        if cons:
            line += f" &middot; {cons.get('agree')}/{cons.get('total')} models agree"
        if r.get('affected_area_estimate') is not None:
            line += f" &middot; affected leaf area &asymp; {_pct(r['affected_area_estimate'])}%"
        story.append(Paragraph(line, S['body']))
    if r.get('message'):
        story.append(Paragraph(_esc(r['message']), S['body']))

    prev = r.get('preview') or {}
    ex = (pred.get('explain') or {}).get('images') or {}
    cells, caps = [], []
    for key, cap, src in (('original', 'Photo', prev), ('background_removed', 'Background removed', prev),
                          ('gradcam', 'Grad-CAM', ex), ('disease_regions', 'Disease regions', ex)):
        im = _img(src.get(key), 42, 42)
        if im is not None:
            cells.append(im); caps.append(Paragraph(cap, S['small']))
    if cells:
        story += [Spacer(1, 6), _table([cells, caps], [45 * mm] * len(cells), header=False, zebra=False)]

    if accepted and pred.get('top5'):
        story.append(Paragraph('Top predictions', S['h2']))
        rows = [['#', 'Class', 'Probability']] + [[str(i + 1), t.get('display_name'), f"{t.get('prob', 0) * 100:.1f}%"]
                                                  for i, t in enumerate(pred['top5'])]
        story.append(_table(rows, [10 * mm, 110 * mm, 30 * mm]))
        e = pred.get('explain') or {}
        if e.get('lfs') is not None:
            story.append(Spacer(1, 4))
            story.append(Paragraph(
                f"Leaf-Focus Score {e['lfs']:.2f} (share of the Grad-CAM attention on the leaf)"
                + (f" &middot; Lesion-Focus Score {e['lefs']:.2f}" if e.get('lefs') is not None else ''), S['small']))

    models = r.get('models') or {}
    if models:
        story.append(Paragraph('All models', S['h2']))
        rows = [['Model', 'Diagnosis', 'Confidence', 'Leaf-Focus', 'Time']]
        for m in models.values():
            rows.append([Paragraph(_esc(m.get('label')), S['cell']), Paragraph(_esc(m.get('display_name')), S['cell']),
                         f"{(m.get('confidence') or 0) * 100:.1f}%",
                         '' if (m.get('explain') or {}).get('lfs') is None else f"{m['explain']['lfs']:.2f}",
                         f"{m.get('time_ms', '')} ms"])
        story.append(_table(rows, [50 * mm, 55 * mm, 22 * mm, 20 * mm, 20 * mm]))

    g = r.get('gate') or {}
    if g:
        story.append(Paragraph('Tomato-leaf check', S['h2']))
        story.append(Paragraph(f"Decision <b>{_esc(g.get('decision'))}</b> &middot; gate confidence "
                               f"{(g.get('confidence') or 0):.2f}" + (f" &middot; reason: {_esc(g.get('veto'))}" if g.get('veto') else ''),
                               S['body']))
        for reason in (g.get('reasons') or [])[:4]:
            story.append(Paragraph('&bull; ' + _esc(reason), S['small']))

    tr = r.get('treatment') or {}
    if accepted and tr:
        story.append(Paragraph(f"Management advice &mdash; {_esc(tr.get('disease_name') or verdict)}", S['h2']))
        for key, title in (('symptoms_observed', 'What to look for'), ('cultural_practices', 'Cultural practices'),
                           ('sanitation', 'Sanitation'), ('irrigation_water_management', 'Irrigation and water'),
                           ('monitoring_advice', 'Monitoring'), ('prevention_advice', 'Prevention')):
            items = tr.get(key) or []
            if items:
                block = [Paragraph(f'<b>{title}</b>', S['body'])] + [Paragraph('&bull; ' + _esc(x), S['body']) for x in items]
                story.append(KeepTogether(block))
                story.append(Spacer(1, 3))
    story += [Spacer(1, 8), Paragraph(_esc((tr or {}).get('disclaimer') or T.DISCLAIMER), S['small'])]
    return story


def pdf_single(item):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm,
                            bottomMargin=16 * mm, title='TomatoLeafAI diagnosis report', author='TomatoLeafAI')
    doc.build(_single_story(item, _styles()), onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


def pdf_batch(items, title='TomatoLeafAI batch report', detail_pages=True):
    """Summary table (thumbnail, file, diagnosis, confidence, status) + optionally one page per photo."""
    S = _styles()
    story = [Paragraph(_esc(title), S['h1']),
             Paragraph(f'{len(items)} photo(s) &middot; generated {_fmt_time(time.time())}', S['small']), Spacer(1, 8)]
    counts = {}
    for it in items:
        st = _status(it)
        counts[st] = counts.get(st, 0) + 1
    story.append(Paragraph(' &middot; '.join(f'{STATUS_TEXT.get(k, k)}: <b>{v}</b>' for k, v in counts.items()), S['body']))
    story.append(Spacer(1, 6))
    rows = [['', 'File', 'Diagnosis', 'Confidence', 'Status']]
    for it in items:
        r = it.get('result') or {}
        thumb = _img((r.get('preview') or {}).get('background_removed'), 14, 14) or ''
        rows.append([thumb, Paragraph(_esc(it.get('filename')), S['cell']), Paragraph(_esc(r.get('display_name')), S['cell']),
                     '' if r.get('confidence') is None else f"{r['confidence'] * 100:.1f}%",
                     Paragraph(STATUS_TEXT.get(_status(it), ''), S['cell'])])
    story.append(_table(rows, [17 * mm, 52 * mm, 52 * mm, 22 * mm, 37 * mm]))
    if detail_pages:
        for it in items:
            story.append(PageBreak())
            story += _single_story(it, S)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm,
                            bottomMargin=16 * mm, title=title, author='TomatoLeafAI')
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


def _status(item):
    r = item.get('result') or {}
    if not r.get('accepted'):
        return 'ood'
    return 'ok' if r.get('confident') else 'unsure'
