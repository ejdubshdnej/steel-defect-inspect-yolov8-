"""把 docs/PAPER.md 转成中文学术论文格式的 DOCX。

用法：
    python scripts/md2docx.py

产物：docs/工业表面缺陷检测_技术报告.docx

支持的 Markdown 子集（够用于 PAPER.md，不是通用转换器）：
标题（#/##/###/####）、正文段落（连续行合并）、表格、代码块、图片、
有序/无序列表、摘要与关键词、参考文献条目、分隔线。

格式约定：
- 标题：黑体 小二(18pt) 居中
- 一级节标题：黑体 四号(14pt)
- 二级节标题：黑体 小四(12pt)
- 正文：宋体/Times New Roman 小四(12pt)，行距 1.5，首行缩进 2 字符
- 摘要/关键词：楷体 小四
- 表格：五号(10.5pt)，表题在表上方居中
- 图：居中，图题在图下方居中，五号
- 代码块：Consolas 小五(9pt)，浅灰底
- 参考文献：五号，悬挂缩进

依赖 python-docx。图片路径先按 PAPER.md 所在目录（`docs/`）解析，取不到再退回
项目根目录，所以 `![](fail_1_rolled_pit.jpg)` 与 `![](docs/fail_1_rolled_pit.jpg)`
两种写法都能取到图；都找不到时打印「图片缺失」并跳过，不会中止整份文档的生成。
"""
import io
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "docs", "PAPER.md")
OUT = os.path.join(BASE, "docs", "工业表面缺陷检测_技术报告.docx")

SONG, HEI, KAI, MONO = "宋体", "黑体", "楷体", "Consolas"
LATIN = "Times New Roman"


def set_run_font(run, cn=SONG, latin=LATIN, size=12, bold=False, italic=False, color=None):
    run.font.name = latin
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color:
        run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), latin)
    rfonts.set(qn("w:hAnsi"), latin)
    rfonts.set(qn("w:eastAsia"), cn)


def para_fmt(p, first_indent_chars=0, space_before=0, space_after=6,
             line=1.5, align=None):
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing = line
    if first_indent_chars:
        # 用字符数设置首行缩进（w:ind firstLineChars）
        pPr = p._element.get_or_add_pPr()
        ind = pPr.find(qn("w:ind"))
        if ind is None:
            ind = OxmlElement("w:ind")
            pPr.append(ind)
        ind.set(qn("w:firstLineChars"), str(first_indent_chars * 100))
        ind.set(qn("w:firstLine"), str(int(first_indent_chars * 12 * 20)))
    if align is not None:
        p.alignment = align


def add_para(doc, text, cn=SONG, size=12, bold=False, indent=2,
             align=None, space_before=0, space_after=6, line=1.5):
    p = doc.add_paragraph()
    para_fmt(p, indent, space_before, space_after, line, align)
    add_rich_text(p, text, cn=cn, size=size, bold=bold)
    return p


def add_rich_text(p, text, cn=SONG, size=12, bold=False):
    """处理 **粗体** 与 `code` 行内标记。"""
    tokens = re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text)
    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            r = p.add_run(tok[2:-2])
            set_run_font(r, cn=cn, size=size, bold=True)
        elif tok.startswith("`") and tok.endswith("`") and len(tok) > 2:
            r = p.add_run(tok[1:-1])
            set_run_font(r, cn=cn, latin=MONO, size=size - 1)
        else:
            r = p.add_run(tok)
            set_run_font(r, cn=cn, size=size, bold=bold)


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hexcolor)
    tcPr.append(shd)


def set_cell_margins(table, top=40, bottom=40, left=80, right=80):
    tblPr = table._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for tag, val in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        e = OxmlElement(f"w:{tag}")
        e.set(qn("w:w"), str(val))
        e.set(qn("w:type"), "dxa")
        mar.append(e)
    tblPr.append(mar)


def parse_table_row(line):
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def is_sep_row(line):
    cells = parse_table_row(line)
    return all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c != "")


def _set_tbl_borders(table, sz=6, header_sz=4):
    """三线表：表格四周无框线，仅顶线（粗）、表头下线（细）、底线（粗）。"""
    tblPr = table._tbl.tblPr
    for old in tblPr.findall(qn("w:tblBorders")):
        tblPr.remove(old)
    borders = OxmlElement("w:tblBorders")
    for edge, val, width in (
        ("top", "single", sz),
        ("left", "none", 0),
        ("bottom", "single", sz),
        ("right", "none", 0),
        ("insideH", "none", 0),
        ("insideV", "none", 0),
    ):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), val)
        el.set(qn("w:sz"), str(width))
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
        borders.append(el)
    tblPr.append(borders)

    # 表头行下线（细）
    for cell in table.rows[0].cells:
        tcPr = cell._tc.get_or_add_tcPr()
        for old in tcPr.findall(qn("w:tcBorders")):
            tcPr.remove(old)
        tcb = OxmlElement("w:tcBorders")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), str(header_sz))
        bottom.set(qn("w:space"), "0")
        bottom.set(qn("w:color"), "000000")
        tcb.append(bottom)
        tcPr.append(tcb)


def add_table(doc, rows, caption=None):
    header, body = rows[0], rows[1:]
    ncol = len(header)
    if caption:
        p = doc.add_paragraph()
        para_fmt(p, 0, 6, 3, 1.2, WD_ALIGN_PARAGRAPH.CENTER)
        add_rich_text(p, caption, cn=HEI, size=10.5)

    table = doc.add_table(rows=1, cols=ncol)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    set_cell_margins(table)

    # 表格铺满正文宽度（15.4 cm = A4 21 - 左右边距各 2.8）
    from docx.shared import Cm as _Cm
    tblW = OxmlElement("w:tblW")
    tblW.set(qn("w:type"), "pct")
    tblW.set(qn("w:w"), "5000")
    table._tbl.tblPr.append(tblW)

    hdr = table.rows[0].cells
    for i, cell_text in enumerate(header):
        cell = hdr[i]
        cell.text = ""
        p = cell.paragraphs[0]
        para_fmt(p, 0, 2, 2, 1.0, WD_ALIGN_PARAGRAPH.CENTER)
        add_rich_text(p, cell_text, cn=HEI, size=10.5, bold=True)

    for row_cells in body:
        cells = table.add_row().cells
        for i in range(ncol):
            txt = row_cells[i] if i < len(row_cells) else ""
            cell = cells[i]
            cell.text = ""
            p = cell.paragraphs[0]
            para_fmt(p, 0, 2, 2, 1.0, WD_ALIGN_PARAGRAPH.CENTER)
            add_rich_text(p, txt, cn=SONG, size=10.5)

    _set_tbl_borders(table)

    # 表后留白
    p = doc.add_paragraph()
    para_fmt(p, 0, 0, 6, 1.0)


def add_code_block(doc, lines):
    p = doc.add_paragraph()
    para_fmt(p, 0, 4, 4, 1.0)
    pPr = p._element.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), "F5F5F5")
    pPr.append(shd)
    ind = OxmlElement("w:ind")
    ind.set(qn("w:left"), "284")
    pPr.append(ind)
    for i, ln in enumerate(lines):
        if i:
            p.add_run().add_break()
        r = p.add_run(ln)
        set_run_font(r, cn=MONO, latin=MONO, size=9)


def resolve_image(src):
    """把 Markdown 里的图片路径解析成磁盘路径。

    先按 PAPER.md 所在目录（docs/）解析，再退回项目根目录——同一份文档在
    GitHub 上与在这里渲染时，相对路径的基准不一样，两种写法都要能取到图。
    """
    if os.path.isabs(src):
        return src
    for base in (os.path.dirname(SRC), BASE):
        p = os.path.join(base, src)
        if os.path.exists(p):
            return p
    return os.path.join(BASE, src)      # 都不存在：交给调用方打印「图片缺失」


def add_figure(doc, path, caption, width_cm=15.2):
    if not os.path.exists(path):
        print(f"  !! 图片缺失: {path}")
        return
    p = doc.add_paragraph()
    para_fmt(p, 0, 8, 3, 1.0, WD_ALIGN_PARAGRAPH.CENTER)
    p.add_run().add_picture(path, width=Cm(width_cm))
    cp = doc.add_paragraph()
    para_fmt(cp, 0, 0, 8, 1.2, WD_ALIGN_PARAGRAPH.CENTER)
    add_rich_text(cp, caption, cn=HEI, size=10.5)


def add_page_number_footer(section):
    """在页脚居中插入 PAGE 域，显示当前页码。"""
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para_fmt(p, 0, 0, 0, 1.0, WD_ALIGN_PARAGRAPH.CENTER)

    run = p.add_run()
    set_run_font(run, cn=SONG, size=10.5)
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    for e in (fld_begin, instr, fld_sep, fld_end):
        run._element.append(e)


def main():
    md = io.open(SRC, encoding="utf-8").read()
    lines = md.split("\n")

    doc = Document()

    # 页面：A4，页边距
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.top_margin, sec.bottom_margin = Cm(2.5), Cm(2.5)
    sec.left_margin, sec.right_margin = Cm(2.8), Cm(2.8)

    # 正文默认样式
    style = doc.styles["Normal"]
    style.font.name = LATIN
    style.font.size = Pt(12)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), SONG)

    # 页脚页码
    add_page_number_footer(sec)

    i = 0
    n = len(lines)
    first_h1 = True

    while i < n:
        raw = lines[i]
        line = raw.rstrip()
        s = line.strip()

        if not s:
            i += 1
            continue

        # 分隔线
        if re.fullmatch(r"-{3,}", s):
            i += 1
            continue

        # 一级标题 = 论文题目
        if s.startswith("# ") and not s.startswith("## "):
            title = s[2:].strip()
            if first_h1:
                tp = doc.add_paragraph()
                para_fmt(tp, 0, 0, 18, 1.5, WD_ALIGN_PARAGRAPH.CENTER)
                add_rich_text(tp, title, cn=HEI, size=18, bold=True)
                first_h1 = False
            else:
                hp = doc.add_paragraph()
                para_fmt(hp, 0, 18, 10, 1.5, WD_ALIGN_PARAGRAPH.CENTER)
                add_rich_text(hp, title, cn=HEI, size=16, bold=True)
            i += 1
            continue

        # 二级标题
        if s.startswith("## "):
            txt = s[3:].strip()
            if txt in ("参考文献", "附录 A　复现命令", "附录 B　产物清单") or txt.startswith("附录"):
                p = doc.add_paragraph()
                para_fmt(p, 0, 18, 10, 1.5, WD_ALIGN_PARAGRAPH.CENTER)
                add_rich_text(p, txt, cn=HEI, size=14, bold=True)
            else:
                p = doc.add_paragraph()
                para_fmt(p, 0, 16, 8, 1.5)
                add_rich_text(p, txt, cn=HEI, size=14, bold=True)
            i += 1
            continue

        # 三级标题
        if s.startswith("### "):
            txt = s[4:].strip()
            p = doc.add_paragraph()
            para_fmt(p, 0, 12, 6, 1.5)
            add_rich_text(p, txt, cn=HEI, size=12, bold=True)
            i += 1
            continue

        # 四级标题
        if s.startswith("#### "):
            txt = s[5:].strip()
            p = doc.add_paragraph()
            para_fmt(p, 0, 10, 6, 1.5)
            add_rich_text(p, txt, cn=HEI, size=12, bold=True)
            i += 1
            continue

        # 代码块
        if s.startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i].rstrip())
                i += 1
            i += 1
            add_code_block(doc, buf)
            continue

        # 图片
        m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", s)
        if m:
            cap, src = m.group(1), m.group(2)
            if not src.startswith("http"):
                add_figure(doc, resolve_image(src), cap)
            i += 1
            continue

        # 表题（表格上方居中，无首行缩进）
        m = re.match(r"^\*\*表\s*\d+.*\*\*$", s)
        if m:
            p = doc.add_paragraph()
            para_fmt(p, 0, 10, 3, 1.2, WD_ALIGN_PARAGRAPH.CENTER)
            add_rich_text(p, s.strip("*"), cn=HEI, size=10.5, bold=True)
            i += 1
            continue

        # 表格
        if s.startswith("|") and i + 1 < n and is_sep_row(lines[i + 1]):
            rows = [parse_table_row(s)]
            i += 2
            while i < n and lines[i].strip().startswith("|"):
                rows.append(parse_table_row(lines[i]))
                i += 1
            add_table(doc, rows)
            continue

        # 有序列表
        m = re.match(r"(\d+)\.\s+(.*)", s)
        if m:
            p = doc.add_paragraph()
            para_fmt(p, 2, 0, 6, 1.5)
            p.paragraph_format.left_indent = Cm(0.74)
            add_rich_text(p, f"{m.group(1)}. {m.group(2)}")
            i += 1
            continue

        # 无序列表
        if s.startswith("- "):
            p = doc.add_paragraph()
            para_fmt(p, 2, 0, 4, 1.5)
            p.paragraph_format.left_indent = Cm(0.74)
            add_rich_text(p, "· " + s[2:])
            i += 1
            continue

        # 摘要 / 关键词 / 参考文献条目
        if s.startswith("**摘要**"):
            p = doc.add_paragraph()
            para_fmt(p, 2, 6, 8, 1.5)
            body = s[len("**摘要**"):].strip()
            r = p.add_run("摘要　")
            set_run_font(r, cn=HEI, size=12, bold=True)
            add_rich_text(p, body, cn=KAI, size=12)
            i += 1
            continue

        if s.startswith("**关键词**"):
            p = doc.add_paragraph()
            para_fmt(p, 2, 0, 12, 1.5)
            body = s[len("**关键词**"):].strip()
            r = p.add_run("关键词　")
            set_run_font(r, cn=HEI, size=12, bold=True)
            add_rich_text(p, body, cn=KAI, size=12)
            i += 1
            continue

        # 参考文献 [1] ...
        if re.match(r"\[\d+\]\s", s):
            p = doc.add_paragraph()
            para_fmt(p, 0, 0, 4, 1.4)
            p.paragraph_format.left_indent = Cm(0.9)
            p.paragraph_format.first_line_indent = Cm(-0.9)
            add_rich_text(p, s, cn=SONG, size=10.5)
            i += 1
            continue

        # 正文段落（合并连续行）
        buf = [s]
        i += 1
        while i < n:
            nxt = lines[i].strip()
            if (not nxt or nxt.startswith(("#", "|", "```", "![", "- ", "> "))
                    or re.fullmatch(r"-{3,}", nxt)
                    or re.match(r"\d+\.\s", nxt)
                    or re.match(r"\[\d+\]\s", nxt)):
                break
            buf.append(nxt)
            i += 1
        add_para(doc, "".join(buf))

    doc.save(OUT)
    print(f"saved: {OUT}")
    print(f"size: {os.path.getsize(OUT):,} B")


if __name__ == "__main__":
    main()
