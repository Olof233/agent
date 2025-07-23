from PyPDF2 import PdfReader
import re
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline


def get_pdf_text(pdf):
    with open(pdf, "rb") as f:
        pdf_reader = PdfReader(f)
        lines = []
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            for line in text.split('\n'):
                lines.append(line)
    return lines


def process_file(lines):
    lines = [subline for line in lines for subline in line.replace('。', '。\n').split('\n')]
    raw = '\n'.join(lines)
    raw = re.sub(r'“([\S]+?)”', lambda m: m.group(0).replace('\n', ''), raw)
    raw = re.sub(r'“([\S\r\n]+?)”', lambda m: re.sub(r'[\r\n]', '', m.group(0)), raw)
    lines = raw.split('\n')
    lines = [re.sub(r's', '', line).strip().replace('s', '—') for line in lines]
    lines = [line for line in lines if line]
    return lines


def group_lines(lines, group_size=20):
    grouped = []
    for i in range(0, len(lines), group_size):
        group = ' '.join(lines[i:i+group_size])
        grouped.append(group)
    return grouped


def bert_chunk(lines):
    p = pipeline(
    task=Tasks.document_segmentation,
    model='bert', model_revision='master')

    line = group_lines(lines, len(lines))
    result = p(line)
    p = None
    return [chunk.strip("\t") for chunk in result[OutputKeys.TEXT].split("\n") if len(chunk.strip("\t")) > 0] #type: ignore
