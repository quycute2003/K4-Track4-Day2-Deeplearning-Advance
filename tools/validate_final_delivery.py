"""Verify exported workbook caches and all final deliverables without running a model."""
import hashlib
import json
import math
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import eval as ev


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def worksheets(path):
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    rel_ns = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    with zipfile.ZipFile(path) as archive:
        strings = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            strings = [''.join(t.text or '' for t in item.findall('.//m:t', ns))
                       for item in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('m:si', ns)]
        rels = {r.attrib['Id']: r.attrib['Target'] for r in ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))}
        result = {}
        for sheet in ET.fromstring(archive.read('xl/workbook.xml')).findall('m:sheets/m:sheet', ns):
            target = rels[sheet.attrib['{' + rel_ns + '}id']]
            filename = target.lstrip('/') if target.startswith('/') else 'xl/' + target
            cells = {}
            for cell in ET.fromstring(archive.read(filename)).findall('.//m:sheetData/m:row/m:c', ns):
                kind = cell.get('t')
                assert kind != 'e', f"Excel cached error: {sheet.get('name')}!{cell.get('r')}"
                value = cell.find('m:v', ns)
                if kind == 'inlineStr':
                    parsed = ''.join(t.text or '' for t in cell.findall('.//m:t', ns))
                elif value is None or value.text is None:
                    parsed = None
                elif kind == 's':
                    parsed = strings[int(value.text)]
                elif kind == 'b':
                    parsed = value.text == '1'
                elif kind == 'str':
                    parsed = value.text
                else:
                    parsed = float(value.text)
                formula = cell.find('m:f', ns)
                cells[cell.get('r')] = (parsed, formula.text if formula is not None else None)
            result[sheet.get('name')] = cells
    return result


def main():
    plan = read('step4/frozen_plan.json')
    audit = read('step5/verification.json')
    digest = hashlib.sha256(b''.join((ROOT / f'code/{n}.py').read_bytes()
        for n in ('step4', 'train', 'dataset', 'model', 'losses', 'inference', 'benchmark'))).hexdigest()
    assert digest == plan['code_sha256'], 'Frozen training/inference source changed'
    assert hashlib.sha256((ROOT / 'step4/frozen_plan.json').read_bytes()).hexdigest() == audit['frozen_plan_sha256']
    assert hashlib.sha256((ROOT / 'eval.py').read_bytes()).hexdigest() == '1210f3c7ca1cb9b4488383f1ceeaa139bfdf5c683694b300f6c55b6b9cdde28e'
    book = worksheets(ROOT / 'results.xlsx')
    assert set(book) == {'Backbones', 'Training', 'Inference', 'Latency', 'Summary', 'Final', 'PerClass'}
    expected = read('step5/final_workbook.json')
    for row_id, row in enumerate(expected, 2):
        for column_id, (key, value) in enumerate(row.items()):
            address = chr(65 + column_id) + str(row_id)
            actual, formula = book['Final'][address]
            if isinstance(value, (int, float)):
                assert isinstance(actual, (int, float)) and math.isfinite(actual)
                assert abs(actual - value) < 1e-10, 'Final cell mismatch ' + address
            elif value is not None:
                assert actual == value, address
            if row_id in (8, 9) and key.endswith('_std'):
                assert formula and 'STDEV.S' in formula
    assert abs(book['Summary']['B17'][0] - expected[7]['macro_f1_test']) < 1e-12
    assert abs(book['Summary']['D19'][0] - 100 * read('step5/analysis.json')['delta_f1_test']) < 1e-10
    assert abs(book['Summary']['D20'][0] - 100 * read('step5/analysis.json')['larger_std']) < 1e-10
    assert all(book['Backbones'][f'T{i}'][0] == 224 for i in range(2, 7))
    schema = read('step3/latency_schema.json') + ['seed', 'temperature']
    timings = read('step5/latency_workbook.json')
    assert len(timings) == 24
    for row_id, row in enumerate(timings, 2):
        for column_id, key in enumerate(schema):
            actual = book['Latency'][chr(65 + column_id) + str(row_id)][0]
            value = row.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                assert isinstance(actual, (int, float)) and abs(actual - value) < 1e-10
            else:
                assert actual == value
    gold = ROOT / 'runs/step0_validation/labels'
    for tag in ('F00', 'F01', 'F01_uncal'):
        for split in ('val', 'test'):
            group = ev.load_group(str(ROOT / f'predictions/{tag}_seed*_{split}.csv'), str(gold / f'{split}_subset0.csv'), ref_what=split)
            assert group.seeds == [0, 1, 2]
            if split == 'test':
                saved = read(f'step4/eval_outputs/{tag}_summary.json')
                for metric in ev.SCALARS:
                    np.testing.assert_allclose(group.summary[metric], [saved[metric]['mean'], saved[metric]['std']], rtol=0, atol=1e-12)
    for group in ('F00', 'F01'):
        for seed in (0, 1, 2):
            assert (ROOT / f'curves/{group}_convnext_tiny_seed{seed}.png').is_file()
    for exp in ['B01', 'B02', 'B03', 'B04', 'B05', *[f'T{s:02}' for s in range(1, 8)]]:
        assert list((ROOT / 'curves').glob(exp + '_*.png')), exp
    report = (ROOT / 'report.md').read_text(encoding='utf-8')
    assert '96.86 ± 0.27%' in report and '97.58 ± 0.19%' in report
    for image in re.findall(r'!\[[^\]]*\]\(([^)]+)\)', report):
        assert (ROOT / image).is_file(), image
    pred = pd.read_csv(ROOT / 'predictions/F01_seed0_test.csv').set_index('Filename')
    for example in read('step5/error_examples.json'):
        source = pred.loc[example['Filename']]
        assert (int(source.y_true), int(source.y_pred)) == (example['y_true'], example['y_pred'])
        assert example['y_true'] != example['y_pred']
    assert not read('step5/status.json')['pending_steps']
    assert all((ROOT / p).exists() for p in ('results.xlsx', 'report.md', 'curves', 'code', 'REPRODUCE.md', 'predictions'))
    print('Six deliverables verified: seven Excel sheets, mean/std caches/formulas, 24 timings, 18 final CSVs and 18 B/T/F curves.')
    print('Frozen code/plan/evaluator preserved. Official scores and error-gallery filenames match; no model forward performed.')


if __name__ == '__main__':
    main()
