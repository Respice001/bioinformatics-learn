"""Extract aggregate model only; NEVER publish original full model.json or ODS."""
import argparse
import json
import hashlib
from datetime import date
from pathlib import Path


def export(source, target, validation=None):
    src = Path(source)
    orig = json.loads(src.read_text(encoding='utf-8'))
    if 'compound' not in orig or 'loci' not in orig:
        raise ValueError('The input is not the trained v2/v3 model.')
    raw = {
        'schema_version': 1,
        'description': '匿名集計済みの和十姉妹羽色予測パラメータ。外部個体への適用は未検証。',
        'training_model_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
        'export_date': date.today().isoformat(),
        'counts': {'records': len(orig.get('birds',[])),
                   'offspring': sum(b.get('sire_no') not in (None,'') for b in orig.get('birds',[])),
                   'mating_pairs': len({(b.get('sire_no'),b.get('dam_no')) for b in orig.get('birds',[])
                                        if b.get('sire_no') not in (None,'')})},
        'compound': orig['compound'],
        'loci': {
            k: {'mode': v['config']['mode'],
                'allele_frequency': v['allele_frequency'],
                'comparison': {x:v['model_comparison'][x] for x in (
                    'selected','penetrance','posterior_beta','n_informative',
                    'n_genotype_groups','identifiable','delta_log_evidence')},
                'bp_converged': v.get('bp',{}).get('converged',None)}
            for k,v in orig['loci'].items()},
        'validation': {},
    }
    if validation and Path(validation).exists():
        v = json.loads(Path(validation).read_text(encoding='utf-8'))
        allowed=('status','heldout_mating_pairs','heldout_rows','evaluated_rows','n_categories',
                 'multiclass_brier','baseline_brier','multiclass_logloss','baseline_logloss',
                 'accuracy_top1','baseline_accuracy_top1','warning')
        raw['validation'] = {k:v[k] for k in allowed if k in v}
    encoded = json.dumps(raw,ensure_ascii=False,indent=2).encode('utf-8')
    if any(field in raw for field in ('birds','posterior','name','raw','uid','parent_id')):
        raise RuntimeError('Privacy gate: private fields found')
    dest=Path(target); dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(encoded)
    return raw

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--validation')
    args=ap.parse_args()
    item=export(args.model,args.output,args.validation)
    print('Public model saved:',args.output, 'records:',item['counts']['records'])
