"""Public phenotype/pedigree -> posterior genotype -> complex offspring coat.

Scientific cautions:
- This is a predictive, not causal, genotype model.
- Founder allele frequencies came from ONE source breeding population.
- Local two-generation pedigree is processed exactly per locus; loci are
  subsequently combined approximately independently by compound.py.
- Allele penetrances are estimated from breeder phenotype labels and may not
  generalize to other lineages or distinguish masking from non-expression.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from compound import BASES, SHADES, predict_compound
from mendel import mendelian_tensor, sex_transition

LOCI_LABEL = {
    'rpd':'劣性パイド（仮説）',
    'ch':'チェスナット／ハバナ関連（仮説）',
    'fwn':'フォーン／猩々関連（仮説）',
    'pl':'和パール関連（Z連鎖仮説）',
}


def load_public_model(path=None):
    path = Path(path or Path(__file__).with_name('public_model.json'))
    model = json.loads(path.read_text(encoding='utf-8'))
    if model.get('schema_version') != 1 or any(k not in model for k in ('compound','loci','validation')):
        raise ValueError('公開モデルの形式が違います')
    if 'birds' in model: raise ValueError('個体別情報を含むモデルは公開アプリに使用できません')
    return model


def trait(base=None, shade=None, pied=None, morphology=()):
    """Explicit user-observed traits, with missing values preserved as None."""
    base = base if base in BASES else None
    shade = shade if shade in SHADES else None
    if base in ('白','アルビノ'):
        shade = None
        pied = None  # masked; absence of spots is NOT observed negative
    if base == '薄墨':
        shade = '通常'  # distinct base label rather than dilution evidence
    return dict(base=base, shade=shade, pied=pied if pied in (0,1) else None,
                morphology=tuple(morphology) if morphology is not None else None)


def _norm(x):
    a=np.asarray(x,dtype=float)
    z=float(a.sum())
    if z <= 0 or not np.isfinite(z):
        raise ValueError('遺伝子型確率を正規化できません')
    return a/z


def prior(q, mode, sex):
    """Local source-lineage allele-frequency prior, not universal population q."""
    q=float(q)
    if mode == 'z_linked' and sex == 'F':
        return np.array([1-q,0.0,q])
    return np.array([(1-q)**2,2*q*(1-q),q*q])


def evidence_value(t, locus):
    """Return 1, 0, or unknown (None); avoid imputing hidden color markers.

    In multi-locus color effects, 'absence' is a weak/non-causal observation.
    These mappings mirror the v3 model labels without claiming mutation IDs.
    """
    base = t.get('base')
    if locus == 'rpd':
        return t.get('pied')  # can be positive despite uncertain base
    if base in (None, '白', 'アルビノ', 'その他', '薄墨', 'ハバナグレー'):
        return None
    positives={'ch':'ハバナ','fwn':'猩々','pl':'和パール'}
    if locus in positives:
        return int(base == positives[locus])
    return None


def likelihood(t, locus, params):
    observed=evidence_value(t,locus)
    if observed is None:
        return np.ones(3,dtype=float)
    pen=np.clip(np.asarray(params['comparison']['penetrance'],dtype=float),1e-6,1-1e-6)
    return pen if observed == 1 else 1-pen


def founder_posterior(t, sex, locus, params):
    return _norm(prior(params['allele_frequency'],params['mode'],sex) * likelihood(t,locus,params))


def parent_posterior(t, sex, locus, params, grandparents=None):
    """Exact marginal for one parent + two grandparents in a small acyclic trio.

    Granparent tuple: (paternal/maternal grandfather observed traits,
    grandmother observed traits). If no grandparents, use founder prior.
    """
    if grandparents is None:
        gen_prior=prior(params['allele_frequency'],params['mode'],sex)
    else:
        grandpa,grandma=grandparents
        pf=founder_posterior(grandpa,'M',locus,params)
        pm=founder_posterior(grandma,'F',locus,params)
        t3=mendelian_tensor([{'sex':sex}],[0],params['mode'])[0]
        gen_prior=np.einsum('a,b,abc->c',pf,pm,t3)
    return _norm(gen_prior*likelihood(t,locus,params))


def make_synthetic_model(aggregate, father, mother, grandparents=None):
    synthetic_birds=[{'parent_id':'external-father','phenotype':'', 'sex':'M'},
                     {'parent_id':'external-mother','phenotype':'', 'sex':'F'}]
    grandparents=grandparents or {}
    layers={}
    locus_summaries={}
    for locus,params in aggregate['loci'].items():
        gp_f = grandparents.get('father')
        gp_m = grandparents.get('mother')
        pf=parent_posterior(father,'M',locus,params,gp_f)
        pm=parent_posterior(mother,'F',locus,params,gp_m)
        layers[locus]=dict(posterior=np.stack([pf,pm]),
                           locus={'mode':params['mode']},
                           comparison=params['comparison'],
                           bp={'converged':params['bp_converged'] is True})
        locus_summaries[locus]={'father':pf,'mother':pm}
    return {'birds':synthetic_birds,'loci':layers,'compound':aggregate['compound']},locus_summaries


def predict(aggregate, father, mother, grandparents=None, sex='?', draws=800, seed=20261025):
    synthetic, genotypes=make_synthetic_model(aggregate,father,mother,grandparents)
    if sex not in ('M','F','?'):
        raise ValueError('性別は M/F/?')
    pred=predict_compound(synthetic,'external-father','external-mother', sex, draws=draws,
                          seed=seed, parent_traits=(father,mother))
    states={}
    for locus,post in genotypes.items():
        modes=aggregate['loci'][locus]['mode']
        child=sex_transition(post['father'],post['mother'],modes,sex)
        states[locus]={'father':post['father'].tolist(),'mother':post['mother'].tolist(),
                       'offspring':child.tolist(),'mode':modes,
                       'marker_probability':float(np.dot(child,aggregate['loci'][locus]['comparison']['penetrance']))}
    pred['genotype_posteriors']=states
    pred['warnings'] += [
        '外部飼育者の個体の外見入力から親の遺伝子型を推定する機能は、外部血統で独立検証されていません。',
        '95%信用区間は固定した遺伝様式と学習集団に条件づけた近似区間であり、校正済みの保証区間ではありません。',
    ]
    if any(params['comparison']['identifiable'] is False for params in aggregate['loci'].values()):
        pred['warnings'].append('一部の形質は遺伝様式の識別に使える記録が不足しています（特に和パール）。')
    if grandparents:
        pred['warnings'].append('祖父母の情報は座位ごとに更新しましたが、祖父母同士の血縁・多世代遺伝・表現型の誤分類を完全にはモデル化していません。')
    return pred
