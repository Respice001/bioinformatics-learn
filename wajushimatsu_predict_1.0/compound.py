"""Approximate empirical-Bayes compositional phenotype prediction.

This is a *predictive*, partial-pooling model, not proven causal epistasis.
The numerical 95% credible intervals are conditional on this model's rules,
selected priors, and approximate locus-posteriors (not independently calibrated
coverage guarantees). Parent-genotype independence is an approximation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from itertools import product

import numpy as np

BASES = ('和黒', 'ハバナ', '猩々', '和パール', '薄墨', 'ハバナグレー', 'アルビノ', '白', 'その他')
SHADES = ('通常', '薄', '極薄')
UNKNOWN = ('', '-', '?', '不明')
GENE_LOCI = ('ch', 'fwn', 'pl')
MORPHOLOGY = ('梵天', '千代田', '中納言', '大納言', 'フリル')


def parse_traits(phenotype):
    """Parse explicitly recorded phenotype parts; ambiguous cases stay NA."""
    s = str(phenotype or '').strip().replace('（薄）', '_薄').replace('薄和黒', '薄和黒')
    if s in UNKNOWN:
        return dict(base=None, shade=None, pied=None, morphology=None)
    dubious = ('？' in s or '?' in s or 'or' in s.lower() or 'Mix' in s or 'ミックス' in s)
    if dubious:
        return dict(base=None, shade=None, pied=None, morphology=None)
    # 'sp和パール' (possible carrier) is not a pearl phenotype.
    s_base = s.replace('sp和パール', '').replace('sp和パール', '')
    if '/黒' in s_base or s_base.startswith('/'):
        base = None   # mixed/slashed annotation: do not assume a defined coat
    elif '和アルビノ' in s_base:
        base = 'アルビノ'
    elif 'ハバナグレー' in s_base:
        base = 'ハバナグレー'
    elif 'ハバナ' in s_base:
        base = 'ハバナ'
    elif '猩々' in s_base:
        base = '猩々'
    elif '和パール' in s_base:
        base = '和パール'
    elif '薄墨' in s_base:
        base = '薄墨'
    elif '和黒' in s_base or '薄黒' in s_base or s_base.startswith('黒'):
        base = '和黒'
    elif s_base.startswith('白'):
        base = '白'
    elif 'チョコレート' in s_base or 'チェスナット' in s_base:
        base = 'その他'
    else:
        base = None   # e.g. 小斑 only: base coat not stated
    if base is None:
        shade = None
    elif '極薄' in s or '極薄' in s_base:
        shade = '極薄'
    elif base == '薄墨':
        shade = '通常'  # "薄墨" is a phenotype label, not a verified dilution level
    elif '薄' in s and base not in ('白','アルビノ'):
        shade = '薄'
    else:
        shade = '通常'
    pied = None if base in ('白', 'アルビノ') else int('pd' in s.lower() or '小斑' in s or '茶斑' in s)
    # For base-unobserved pied-only labels: pied is observable and should remain available.
    if base is None and ('pd' not in s.lower() and '小斑' not in s and '茶斑' not in s):
        pied = None
    morph = tuple(t for t in MORPHOLOGY if t in s)
    return dict(base=base, shade=shade, pied=pied, morphology=morph)


def _parent_pair_key(a, b):
    return '|'.join(sorted((a or '?', b or '?')))


def _parent_shade_key(a, b):
    traits = (a, b)
    if any(t['shade'] is None for t in traits):
        return '?'
    return str(sum(t['shade'] != '通常' for t in traits))


def _parent_pied_key(a, b):
    if a['pied'] is None or b['pied'] is None:
        return '?'
    return str(a['pied'] + b['pied'])


def _norm(x):
    v = np.asarray(x, dtype=float)
    return v / v.sum(axis=-1, keepdims=True)


def _dct_tables(d):
    return {k: np.asarray(v, float).tolist() for k,v in d.items()}


def _get(t, k, n):
    return np.asarray(t.get(str(k), [0.] * n), dtype=float)


def _signature_weights(risks):
    """Soft probability of eight possible 3-marker signatures (independence approx)."""
    c, f, p = [float(np.clip(x, 0, 1)) for x in risks]
    return np.array([np.prod([a if bit else 1-a for a,bit in zip((c,f,p),bits)])
                     for bits in product((0,1),repeat=3)], float)


def _child_risks(locus_models, dad_idx, mom_idx, sex):
    from mendel import sex_transition
    result = {}
    for locus in ('rpd',) + GENE_LOCI:
        m = locus_models[locus]
        post = sex_transition(m['posterior'][dad_idx], m['posterior'][mom_idx],
                              m['locus']['mode'], sex)
        result[locus] = float(post[2])
    return result


def fit_compound(birds, loci, training_mask=None):
    """Fit smoothed Dirichlet/Beta tables to parent-conditioned progeny data.

    Unlike using the observed child's posterior genotype, inherited-marker
    features are calculated from *parents*, which avoids child-label leakage.
    """
    if training_mask is None:
        training_mask = np.ones(len(birds), dtype=bool)
    traits = [parse_traits(b['phenotype']) for b in birds]
    ix_b = {x:i for i,x in enumerate(BASES)}
    ix_s = {x:i for i,x in enumerate(SHADES)}
    B, S = len(BASES),len(SHADES)
    bg = np.zeros(B)
    bp = defaultdict(lambda: np.zeros(B))
    bs = defaultdict(lambda: np.zeros(B))
    bgen = {str(i): np.zeros(B) for i in range(8)}
    sg = np.zeros(S)
    s_base = defaultdict(lambda: np.zeros(S))
    s_cond = defaultdict(lambda: np.zeros(S))
    pg = np.zeros(2)
    p_base = defaultdict(lambda: np.zeros(2))
    p_genebase = defaultdict(lambda: np.zeros(2))
    p_gene_all = defaultdict(lambda: np.zeros(2))
    p_par = defaultdict(lambda: np.zeros(2))
    morph = {k: np.zeros(2) for k in MORPHOLOGY}
    morph_par = defaultdict(lambda: np.zeros(2))
    data = Counter()
    for i, bird in enumerate(birds):
        if not training_mask[i] or bird['sire'] is None:
            continue
        dad, mom = bird['sire'], bird['dam']
        t,td,tm = traits[i],traits[dad],traits[mom]
        pair = _parent_pair_key(td['base'], tm['base'])
        spair = _parent_shade_key(td,tm)
        ppair = _parent_pied_key(td,tm)
        if t['base'] is not None:
            bi=ix_b[t['base']]
            bg[bi]+=1;bp[pair][bi]+=1
            for parent in (td, tm):
                if parent['base'] is not None:
                    bs[parent['base']][bi]+=.5
            risks=_child_risks(loci,dad,mom,bird['sex'])
            weight=_signature_weights([risks[x] for x in GENE_LOCI])
            for j, w in enumerate(weight):
                bgen[str(j)][bi]+=w
            data['n_base']+=1
            if t['shade'] is not None:
                si=ix_s[t['shade']]
                sg[si]+=1;s_base[t['base']][si]+=1
                s_cond[t['base']+'|'+spair][si]+=1
                data['n_shade']+=1
        if t['pied'] is not None:
            pi=t['pied']
            pg[pi]+=1;p_par[ppair][pi]+=1
            # Keep pied-only labels (e.g. 小斑) even when the base color is missing.
            r = _child_risks(loci,dad,mom,bird['sex'])['rpd']
            p_gene_all['0'][pi]+=1-r
            p_gene_all['1'][pi]+=r
            if t['base'] is not None:
                p_base[t['base']][pi]+=1
                p_genebase[t['base']+'|0'][pi]+=1-r
                p_genebase[t['base']+'|1'][pi]+=r
            data['n_pied']+=1
        if t['morphology'] is not None:
            for name in MORPHOLOGY:
                y=int(name in t['morphology'])
                morph[name][y]+=1
                # Both, one or neither parental carriers of this *label*.
                parent_obs=int(name in (td['morphology'] or ()))+int(name in (tm['morphology'] or ()))
                morph_par[name+'|'+str(parent_obs)][y]+=1
                data['n_morphology']+=1
    return dict(version=2, counts=dict(data), base_global=bg.tolist(),base_parents=_dct_tables(bp),
                base_single_parent=_dct_tables(bs),base_gene=_dct_tables(bgen),
                shade_global=sg.tolist(),shade_base=_dct_tables(s_base),shade_condition=_dct_tables(s_cond),
                pied_global=pg.tolist(),pied_base=_dct_tables(p_base),
                pied_genebase=_dct_tables(p_genebase),pied_gene_all=_dct_tables(p_gene_all),
                pied_parents=_dct_tables(p_par),
                morphology={k:v.tolist() for k,v in morph.items()},
                morphology_parent=_dct_tables(morph_par),
                assumptions=['基底色/濃淡/パイドを条件付き確率として分離（濃淡とパイドの残差相関は未推定）',
                             '各遺伝子座の親の潜在遺伝子型をMonte Carlo抽出（遺伝子座間と両親間の事後相関は近似的に独立）',
                             '階層Dirichlet/Betaの事前分布をデータ由来平均で近似する経験ベイズ',
                             '区間は固定した遺伝様式・事前強度・近似事後分布の条件下での信用区間',
                             '同じ血統内の血縁相関・集団構造・未記載遺伝子を完全には反映しない'])


def _parameters(compound, traits_f, traits_m, risk):
    """Return Dirichlet concentration vectors for parts of joint model."""
    B=len(BASES);S=len(SHADES)
    # global posterior provides a weak shared prior to rare classes
    bg=_get({'g':compound['base_global']},'g',B)
    base_prior=(bg + .5) / (bg.sum()+.5*B)
    pair=_parent_pair_key(traits_f['base'],traits_m['base'])
    pair_count=_get(compound['base_parents'],pair,B)
    single=np.zeros(B)
    for parent in (traits_f,traits_m):
        if parent['base'] is not None:
            single += .25*_get(compound['base_single_parent'],parent['base'],B)
    parent_alpha=16*base_prior+pair_count+single
    w=_signature_weights([risk[x] for x in GENE_LOCI])
    gene_alphas=np.array([20*base_prior+_get(compound['base_gene'],i,B) for i in range(8)])
    sg=np.asarray(compound['shade_global'],float)
    shade_global=(sg+.7)/(sg.sum()+S*.7)
    skey=_parent_shade_key(traits_f,traits_m)
    shade_alphas=[]
    pied_alphas=[]
    pg=np.asarray(compound['pied_global'],float)
    pied_global=(pg+.7)/(pg.sum()+1.4)
    pkey=_parent_pied_key(traits_f,traits_m)
    parent_pied=12*pied_global + _get(compound['pied_parents'],pkey,2)
    for base in BASES:
        sc=_get(compound['shade_base'],base,S)
        base_shade=(sc+8*shade_global)/(sc.sum()+8)
        shade_alphas.append(12*base_shade+_get(compound['shade_condition'],base+'|'+skey,S))
        pbc=_get(compound['pied_base'],base,2)
        base_pied=(pbc+10*pied_global)/(pbc.sum()+10)
        all_rpd=np.array([12*pied_global+_get(compound['pied_gene_all'],str(i),2) for i in range(2)])
        all_rpd=_norm(all_rpd)
        by_rpd=np.array([16*(.5*base_pied+.5*all_rpd[i])+
                         _get(compound['pied_genebase'],base+'|'+str(i),2)
                         for i in range(2)])
        pied_alphas.append(by_rpd)
    return parent_alpha, gene_alphas, w, np.array(shade_alphas), np.array(pied_alphas), parent_pied


def predict_compound(model, sire_id, dam_id, sex='?', draws=1000, seed=20261025, minimum=0.0, parent_traits=None):
    """Posterior predictive class probabilities and conditional 95% credible bands."""
    if sex not in ('M','F','?'):
        raise ValueError('子の性別は M / F / ? から指定してください')
    if draws < 100 or draws > 10000:
        raise ValueError('Monte Carlo drawsは100〜10000を指定してください')
    birds=model['birds']
    index={b['parent_id']:i for i,b in enumerate(birds) if b['parent_id']}
    if str(sire_id) not in index or str(dam_id) not in index:
        raise ValueError('親№が見つかりません')
    f=index[str(sire_id)]; m=index[str(dam_id)]
    if f==m or birds[f]['sex']=='F' or birds[m]['sex']=='M':
        raise ValueError('親の組合せ・性別が不正です')
    compound=model.get('compound')
    if not compound:
        raise ValueError('旧版モデルです。python main.py update でv2に更新してください')
    ftraits=parse_traits(birds[f]['phenotype']);mtraits=parse_traits(birds[m]['phenotype'])
    if parent_traits is not None:
        ftraits, mtraits = parent_traits
    risks=_child_risks(model['loci'],f,m,sex)
    pa,ga,ww,sa,pga,ppa = _parameters(compound,ftraits,mtraits,risks)
    # Deterministic per-parent/sex for repeatable reports independent of PYTHONHASHSEED.
    stable=int.from_bytes(sha256(f'{sire_id}/{dam_id}/{sex}/{seed}'.encode()).digest()[:8],'big')
    rng=np.random.default_rng(stable)
    # Propagate uncertainty over the UNKNOWN PARENTAL genotypes, but integrate
    # rather than sample the offspring's Mendelian segregation (aleatory risk).
    from mendel import mendelian_tensor
    tmp=[{'sex':sex}]
    draw_risks={}
    child_rpd_distribution=None
    for name in ('rpd',)+GENE_LOCI:
        lm=model['loci'][name]
        dad_state=rng.choice(3,size=draws,p=_norm(lm['posterior'][f]))
        mom_state=rng.choice(3,size=draws,p=_norm(lm['posterior'][m]))
        transitions=mendelian_tensor(tmp,[0],lm['locus']['mode'])[0]
        child_states=transitions[dad_state,mom_state,:]
        draw_risks[name]=child_states[:,2]
        if name=='rpd':
            child_rpd_distribution=child_states
    ws=np.stack([np.prod([draw_risks[name] if bit else 1-draw_risks[name]
                           for name,bit in zip(GENE_LOCI,bits)],axis=0)
                 for bits in product((0,1),repeat=3)],axis=1)
    pb=.45*rng.dirichlet(pa,size=draws)+.55*sum(ws[:,j,None]*rng.dirichlet(g,size=draws)
                                                    for j,g in enumerate(ga))
    ps=np.stack([rng.dirichlet(x,size=draws) for x in sa],axis=1)
    # rpd genotype prediction enters as a soft mix of two beta-Binomial tables;
    # parental pied phenotype contributes a separate weak hierarchical term.
    g0=np.stack([rng.dirichlet(a[0],size=draws) for a in pga],axis=1)
    g1=np.stack([rng.dirichlet(a[1],size=draws) for a in pga],axis=1)
    pparent=rng.dirichlet(ppa,size=draws)
    # Mendelian penetrance posterior is a strong anchor, rather than
    # allowing rare phenotype categories to override observed rpd inheritance.
    rlm=model['loci']['rpd']
    beta=rlm['comparison'].get('posterior_beta')
    if beta:
        penetrance=np.stack([rng.beta(a,b,size=draws) for a,b in beta],axis=1)
        anchor=np.sum(penetrance*child_rpd_distribution,axis=1)
    else:
        anchor=child_rpd_distribution @ np.asarray(rlm['comparison']['penetrance'])
    local=.8*((1-draw_risks['rpd'])[:,None,None]*g0+draw_risks['rpd'][:,None,None]*g1)+.2*pparent[:,None,:]
    pd_prob=.70*anchor[:,None]+.30*local[:,:,1]
    pp=np.stack([1-pd_prob,pd_prob],axis=2)
    joint=pb[:,:,None,None]*ps[:,:,:,None]*pp[:,:,None,:]
    # A white/albino coat masks pied and shade phenotypes; treating their
    # unobservable combinations as distinct categories would be misleading.
    for base in ('アルビノ','白'):
        ib=BASES.index(base)
        joint[:,ib,1:,:]=0.
        joint[:,ib,:,1]=0.
    # '薄墨' is a single named base category, not proof of a general dilution.
    joint[:,BASES.index('薄墨'),1:,:]=0.
    joint=joint/joint.sum(axis=(1,2,3),keepdims=True)
    means=joint.mean(axis=0)
    lo=np.quantile(joint,.025,axis=0)
    hi=np.quantile(joint,.975,axis=0)
    combos=[]
    for i,base in enumerate(BASES):
        for j,shade in enumerate(SHADES):
            for k,pied in enumerate((0,1)):
                p=float(means[i,j,k])
                if p <= minimum:continue
                label=(('薄' if shade=='薄' else '極薄' if shade=='極薄' else '')
                       +base+('pd' if pied else ''))
                # 薄墨 category is already a name; do not add "薄薄墨" in output.
                if base in ('薄墨','白','アルビノ') and shade != '通常':
                    label=f'{base}（濃淡: {shade}）'+('pd' if pied else '')
                combos.append(dict(label=label, base=base,shade=shade,pied=bool(pied),
                                   probability=round(p,5),ci95_low=round(float(lo[i,j,k]),5),
                                   ci95_high=round(float(hi[i,j,k]),5)))
    combos.sort(key=lambda x:x['probability'],reverse=True)
    # Marginal trait probabilities and 95% bands from *the same* posterior draws.
    marg={}
    for dim,labels in [('base',BASES),('shade',SHADES),('pied',('斑なし','斑あり'))]:
        axes={'base':(2,3),'shade':(1,3),'pied':(1,2)}[dim]
        arr=joint.sum(axis=axes)
        mu=arr.mean(axis=0); ll=np.quantile(arr,.025,axis=0);hh=np.quantile(arr,.975,axis=0)
        marg[dim]=[dict(label=str(lb),probability=round(float(mu[i]),5),
                        ci95_low=round(float(ll[i]),5),ci95_high=round(float(hh[i]),5))
                   for i,lb in enumerate(labels)]
    mor={}
    for label in MORPHOLOGY:
        glob=np.asarray(compound['morphology'][label],float)
        base=(glob+1)/(glob.sum()+2)
        parent_obs=int(label in (ftraits['morphology'] or ()))+int(label in (mtraits['morphology'] or ()))
        alpha=12*base+_get(compound['morphology_parent'],label+'|'+str(parent_obs),2)
        samp=rng.beta(alpha[1],alpha[0],size=draws)
        mor[label]=dict(probability=round(float(samp.mean()),5),
                        ci95_low=round(float(np.quantile(samp,.025)),5),
                        ci95_high=round(float(np.quantile(samp,.975)),5))
    # Conditions for insufficiency are indicators, not invented confidence levels.
    nobs=compound['counts']
    warnings=[]
    if nobs.get('n_base',0)<50 or nobs.get('n_shade',0)<50:
        warnings.append('条件に使える記録が少ないため、確率は強い事前分布に依存します。')
    if traits_f_missing(ftraits,mtraits):
        warnings.append('親の表現型のいずれかが不明・曖昧です。親の形質条件付けが弱くなります。')
    if not all(x['bp'].get('converged',False) for x in model['loci'].values()):
        warnings.append('遺伝子型の近似推論が収束していません。')
    warnings.append('信用区間は親の未知の遺伝子型とパラメータ推定を近似反映しますが、未同定の遺伝子や分類ミスの影響を含みません。')
    return dict(sire_no=str(sire_id),dam_no=str(dam_id),sex=sex,method='経験ベイズ階層Dirichlet+近似メンデル推論',
                draws=draws,components=marg,top_combinations=combos[:20],
                all_combinations=combos,morphology=mor,
                records_used=nobs,genetic_mutant_probability={x:round(y,5) for x,y in risks.items()},
                warnings=warnings)


def traits_f_missing(tf,tm):
    return tf['base'] is None or tm['base'] is None


def validate_compound(birds, fraction=.20):
    """Hold out *entire* latest mating combinations, avoiding sibling leakage.

    Parents from other matings may remain known. Input data do not certify
    external independence or stable recording protocols.
    """
    from genetics import LOCI,fit_locus
    groups=defaultdict(list)
    for i,b in enumerate(birds):
        if b['sire'] is not None and b['dob']:
            groups[(b['sire_no'],b['dam_no'])].append(i)
    if len(groups)<10:
        return dict(status='insufficient_mating_groups',n_groups=len(groups))
    group_keys=sorted(groups,key=lambda key:max(birds[i]['dob'] for i in groups[key]))
    selected=group_keys[-max(1,round(len(group_keys)*fraction)):]
    heldout={i for key in selected for i in groups[key]}
    train=np.array([i not in heldout for i in range(len(birds))],bool)
    fitted={x['locus']:fit_locus(birds,x,train) for x in LOCI}
    compound=fit_compound(birds,fitted,train)
    mm=dict(birds=birds,loci=fitted,compound=compound)
    # Evaluate joint base+shade+pied when all components were recorded.
    ys=[];probs=[];baseline=[]
    baseline_counts=Counter()
    for i,b in enumerate(birds):
        if train[i] and b['sire'] is not None:
            t=parse_traits(b['phenotype'])
            if t['base'] is not None and t['shade'] is not None and t['pied'] is not None:
                baseline_counts[(t['base'],t['shade'],t['pied'])]+=1
    total=sum(baseline_counts.values())
    cats=[(base,shade,pied) for base,shade,pied in product(BASES,SHADES,(0,1))
          if not ((base in ('白','アルビノ') and (shade != '通常' or pied==1))
                  or (base=='薄墨' and shade!='通常'))]
    base_probs=np.array([(baseline_counts[x]+.25)/(total+.25*len(cats)) for x in cats])
    from mendel import sex_transition
    for i in sorted(heldout):
        b=birds[i];t=parse_traits(b['phenotype'])
        if t['base'] is None or t['shade'] is None or t['pied'] is None:
            continue
        result=predict_compound(mm,birds[b['sire']]['parent_id'],birds[b['dam']]['parent_id'],b['sex'],draws=100,seed=2121)
        p=np.zeros(len(cats),float)
        for e in result['all_combinations']:
            key=(e['base'],e['shade'],int(e['pied']))
            p[cats.index(key)]=e['probability']
        p=np.clip(p,1e-8,1);p=p/p.sum()
        ys.append(cats.index((t['base'],t['shade'],t['pied'])))
        probs.append(p)
    if not ys:
        return dict(status='no_evaluable_test_rows',heldout_n=len(heldout))
    ps=np.array(probs);ys=np.array(ys)
    indices=np.arange(len(ys));truth=ps[indices,ys];baseline_truth=base_probs[ys]
    brier=float(np.mean(np.sum((ps-np.eye(len(cats))[ys])**2,axis=1)))
    base_brier=float(np.mean(np.sum((base_probs[None,:]-np.eye(len(cats))[ys])**2,axis=1)))
    return dict(status='evaluated',grouping='hold out entire parent pairs with latest offspring dates',
                heldout_mating_pairs=len(selected),heldout_rows=len(heldout),
                evaluated_rows=len(ys),
                n_categories=len(cats),
                multiclass_brier=round(brier,5),baseline_brier=round(base_brier,5),
                multiclass_logloss=round(float(-np.log(truth).mean()),5),
                baseline_logloss=round(float(-np.log(baseline_truth).mean()),5),
                accuracy_top1=round(float(np.mean(np.argmax(ps,axis=1)==ys)),5),
                baseline_accuracy_top1=round(float(np.mean(np.argmax(base_probs)==ys)),5),
                warning='同一個体が別の交配の親の場合、完全に独立した血統間検証ではない。区間の頻度論的被覆率とは別。')
