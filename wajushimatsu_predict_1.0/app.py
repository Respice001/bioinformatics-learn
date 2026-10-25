"""和十姉妹の交配予測：外部飼育者向け公開Streamlitアプリ。

Run: streamlit run app.py
The application loads aggregate-only public_model.json; it does NOT load
individual breeding records or persist visitor submissions.
"""
from __future__ import annotations
import csv
from datetime import date
from io import StringIO
from pathlib import Path

import streamlit as st
import pandas as pd
from predictor import LOCI_LABEL, load_public_model, predict, trait

st.set_page_config(page_title='和十姉妹｜交配予測ラボ',page_icon='🐦',layout='wide',
                   initial_sidebar_state='collapsed')

st.markdown('''<style>
.block-container {max-width:1120px; padding-top:2.5rem; padding-bottom:4rem}
h1,h2,h3 {letter-spacing:.01em}
.hero {background:linear-gradient(120deg,#122d31,#205a5c);color:#f8f6ef;
 padding:2.1rem 2rem; border-radius:18px; margin-bottom:1.4rem}
.hero h1 {color:#f8f6ef;margin:0 0 .4rem 0;font-size:2.0rem}
.hero p {color:#e4eceb;margin:0;line-height:1.8;font-size:1rem}
.small-muted {color:#667678;font-size:.88rem}
.note {border-left:4px solid #b87934;background:rgba(184,121,52,.09);
 padding:.9rem 1.2rem;border-radius:5px;line-height:1.7}
div[data-testid="stMetric"] {background:rgba(50,119,112,.05);border:1px solid rgba(95,140,131,.2);
 border-radius:12px;padding:12px}
</style>''',unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def data():
    return load_public_model()

model=data()
validate=model.get('validation',{})

st.markdown('''<div class="hero"><h1>和十姉妹｜交配予測ラボ</h1>
<p>親鳥の外見と、わかれば祖父母の外見から、子の羽色・模様の出現確率を推定します。<br>
飼育記録に基づく統計的な予測です。<strong>遺伝子検査や将来の羽色を確定するものではありません。</strong></p></div>''',unsafe_allow_html=True)

st.markdown('''<div class="note"><strong>公開試験版の重要事項</strong>：95%信用区間は採用した遺伝モデルのもとでの推定幅です。
未知の血統に対する精度・区間の被覆率は未検証です。珍しい色や複雑な斑模様は特に不確実です。</div>''',unsafe_allow_html=True)

COLOR_CHOICES=['未確認','和黒','ハバナ','猩々','和パール','薄墨','ハバナグレー','白','アルビノ','その他']
SHADE_CHOICES=['未確認','通常','薄','極薄']
PIED_CHOICES=['未確認','斑あり','斑なし']
MORPH=['梵天','千代田','中納言','大納言','フリル']

def collect(prefix, title, *, main=True):
    st.markdown(f'**{title}**')
    cc=st.columns([1.5,1,1])
    with cc[0]: col=st.selectbox('基本色',COLOR_CHOICES,key=f'{prefix}base')
    with cc[1]: sh=st.selectbox('濃淡',SHADE_CHOICES,key=f'{prefix}shade',index=0)
    with cc[2]: pied=st.selectbox('パイド（斑）',PIED_CHOICES,key=f'{prefix}pied',index=0)
    if main:
        with st.expander('羽毛の形態（任意・補助予測）',expanded=False):
            mor=st.multiselect('観察された形態',MORPH,key=f'{prefix}morph')
    else:
        mor=()
    return trait(None if col=='未確認' else col,
                 None if sh=='未確認' else sh,
                 None if pied=='未確認' else (1 if pied=='斑あり' else 0),mor)

p1,p2,p3=st.tabs(['交配を予測する','解析手法・検証結果','科学的根拠・公開方針'])
with p1:
    st.subheader('親鳥の特徴を入力')
    st.caption('「未確認」は未知として扱います。白・アルビノでは濃淡やパイドが外見から隠れる可能性があるため、その情報を無理に推定しません。')
    with st.form('breeding'):
        cf,cm=st.columns(2,gap='large')
        with cf:
            st.markdown('#### 父鳥（♂）')
            father=collect('f','外見の特徴')
        with cm:
            st.markdown('#### 母鳥（♀）')
            mother=collect('m','外見の特徴')
        st.divider()
        use_anc=st.checkbox('祖父母の外見も入力して推定を更新する',value=False)
        gp={}
        if use_anc:
            st.caption('血統の上流情報から、父母が外見に現れない変異を保有している確率を更新します。情報を増やしても必ず区間が狭まるとは限りません。')
            with st.expander('父方の祖父母',expanded=True):
                g1,g2=st.columns(2,gap='large')
                with g1: gf_f=collect('ff','父鳥の父（祖父）',main=False)
                with g2: gf_m=collect('fm','父鳥の母（祖母）',main=False)
                gp['father']=(gf_f,gf_m)
            with st.expander('母方の祖父母',expanded=False):
                g1,g2=st.columns(2,gap='large')
                with g1: gm_f=collect('mf','母鳥の父（祖父）',main=False)
                with g2: gm_m=collect('mm','母鳥の母（祖母）',main=False)
                gp['mother']=(gm_f,gm_m)
        st.divider()
        opts=st.columns([2,1,1])
        with opts[0]:
            sex_text=st.selectbox('子の性別', ['指定しない（雄雌50:50）','雄（ZZ）','雌（ZW）'])
        with opts[1]:
            draws=st.select_slider('計算の細かさ', options=[200,500,1000,2000,4000],value=1000,
                                   help='モンテカルロ標本数。増やしてもモデルの構造的な誤りは解消しません。')
        with opts[2]:
            visible=st.selectbox('表に表示する候補', ['上位10件','すべて（40分類）'],index=0)
        submitted=st.form_submit_button('出現確率を計算する',use_container_width=True,type='primary')
    if submitted:
        sex={'指定しない（雄雌50:50）':'?','雄（ZZ）':'M','雌（ZW）':'F'}[sex_text]
        try:
            with st.spinner('遺伝子型の推定と複合羽色の事後予測を計算しています'):
                res=predict(model,father,mother,grandparents=gp if use_anc else None,sex=sex,draws=draws)
            st.session_state['prediction']=res
            st.session_state['submitted_traits']=(father,mother)
        except Exception as exc:
            st.error(f'計算に失敗しました：{exc}')
            st.session_state.pop('prediction',None)
    if 'prediction' in st.session_state:
        res=st.session_state['prediction']
        st.divider()
        st.subheader('予測結果')
        all_rows=res['all_combinations']
        best=all_rows[0]
        met=st.columns(3)
        met[0].metric('最も出やすい羽色（モデル上）',best['label'])
        met[1].metric('予測出現確率',f'{100*best["probability"]:.1f}%')
        met[2].metric('95%信用区間',f'{100*best["ci95_low"]:.1f}～{100*best["ci95_high"]:.1f}%')
        st.caption('この区間は「次の子が95%の確率でこの羽色になる」という意味ではありません。')
        rows=all_rows if visible.startswith('すべて') else all_rows[:10]
        simple=[{'羽色':r['label'], '予測確率 (%)':round(100*r['probability'],2),
                 '95%下限 (%)':round(100*r['ci95_low'],2),
                 '95%上限 (%)':round(100*r['ci95_high'],2),
                 '基本色':r['base'], '濃淡':r['shade'],'パイド':'あり' if r['pied'] else 'なし'} for r in rows]
        st.dataframe(pd.DataFrame(simple),hide_index=True,use_container_width=True,
                     column_config={'予測確率 (%)':st.column_config.ProgressColumn(
                         '予測確率 (%)',min_value=0,max_value=100,format='%.2f%%')})
        st.caption('確率上位の一覧です。低頻度の色や「その他」も含む40分類の合計は約100%になります。')
        st.subheader('予測確率の内訳')
        lt,rt=st.columns([3,2])
        with lt:
            bar=pd.DataFrame({'羽色':[x['label'] for x in all_rows[:8]],
                              '出現確率 (%)':[x['probability']*100 for x in all_rows[:8]]}).set_index('羽色')
            st.bar_chart(bar,horizontal=True)
        with rt:
            for tag,name in [('base','基本色'),('shade','濃淡'),('pied','パイド')]:
                vals=sorted(res['components'][tag],key=lambda x:x['probability'],reverse=True)
                st.markdown(f'**{name}**')
                st.caption('、'.join(f'{v["label"]} {v["probability"]*100:.1f}%' for v in vals[:4]))
        st.subheader('父母の推定遺伝子型')
        st.caption('これは遺伝子検査の結果ではありません。各行は仮定した1座位についての推定分布です。')
        genotype=[]
        for locus,item in res['genotype_posteriors'].items():
            is_z=item['mode']=='z_linked'
            for sex_id,caption in [('father','父鳥'),('mother','母鳥')]:
                arr=item[sex_id]
                genotype.append({'仮説座位':LOCI_LABEL.get(locus,locus),'個体':caption,
                                 '非保有 (%)':round(arr[0]*100,1),
                                 'ヘテロ接合 (%)':None if is_z and sex_id=='mother' else round(arr[1]*100,1),
                                 '変異型2コピー／雌Z変異 (%)':round(arr[2]*100,1),
                                 '変異対立遺伝子を保有 (%)':round((arr[1]+arr[2])*100,1)})
        st.dataframe(pd.DataFrame(genotype),hide_index=True,use_container_width=True)
        with st.expander('子の遺伝子型確率と補助形質',expanded=False):
            child_rows=[]
            for loc,obj in res['genotype_posteriors'].items():
                child_rows.append({'仮説座位':LOCI_LABEL.get(loc,loc),
                                   '基準型 (%)':round(100*obj['offspring'][0],1),
                                   'ヘテロ接合型 (%)':round(100*obj['offspring'][1],1),
                                   '変異型 (%)':round(100*obj['offspring'][2],1)})
            st.dataframe(pd.DataFrame(child_rows),use_container_width=True,hide_index=True)
            st.caption('性別を指定しない場合、Z連鎖座位の結果は雄ZZと雌ZWの混合です。')
            st.markdown('**梵天・フリル等（周辺予測。複合羽色との同時分布ではありません）**')
            st.dataframe(pd.DataFrame([
                {'形態形質':key,'予測確率 (%)':round(v['probability']*100,1),
                 '95%区間 (%)':f'{100*v["ci95_low"]:.1f}～{100*v["ci95_high"]:.1f}'}
                for key,v in res['morphology'].items()]),hide_index=True,use_container_width=True)
        st.markdown('**この結果の限界**')
        for warning in dict.fromkeys(res['warnings']):
            st.caption('・'+warning)
        buf=StringIO()
        writer=csv.DictWriter(buf,fieldnames=['羽色','基本色','濃淡','パイド','予測確率','95%下限','95%上限'])
        writer.writeheader()
        for r in all_rows:
            writer.writerow({'羽色':r['label'],'基本色':r['base'],'濃淡':r['shade'],
                             'パイド':'あり' if r['pied'] else 'なし',
                             '予測確率':r['probability'],'95%下限':r['ci95_low'],
                             '95%上限':r['ci95_high']})
        st.download_button('予測表をCSVで保存',data=('\ufeff'+buf.getvalue()).encode('utf-8'),
                           file_name='和十姉妹_予測結果.csv',mime='text/csv')
    else:
        st.info('父鳥と母鳥を入力し、「出現確率を計算する」を押してください。')

with p2:
    st.subheader('解析の設計')
    st.markdown('''**データ**：単一の飼育者のODS繁殖記録から推定した匿名集計パラメータを使用します。
    元ODS・個体ID・個体名・血統別記録はこの公開版に含まれません。''')
    c1,c2,c3=st.columns(3)
    c1.metric('学習元の登録個体',f'{model["counts"]["records"]}羽')
    c2.metric('親子関係つき記録',f'{model["counts"]["offspring"]}羽')
    c3.metric('交配組',f'{model["counts"]["mating_pairs"]}組')
    st.markdown('''**推定の流れ**

    1. 遺伝子座ごとの集団内対立遺伝子頻度を事前分布とする（常染色体はHardy–Weinberg型、Z連鎖は性別を考慮）。
    2. 親鳥の外見から発現・非発現マーカーの尤度を計算し、ベイズ更新する。
    3. 祖父母を入力した場合、祖父母の遺伝子型の不確実性をメンデル伝達で積分し、親鳥の遺伝子型分布を更新する。
    4. 子の遺伝子型についてメンデル分離を適用する。
    5. 基本色・濃淡・斑を学習データ由来の階層Dirichlet/Beta分布で予測する。
    6. 未知の親遺伝子型と一部のモデルパラメータをモンテカルロ抽出し、複合羽色の予測確率と95%信用区間を求める。

    **重要**：これらは特定DNA変異を特定した因果モデルではありません。複合形質は近似的に結合され、遺伝子座間の相関を完全には扱っていません。''')
    st.markdown('**各遺伝子座の観測情報**')
    loc=[]
    for name,info in model['loci'].items():
        comp=info['comparison']
        loc.append({'形質マーカー':LOCI_LABEL.get(name,name),'仮定':('Z連鎖' if info['mode']=='z_linked' else '常染色体')+'・'+comp['selected'],
                    '比較可能数':comp['n_informative'],'遺伝様式を識別可能か':('条件付き' if comp['identifiable'] else '資料不足')})
    st.dataframe(pd.DataFrame(loc),hide_index=True,use_container_width=True)
    st.markdown('**過去記録による交配組単位のホールドアウト検証**')
    if validate.get('status')=='evaluated':
        score=[
            {'指標':'多クラスBrier（低いほど良い）','統計モデル':validate['multiclass_brier'],'単純な出現率モデル':validate['baseline_brier']},
            {'指標':'対数損失（低いほど良い）','統計モデル':validate['multiclass_logloss'],'単純な出現率モデル':validate['baseline_logloss']},
            {'指標':'Top-1正解率（高いほど良い）','統計モデル':validate['accuracy_top1'],'単純な出現率モデル':validate['baseline_accuracy_top1']}]
        st.dataframe(pd.DataFrame(score),hide_index=True,use_container_width=True)
        st.caption(f'直近の親ペア {validate["heldout_mating_pairs"]}組、評価可能 {validate["evaluated_rows"]}羽、{validate["n_categories"]}分類。')
        st.warning('Brier・対数損失は改善しましたが、Top-1正解率はベースラインより低い結果です。外部血統や95%信用区間の被覆率はまだ検証していません。')
    else:
        st.warning('このバージョンの検証結果は未登録・未実施です。')
    st.markdown('''**バイアス・構造上の限界**

    - 1飼育群から推定した集団頻度を、別の系統の鳥の初期分布に代用しています。
    - 外見による羽色判定の標準化・誤分類、白色によるマスキング、優性パイドと劣性パイドの区別は不十分です。
    - 記録された遺伝子型は外見から推定された可能性があり、DNA検査で検証した型とは区別が必要です。
    - 学習元と検証元に個体・血統の重複が残り得ます。生存・記録されなかった雛の選択バイアスも未補正です。
    - 信用区間は**計算上の条件付き区間**であり、外部データに対する校正済みの信頼区間ではありません。
    - 表現型の組み合わせが学習集団の範囲外であれば、確率数値は特に過信できません。''')

with p3:
    st.subheader('科学的根拠と推論の限界')
    st.markdown('''**確立している一般原理**

    - メンデル分離と家系からの確率的遺伝子型推定は、統計遺伝学の標準的な枠組みです。
    - 鳥類では一般に雄がZZ、雌がZWです。Z染色体連鎖の表現は性別で異なります。
    - ベイズ推定では、観測情報と事前分布を合わせて未観測の遺伝子型の確率を更新できます。

    **このアプリの暫定的な仮定**

    - `rpd`、`ch`、`fwn` を常染色体の2対立遺伝子座、`pl` をZ染色体連鎖として近似します。
    - これらの表記を、原因配列が確定した変異の同定結果として扱いません。
    - 希釈・混合色・斑の出現は部分プーリングされた経験的確率分布で近似します。

    **主要参考文献**

    1. Elston RC, Stewart J. *A General Model for the Genetic Analysis of Pedigree Data*. Human Heredity (1971). [doi:10.1159/000152448](https://doi.org/10.1159/000152448) — 血統・表現型・メンデル伝達の確率モデル。
    2. Elston RC, Stewart J. *The Analysis of Quantitative Traits for Simple Genetic Models from Parental, F1 and Backcross Data*. Genetics (1973). [doi:10.1093/genetics/73.4.695](https://doi.org/10.1093/genetics/73.4.695) — 親子記録による遺伝仮説の比較。
    3. Major AT, Smith CA. *Sex Reversal in Birds*. Sexual Development (2016). [doi:10.1159/000448365](https://doi.org/10.1159/000448365) — 鳥のZZ/ZW性染色体。
    4. *The genomics of the domestication syndrome in a songbird model species*. Communications Biology (2025). [doi:10.1038/s42003-025-08235-0](https://doi.org/10.1038/s42003-025-08235-0) — ジュウシマツの家畜化と色素関連ゲノム領域。**今回の和十姉妹の羽色遺伝子を直接同定した研究ではありません。**

    参考文献は方法論や生物学的背景を支持するもので、アプリが設定した各変異の位置や適用妥当性を個別に証明するものではありません。''')
    st.subheader('プライバシーと公開')
    st.markdown('''- 公開版には**元のODS、個体番号、飼い主の氏名、繁殖記録の行データは含まれません**。
    - 公開アプリのPythonコードは入力内容をファイル・データベースに永続保存しません。
    - オンライン公開時は入力値がアプリの実行サーバーへ送信されます。ホスティング事業者のログ・プライバシー条件は別途確認してください。
    - 学術研究上の根拠が確立した遺伝子検査サービスではなく、**予測的な参考ツール**として提供します。
    - 育種・交配における健康面、近親交配、飼育福祉、疾患の危険性は、この羽色モデルでは評価していません。''')
    st.caption(f'公開モデル作成日：{model.get("export_date","不明")} ｜ 公開モデルのスキーマ：{model["schema_version"]}')
