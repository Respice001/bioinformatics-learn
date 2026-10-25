"""Simple terminal demo; does not require Streamlit or pandas."""
from predictor import load_public_model,trait,predict
model=load_public_model()
dad=trait('和黒','通常',0)
mom=trait('ハバナ','通常',0)
res=predict(model,dad,mom,grandparents={'father':(trait('ハバナ'),trait('和黒'))},draws=1000)
print('【仮想交配】父：和黒、母：ハバナ、父方祖父：ハバナ')
for row in res['top_combinations'][:10]:
    print(f'{row["label"]:<15} {100*row["probability"]:5.1f}%（モデル条件付き95%信用区間：{100*row["ci95_low"]:5.1f}～{100*row["ci95_high"]:5.1f}%）')
