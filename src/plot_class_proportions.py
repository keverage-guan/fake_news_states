import pandas as pd, matplotlib.pyplot as plt, os
dfs = []
for f in ['data/splits/Multi_train.tsv'] + [f'data/splits/Multi_test{i}.tsv' for i in range(1,6)]:
    dfs.append(pd.read_csv(f, sep='\t', usecols=['created_utc','6_way_label']))
df = pd.concat(dfs)
df['month'] = pd.to_datetime(df['created_utc'], unit='s').dt.to_period('M')
monthly = df.groupby(['month','6_way_label']).size().unstack(fill_value=0)
monthly_pct = monthly.div(monthly.sum(axis=1), axis=0)
monthly_pct.plot(figsize=(14,5), title='Monthly 6-way class proportions')
plt.savefig('plots/monthly_6way.png', dpi=120, bbox_inches='tight')