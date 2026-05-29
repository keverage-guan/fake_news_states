import pandas as pd

splits = {
    'train': 'data/splits/Multi_train.tsv',
    'test1': 'data/splits/Multi_test1.tsv',
    'test2': 'data/splits/Multi_test2.tsv',
    'test3': 'data/splits/Multi_test3.tsv',
    'test4': 'data/splits/Multi_test4.tsv',
    'test5': 'data/splits/Multi_test5.tsv',
}

dfs = {name: pd.read_csv(path, sep='\t', usecols=['subreddit', '6_way_label'])
       for name, path in splits.items()}

# --- Part 1: which subreddits drive cls4 in training? ---
cls4_train = dfs['train'][dfs['train']['6_way_label'] == 4]
print("=== Top cls4 subreddits in Multi_train ===")
print(cls4_train['subreddit'].value_counts().head(20))
print(f"\nTotal cls4 train: {len(cls4_train):,}")

# --- Part 2: cls4 subreddit counts across ALL splits ---
print("\n=== cls4 subreddit counts by split ===")
top_subs = cls4_train['subreddit'].value_counts().head(10).index.tolist()

rows = []
for split_name, df in dfs.items():
    cls4 = df[df['6_way_label'] == 4]
    total = len(cls4)
    row = {'split': split_name, 'total_cls4': total}
    for sub in top_subs:
        row[sub] = (cls4['subreddit'] == sub).sum()
    rows.append(row)

summary = pd.DataFrame(rows).set_index('split')
print(summary.to_string())

# --- Part 3: which subreddits appear in train but NOT in test windows? ---
print("\n=== Subreddits present in cls4 train but absent/sparse in tests ===")
train_subs = set(cls4_train['subreddit'].unique())
for split_name in ['test1','test2','test3','test4','test5']:
    cls4_test = dfs[split_name][dfs[split_name]['6_way_label'] == 4]
    test_subs = set(cls4_test['subreddit'].unique())
    missing = train_subs - test_subs
    print(f"\n  {split_name} — {len(missing)} train subreddits absent:")
    # Show only ones with meaningful training volume
    sig = cls4_train[cls4_train['subreddit'].isin(missing)]['subreddit'].value_counts().head(10)
    print(sig.to_string())