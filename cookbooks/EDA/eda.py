#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/chayachandana/Desktop/combined_breakfast_lunch.csv")

cols = {c.lower().strip(): c for c in df.columns}
def has(col): return col in cols

if has('school_id'):
    school_key = cols['school_id']
elif has('mapped_normalized_name'):
    school_key = cols['mapped_normalized_name']
elif has('school_name'):
    school_key = cols['school_name']
else:
    school_key = None

candidates = [c for c in [
    cols.get('school_id'),
    cols.get('mapped_normalized_name'),
    cols.get('school_name'),
    cols.get('normalized_school_name_csv'),
    cols.get('normalized_school_name_excel'),
] if c in df.columns]

unique_counts = {c: df[c].nunique(dropna=True) for c in candidates}
print("Unique school counts by column:", unique_counts)

if school_key:
    n_schools = df[school_key].nunique(dropna=True)
    print("Recommended distinct schools:", school_key, "=", n_schools)
else:
    print("No canonical school key found; see unique_counts for options.")

#%%
served_col = cols.get('served_total')
if served_col and school_key:
    active = (df[served_col].fillna(0) > 0)
    n_active = df.loc[active, school_key].nunique(dropna=True)
    print("Active schools (served_total>0 at least once):", n_active)

#%%
cep_col = cols.get('cep_schools')
region_col = cols.get('fcps_region')

if school_key and cep_col:
    by_cep = (df[[school_key, cep_col]].dropna().drop_duplicates()
              .groupby(cep_col)[school_key].nunique().sort_values(ascending=False))
    print("Distinct schools by CEP:\n", by_cep)

if school_key and region_col:
    by_region = (df[[school_key, region_col]].dropna().drop_duplicates()
                 .groupby(region_col)[school_key].nunique().sort_values(ascending=False))
    print("Distinct schools by region:\n", by_region)

#%%
df.columns = (df.columns
              .str.strip().str.lower()
              .str.replace(r'[^a-z0-9]+', '_', regex=True)
              .str.strip('_'))

school_col = None
for cand in ['mapped_normalized_name', 'normalized_school_name_csv',
             'normalized_school_name_excel', 'school_name']:
    if cand in df.columns:
        school_col = cand
        break
assert school_col is not None, "No school name column found."

assert 'fcps_region' in df.columns, "Missing fcps_region column."
reg = df['fcps_region'].astype(str)
reg_num = reg.str.extract(r'(\d+)')[0]
df['region_norm'] = np.where(reg_num.notna(), 'Region ' + reg_num, reg)

pairs = (df[['region_norm', school_col]]
         .dropna()
         .drop_duplicates())

region_to_schools = (pairs.groupby('region_norm')[school_col]
                          .apply(lambda s: sorted(s.unique()))
                          .to_dict())

for r in sorted(region_to_schools.keys(), key=lambda x: (x!='Region 1', x!='Region 2', x!='Region 3', x!='Region 4', x!='Region 5', x!='Region 6', x)):
    print(f"{r}:")
    for name in region_to_schools[r]:
        print("  -", name)

for r, schools in region_to_schools.items():
    out = pd.Series(schools, name='school')
    out.to_csv(f"schools_{r.replace(' ', '_').lower()}.csv", index=False)
#%%
sns.set_theme(context="notebook", style="whitegrid")

counts = (pairs.groupby('region_norm')[school_col]
               .nunique()
               .reset_index(name='n_schools'))

desired = [f"Region {i}" for i in range(1,7)]
order = [r for r in desired if r in counts['region_norm'].unique()] + \
        [r for r in counts['region_norm'].unique() if r not in desired]

plt.figure(figsize=(8,4))
sns.barplot(data=counts, x='region_norm', y='n_schools', order=order)
plt.title('Distinct schools by region')
plt.xlabel('Region')
plt.ylabel('Number of schools')
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

#%%
presence = (pairs.assign(val=1)
                 .pivot_table(index=school_col, columns='region_norm',
                              values='val', aggfunc='max', fill_value=0))

cols = [r for r in desired if r in presence.columns] + \
       [c for c in presence.columns if c not in desired]
presence = presence[cols]

height = min(0.25 * len(presence.index), 20)  
plt.figure(figsize=(10, height if height >= 4 else 4))
sns.heatmap(presence, cmap='Greens', cbar=False)
plt.title('School membership by region (1 = present)')
plt.xlabel('Region')
plt.ylabel('School')
plt.tight_layout()
plt.show()

#%%
df.columns = df.columns.str.strip().str.lower().str.replace(r'[^a-z0-9]+', '_', regex=True).str.strip('_')

school_key = None
for cand in ['school_id', 'mapped_normalized_name', 'school_name', 'normalized_school_name_csv']:
    if cand in df.columns:
        school_key = cand
        break

assert school_key is not None, "No school identifier column found."
assert 'served_total' in df.columns, "No served_total column found."

inactive_schools = (
    df.groupby(school_key, dropna=False)['served_total']
      .sum(min_count=1)
      .reset_index()
      .query('served_total == 0')
)

print("Schools with sum(served_total) == 0:")
print(inactive_schools[school_key].tolist())

#%%
zero_served_ids = [106, 163, 200, 214, 231, 377, 407]
zero_served_schools = df[df['school_id'].isin(zero_served_ids)]
school_names = zero_served_schools['mapped_normalized_name'].dropna().unique()

print("School names with zero served_total sum:")
for name in school_names:
    print(name)
# %%
df.groupby('date')['left_over_total'].sum().plot()
plt.title("Total Leftover Food Over Time")
plt.show()

#%%
df.nlargest(10, 'left_over_total')[['date','school_id','name','left_over_total']]

#%%
school_id = 390 
df[df['school_id'] == school_id].groupby('date')['left_over_total'].sum().plot()
plt.title(f"Leftover Food Over Time for School {school_id}")
plt.show()

#%%
df['date'] = pd.to_datetime(df['date'])
daily_stats = (
    df.groupby('date')[['left_over_total', 'served_total']]
    .sum()
    .reset_index()
)

plt.figure(figsize=(12,6))
plt.plot(daily_stats['date'], daily_stats['left_over_total'], label='Leftover', marker='o')
plt.plot(daily_stats['date'], daily_stats['served_total'], label='Served', marker='o')
plt.legend()
plt.title('Served Food vs. Leftover Food Over Time')
plt.ylabel('Number of Portions')
plt.xlabel('Date')
plt.show()

#%%
daily_stats['leftover_to_served'] = daily_stats['left_over_total'] / daily_stats['served_total']
print(daily_stats[['date', 'left_over_total', 'served_total']].head(10))

#%%
df['date'] = pd.to_datetime(df['date'])
daily = (
    df.groupby('date')[['offered_total', 'served_total', 'left_over_total']]
    .sum()
    .reset_index()
)

plt.figure(figsize=(12,6))
plt.plot(daily['date'], daily['offered_total'], label='Offered Total', marker='o')
plt.plot(daily['date'], daily['served_total'], label='Served Total', marker='x')
plt.plot(daily['date'], daily['left_over_total'], label='Leftover Total', marker='s')
plt.title('Offered, Served, and Leftover Food Over Time')
plt.xlabel('Date')
plt.ylabel('count')
plt.legend()
plt.tight_layout()
plt.show()

#%%
daily['percent_served'] = 100 * daily['served_total'] / daily['offered_total']
daily['percent_leftover'] = 100 * daily['left_over_total'] / daily['offered_total']

print(daily[['date', 'offered_total', 'served_total', 'left_over_total', 'percent_served', 'percent_leftover']].head())

#%%
