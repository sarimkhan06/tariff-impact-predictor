import pandas as pd

df = pd.read_csv('12100163.csv')  # load the StatCan trade table

print("COLUMNS:")
print(df.columns.tolist())  # what fields exist

print("\nROWS:", len(df))  # total row count

print("\nDATE RANGE:")
print(df['REF_DATE'].min(), "to", df['REF_DATE'].max())  # earliest/latest month

print("\nTRADE column:")
print(df['Trade'].unique())  # confirms Import vs Export split

napcs_col = 'North American Product Classification System (NAPCS)'  # shorthand for the long column name

print("\nNAPCS unique count:", df[napcs_col].nunique())  # how many commodity categories total

print("\nSearching for target commodities in NAPCS categories:")
for keyword in ['lumber', 'wood', 'steel', 'aluminum', 'dairy', 'iron']:
    # keep only category names containing this keyword (case-insensitive)
    matches = [c for c in df[napcs_col].unique() if keyword.lower() in str(c).lower()]
    print(f"\n'{keyword}':")
    for m in matches:
        print(" -", m)