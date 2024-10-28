import pandas as pd
import numpy as np

real = pd.read_csv("./fakeNewsNet_dataset/gossipcop_real.csv")
fake = pd.read_csv("./fakeNewsNet_dataset/gossipcop_fake.csv")

fake = fake[['title']].copy()
fake['label'] = 1  # 1 for fake news

real = real[['title']].copy()
real['label'] = 0  # 0 for real news

# Combine both datasets
combined_df = pd.concat([fake, real], ignore_index=True)

# Shuffle the combined dataset to mix fake and real news
combined_df = combined_df.sample(frac=1).reset_index(drop=True)
print(combined_df['label'].value_counts())

combined_df.to_csv("final_dataset.csv")