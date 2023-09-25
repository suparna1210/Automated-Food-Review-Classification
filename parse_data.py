import time
import nltk
import spacy
import pandas as pd
import numpy as np

# Import data
data = pd.read_csv('data/train.csv')

# Clear null values
data = data.dropna()

# Calculate the helpfulness
data['Helpfulness'] = data['HelpfulnessNumerator']/data['HelpfulnessDenominator']
# For scores where the denominator was 0, replace NaN with zeros
data['Helpfulness'] = data['Helpfulness'].fillna(0)
data[['HelpfulnessNumerator', 'HelpfulnessDenominator', 'Helpfulness']].head()

# Remove all entries (actually just one) where the helpfulness score is above 1.
# This is because the helpfulness score cannot ever be over 1.
data = data[data.Helpfulness <= 1]

# Drop duplicate reviews
data.drop_duplicates(subset='Text', keep='first', inplace=True)

# Export to CSV
data.to_csv('data/cleaned_data.csv')