import pandas as pd
import os

def load_data():
    """Load the dataset from a CSV file"""
    df = pd.read_csv("data/mxmh_survey_results.csv")
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df
    
def clean_data(df):
    """CLean the dataset"""  
    #drop useless columns for ml
    df = df.drop(columns=['Timestamp', 'Permissions'])
    #drop rows with missing values in 'Music effects'
    df = df.dropna(subset=['Music effects'])
    #fill missing values in 'BPM' with the median
    df['BPM'] = df['BPM'].fillna(df['BPM'].median())
    #fill missing values in 'Age', 'Primary streaming service', 'While working', 'Instrumentalist', 'Composer', 'Foreign languages' with the mode
    for col in ['Age', 'Primary streaming service', 'While working', 
                    'Instrumentalist', 'Composer', 'Foreign languages']:
            df[col] = df[col].fillna(df[col].mode()[0])
    #remove outliers, nobody listens music 24h per day    
    df = df[df['Hours per day'] <= 16]
    print(f"After cleaning: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def encode_frequencies(df):
    """Convert frequency text columns to ordered numbers"""
    frequency_map = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Very frequently': 3}
    
    freq_cols = [col for col in df.columns if col.startswith('Frequency')]
    for col in freq_cols:
        df[col] = df[col].map(frequency_map)
    print(f"Encoded {len(freq_cols)} frequency columns")
    return df

def encode_categoricals(df):
    """Encode categorical columns into numbers"""
    # Binary columns - simple Yes/No map
    binary_map = {'Yes': 1, 'No': 0}
    binary_cols = ['While working', 'Instrumentalist', 'Composer', 'Exploratory', 'Foreign languages']
    
    for col in binary_cols:
        df[col] = df[col].map(binary_map)

    # One-hot encode multi-value columns
    df = pd.get_dummies(df, columns=['Fav genre', 'Primary streaming service'])

    print(f"After encoding: {df.shape[1]} columns")
    return df

def create_features(df):
    """Create new features and encode the target variable."""
    # Encode target variable
    target_map = {'Worsen': 0, 'No effect': 1, 'Improve': 2}
    df['Music effects'] = df['Music effects'].map(target_map)

    # Convert bool columns to int
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    #feature1: overall mental health score 0-40 scale
    df['mental_health_score'] = df['Anxiety'] + df['Depression'] + df['Insomnia'] + df['OCD']
    
    #feature2: age group
    df['age_group'] = pd.cut(df['Age'], bins=[0, 18, 25, 35, 100], labels=[0, 1, 2, 3])
    df['age_group'] = df['age_group'].astype(int)

    #feature3: music engagement score
    df['music_engagement'] = df['Hours per day'] + df['While working'] + df['Exploratory']

    print(f"Final dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nTarget variable distribution:\n{df['Music effects'].value_counts()}")
    return df


if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    df = encode_frequencies(df)
    df = encode_categoricals(df)
    df = create_features(df)
    print(df[['mental_health_score', 'age_group', 'music_engagement', 'Music effects']].head(10))