import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

def export_train_test_data(df, train_file, test_file, test_size=0.2, random_state=42):
    """
    Splits data into training and testing sets while maintaining the proportions of data from each original dataset.

    Args:
        df: Combined DataFrame containing data from both datasets.
        train_file: Path to the CSV file for the training data.
        test_file: Path to the CSV file for the testing data.
        test_size: Proportion of the dataset to include in the test split (default: 0.2).
        random_state: Controls the shuffling applied to the data before applying the split (default: 42).
    """

    train_dfs = []
    test_dfs = []

    # Separate the dataframes from both datasets
    df1 = df[df['source'] == 'data1'].copy()
    df2 = df[df['source'] == 'data2'].copy()

    # Split each DataFrame separately
    train_df1, test_df1 = train_test_split(df1, test_size=test_size, random_state=random_state, stratify=df1['Label'])
    train_dfs.append(train_df1)
    test_dfs.append(test_df1)

    train_df2, test_df2 = train_test_split(df2, test_size=test_size, random_state=random_state, stratify=df2['Label'])
    train_dfs.append(train_df2)
    test_dfs.append(test_df2)

    # Concatenate the training and testing DataFrames
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    # Print the distribution of labels in the train and test sets
    print("Train label distribution:", Counter(train_df['Label']))
    print("Test label distribution:", Counter(test_df['Label']))

    # Export the DataFrames to CSV files with error handling
    try:
        train_df.to_csv(train_file, index=False)
        print(f"Training data saved to: {train_file}")
    except Exception as e:
        print(f"Error saving training data to {train_file}: {e}")

    try:
        test_df.to_csv(test_file, index=False)
        print(f"Testing data saved to: {test_file}")
    except Exception as e:
        print(f"Error saving testing data to {test_file}: {e}")


if __name__ == "__main__":
    train_path_1 = "cleaned_labeled_data_1.csv"  # Your file with 3 labels
    train_path_2 = "cleaned_labeled_data_2.csv"  # Your file with 2 labels

    # Read the data from both CSV files
    df1 = pd.read_csv(train_path_1)
    df2 = pd.read_csv(train_path_2)

    # Add a 'source' column to distinguish between the datasets after concatenation
    df1['source'] = 'data1'
    df2['source'] = 'data2'

    # Rename columns in df2 to match df1
    df2 = df2.rename(columns={'Content': 'Content', 'Label': 'Label'})

    # Concatenate the DataFrames
    df = pd.concat([df1, df2], ignore_index=True)

    # Define output file paths
    train_output_file = "train_data.csv"
    test_output_file = "test_data.csv"

    # Export the data to train and test files
    export_train_test_data(df, train_output_file, test_output_file)