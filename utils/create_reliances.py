import pandas as pd
import numpy as np
import sys

def split_csv(file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path, sep=',')

    # Get lists of columns that start with "ACCURATEZZA" and "CONFIDENZA"
    risposteiniziali_cols = [col for col in df.columns if col.startswith('H')]
    confidenzainiziale_cols = [col for col in df.columns if col.startswith('C')]
    rispostefinali_cols = [col for col in df.columns if col.startswith('FH')]
    confidenzafinali_cols = [col for col in df.columns if col.startswith('FC')]

    # Create new DataFrames with only the desired columns
    df_risposte_h1 = df[risposteiniziali_cols]
    df_confidenza_h1 = df[confidenzainiziale_cols]
    df_risposte_fh = df[rispostefinali_cols]
    df_confidenza_fh = df[confidenzafinali_cols]

    # Write these DataFrames to new CSV files
    df_risposte_h1.to_csv('h1.csv', index=False, sep=',')
    df_confidenza_h1.to_csv('conf-h1.csv', index=False, sep=',')
    df_risposte_fh.to_csv('fh.csv', index=False, sep=',')
    df_confidenza_fh.to_csv('conf-fh.csv', index=False, sep=',')
    print("split_csv executed")

def compare_csvs(multiline_file_path, singleline_file_path, output_file_path):
    # Load the CSV files into pandas DataFrames
    df_multiline = pd.read_csv(multiline_file_path, sep=',', skiprows=1, header=None)
    df_singleline = pd.read_csv(singleline_file_path, sep=',', skiprows=1, header=None)
    
    print(df_multiline.columns)
    print(df_singleline.columns)

    # Read headers from multiline_file
    headers = pd.read_csv(multiline_file_path, sep=',', nrows=0).columns.tolist()

    # Ensure the single-line DataFrame is broadcastable to the shape of the multi-line DataFrame
    df_singleline = pd.concat([df_singleline]*len(df_multiline), ignore_index=True)

    # Check if cells in both dataframes are empty
    empty_cells = df_multiline.isnull() | df_singleline.isnull()

    # Compare the two DataFrames, considering empty cells
    df_comparison = (df_multiline == df_singleline).astype(int).where(~empty_cells, other=np.nan)

    # Prepend headers to df_comparison
    df_comparison.columns = headers

    # Write the comparison DataFrame to a new CSV file
    df_comparison.to_csv(output_file_path, index=False, sep=',')

    print("compare_csvs executed",multiline_file_path,singleline_file_path)

def create_reliances(h1_file_path, ai_file_path, fh_file_path, output_file_path, c1_file_path=None, fc_file_path=None ):
    # Load the CSV files into pandas DataFrames, skipping the first row (header)
    df_h1 = pd.read_csv(h1_file_path, sep=',', skiprows=1, header=None)
    df_ai = pd.read_csv(ai_file_path, sep=',', skiprows=1, header=None)
    df_fh = pd.read_csv(fh_file_path, sep=',', skiprows=1, header=None)

    df_c1 = None
    df_fc = None
    if c1_file_path is not None:
        try:
            df_c1 = pd.read_csv(c1_file_path, sep=',', skiprows=1, header=None)
        except:
            print("No C1")
    if fc_file_path is not None:
        try:
            df_fc = pd.read_csv(fc_file_path, sep=',', skiprows=1, header=None)
        except:
            print("No FC")
    

    # Create an empty DataFrame for the output
    df_reliances = pd.DataFrame(np.empty((df_h1.shape[0]*df_h1.shape[1], 6)), columns=['id','HD1', 'AI', 'FHD', "C1", "FC"])

    # Iterate over the rows and columns of df_h1
    for i in range(len(df_h1)):
        for j in range(len(df_h1.columns)):
            # Get the corresponding values from df_h1, df_ai, and df_fh
            df_reliances.loc[i*len(df_h1.columns) + j, "id"] = i
            df_reliances.loc[i*len(df_h1.columns) + j, "HD1"] = df_h1.iloc[i, j]
            df_reliances.loc[i*len(df_h1.columns) + j, "AI"] = df_ai.iloc[0, j]
            df_reliances.loc[i*len(df_h1.columns) + j, "FHD"] = df_fh.iloc[i, j]

            if df_c1 is not None:
                df_reliances.loc[i*len(df_h1.columns) + j, "C1"] = df_c1.iloc[i,j]
            if df_fc is not None:
                df_reliances.loc[i*len(df_h1.columns) + j, "FC"] = df_fc.iloc[i,j]

    # Write df_reliances to a new CSV file
    df_reliances = df_reliances.dropna()
    df_reliances = df_reliances.astype(int)
    df_reliances.to_csv(output_file_path, index=False, sep=',')
    
    print("create_reliances executed")
    return df_reliances
    
def main():
    print("Here")
    if len(sys.argv) != 4:
        print("Usage: python create_reliances.py responses.csv groundtruth.csv ai.csv")
        sys.exit(1)

    print("Here")
    groundtruth_file = sys.argv[2]
    ai_file = sys.argv[3]
    responses_file = sys.argv[1]
    ## Define the path to your CSV file
    file_path = responses_file  # replace with your file path
    ## Call the function
    split_csv("./" + file_path)



    basepath = './'

    # Define the paths to your CSV files
    multiline_file_path = basepath + 'h1.csv'  # replace with your file path
    singleline_file_path = basepath + ai_file  # replace with your file path

    # Define the path to the output CSV file
    output_file_path = basepath + 'agreementh1ai.csv'  # replace with your desired output file path

    # Call the function
    compare_csvs(multiline_file_path, singleline_file_path, output_file_path)

    multiline_file_path = basepath + 'h1.csv'  # replace with your file path
    singleline_file_path = basepath + groundtruth_file  # replace with your file path

    # Define the path to the output CSV file
    output_file_path = basepath + 'accuratezze-h1.csv'  # replace with your desired output file path

    # Call the function
    compare_csvs(multiline_file_path, singleline_file_path, output_file_path)

    multiline_file_path = basepath + ai_file  # replace with your file path
    singleline_file_path = basepath + groundtruth_file  # replace with your file path

    # Define the path to the output CSV file
    output_file_path = basepath + 'accuratezze-ai.csv'  # replace with your desired output file path

    # Call the function
    compare_csvs(multiline_file_path, singleline_file_path, output_file_path)

    multiline_file_path = basepath + 'fh.csv'  # replace with your file path
    singleline_file_path = basepath + groundtruth_file  # replace with your file path

    # Define the path to the output CSV file
    output_file_path = basepath + 'accuratezze-fh.csv'  # replace with your desired output file path

    # Call the function
    compare_csvs(multiline_file_path, singleline_file_path, output_file_path)



    # Define the paths to your CSV files
    h1_file_path = basepath + 'accuratezze-h1.csv'  # replace with your file path
    ai_file_path = basepath + 'accuratezze-ai.csv'  # replace with your file path
    fh_file_path = basepath + 'accuratezze-fh.csv'  # replace with your file path
    c1_file_path = basepath + 'conf-h1.csv'
    fc_file_path = basepath + 'conf-fh.csv'

    # Define the path to the output CSV file
    output_file_path = basepath + file_path + '_reliances.csv'  # replace with your desired output file path

    # Call the function
    create_reliances(h1_file_path, ai_file_path, fh_file_path, output_file_path, c1_file_path, fc_file_path)

if __name__=="__main__":
    main()