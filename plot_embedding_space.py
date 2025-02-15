import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_csv_rows(csv_path_1, csv_path_2, amount_to_plot=1):
    # Load the CSV files into DataFrames
    df1 = pd.read_csv(csv_path_1, header=None)
    df2 = pd.read_csv(csv_path_2, header=None)

    # Plot Figure 1
    plt.figure(figsize=(10, 6))
    for i in range(df1.shape[0]):  # Iterate over rows
        if i < amount_to_plot:
          plt.plot(df1.columns, df1.iloc[i], "b", label="BMR")
    for i in range(df2.shape[0]):  # Iterate over rows
      if i < amount_to_plot:
        plt.plot(df2.columns, df2.iloc[i], "r", label="noise")
    plt.title("Embedding space of BMR and noise")
    plt.xlabel("Columns")
    plt.ylabel("Values")
    plt.legend()
    plt.grid()
    plt.show()

path_to_embeddings = "Embeddings"

csv_file_1 = os.path.join(path_to_embeddings, "R1S1_embeddings_ch0_label1.csv")   # Make sure this file is BMR
csv_file_2 = os.path.join(path_to_embeddings, "R1S1_embeddings_ch0_label2.csv")   # Make sure this file is noise

plot_csv_rows(csv_file_1, csv_file_2, 1)