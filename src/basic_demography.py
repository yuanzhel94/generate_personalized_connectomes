import numpy as np
import pandas as pd


if __name__ == "__main__":
    train_traits = pd.read_csv("/data/gpfs/projects/punim1278/projects/vaegm/data/SC_Schaefer7n200p/train_traits.csv", index_col=0)
    test_traits = pd.read_csv("/data/gpfs/projects/punim1278/projects/vaegm/data/SC_Schaefer7n200p/test_traits.csv", index_col=0)
    all_traits = pd.concat([train_traits, test_traits])
    print(f"train - age: {train_traits['age'].mean()}, std {train_traits['age'].std()}, sex: {train_traits['sex'].sum()}")
    print(f"test - age: {test_traits['age'].mean()}, std {test_traits['age'].std()}, sex: {test_traits['sex'].sum()}")
    print(f"all - age: {all_traits['age'].mean()}, std {all_traits['age'].std()}, sex: {all_traits['sex'].sum()}")