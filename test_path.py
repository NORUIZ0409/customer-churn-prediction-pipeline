import os
import pandas as pd

# Define the exact relative path your main script is using
file_path = 'data/raw/ecommerce_churn_data.csv'

# Get the full, absolute path that Python is trying to access
full_path = os.path.abspath(file_path)

print(f"Current Working Directory is: {os.getcwd()}")
print(f"Checking for file at absolute path: {full_path}")

# Check if the file exists at that path
if os.path.exists(full_path):
    print("\n✅ SUCCESS: File was found by Python!")
    try:
        df = pd.read_csv(full_path)
        print("✅ SUCCESS: File was also read successfully by pandas.")
    except Exception as e:
        print(f"\n❌ ERROR: File was found, but pandas could not read it. Error: {e}")
else:
    print("\n❌ FAILURE: File was NOT found by Python at that path.")
    print("ACTION: Please check for hidden extensions (see Investigation 1) or typos.")