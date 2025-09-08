import os

print(f"Current Working Directory is: {os.getcwd()}")
print("-" * 50)

# --- Check 1: Does the 'data' folder exist? ---
data_folder_path = 'data'
print(f"Checking for folder: '{data_folder_path}'")
if os.path.isdir(data_folder_path):
    print(f"✅ SUCCESS: The '{data_folder_path}' directory was found.")
else:
    print(f"❌ FAILURE: The '{data_folder_path}' directory was NOT found.")
    print("ACTION: Please ensure a 'data' folder exists in your main project directory.")
    exit() # Stop the script if 'data' doesn't exist

print("-" * 50)

# --- Check 2: Does the 'data/processed' folder exist? ---
processed_folder_path = os.path.join('data', 'processed')
print(f"Checking for folder: '{processed_folder_path}'")
if os.path.isdir(processed_folder_path):
    print(f"✅ SUCCESS: The '{processed_folder_path}' directory was found.")
else:
    print(f"❌ FAILURE: The '{processed_folder_path}' directory was NOT found.")
    print("ACTION: Please manually create the 'processed' folder inside the 'data' folder.")
    exit() # Stop the script if 'processed' doesn't exist

print("-" * 50)

# --- Check 3: Can we write a file to the 'data/processed' folder? ---
test_file_path = os.path.join(processed_folder_path, 'test_write.txt')
print(f"Attempting to write a test file to: '{test_file_path}'")
try:
    with open(test_file_path, 'w') as f:
        f.write('This is a test.')
    print(f"✅ SUCCESS: A test file was written successfully.")
    
    # Clean up the test file
    os.remove(test_file_path)
    print("Cleaned up the test file.")
    
except Exception as e:
    print(f"❌ FAILURE: Could not write to the '{processed_folder_path}' directory.")
    print(f"This could be a permissions issue. Error: {e}")

print("-" * 50)
print("Test complete.")