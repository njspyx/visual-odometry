import subprocess

def run_script(script_name):
    print(f"Running {script_name}...")
    subprocess.run(["python", script_name])
    print(f"Finished running {script_name}.\n")

if __name__ == "__main__":
    scripts = ["preprocessor.py", "keypt_matching.py", "main.py"]
    
    for script in scripts:
        run_script(script)
    
    print("All scripts executed successfully.")