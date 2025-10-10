import subprocess
import time
import os

def get_project_root():
    """Get the root directory where scripts are located"""
    return os.path.dirname(os.path.abspath(__file__))

scripts = [
    ["Extractor.py"],
    ["Truth extractor.py"],
    ["Denoising.py"],
    ["Segmentaion.py"],
    ["IOU.py"],
    ["Grow rate and intensity.py", "--pipeline"]
]

for script in scripts:
    try:
        script_path = os.path.join(get_project_root(), script[0])
        print(f"Running {script_path}...")
        start_time = time.time()

        result = subprocess.run(
            ["python"] + [script_path] + script[1:],
            cwd=get_project_root(),  # Run from project root
            check=True,
            text=True,
            capture_output=True
        )

        print(result.stdout)
        elapsed = time.time() - start_time
        print(f"{script[0]} completed in {elapsed:.2f} seconds")

        time.sleep(1.0)  # 1 second pause between scripts

    except subprocess.CalledProcessError as e:
        print(f"Error in {script[0]}:")
        print(e.stderr)
        break

    except FileNotFoundError:
        print(f"{script[0]} not found. Check filename/spelling.")
        break

print("All scripts executed in order.")