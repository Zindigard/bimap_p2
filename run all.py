import subprocess
import time

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
        print(f"Running {' '.join(script)}...")
        start_time = time.time()


        result = subprocess.run(
            ["python"] + script,
            check=True,
            text=True,
            capture_output=True
        )


        print(result.stdout)
        elapsed = time.time() - start_time
        print(f"{' '.join(script)} completed in {elapsed:.2f} seconds")


        time.sleep(1.0)  # 1 second pause between scripts

    except subprocess.CalledProcessError as e:
        print(f"Error in {' '.join(script)}:")
        print(e.stderr)
        break

    except FileNotFoundError:
        print(f"{' '.join(script)} not found. Check filename/spelling.")
        break

print("All scripts executed in order.")