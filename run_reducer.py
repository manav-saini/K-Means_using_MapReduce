import sys
import subprocess

def run_reducers(num_mapper,num_reducers):
    processes = []

    for i in range(num_reducers):
        cmd = f"python reducer.py {i+1} {num_mapper}"
        print(f"Starting reducer {i+1} with command: {cmd}")
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append(process)

    try:
        print("Reducers are running. Press Ctrl+C to terminate them.")
        while True:
            continue
    except KeyboardInterrupt:
        print("Terminating Reducer...")
        for i, process in enumerate(processes, start=1):
            process.terminate()
            try:
                output, error = process.communicate(timeout=10)
                print(f"Reducer {i+1} terminated. Output:\n{output.decode()}")
                if error:
                    print(f"Error from Reducer {i+1}: {error.decode()}")
            except subprocess.TimeoutExpired:
                print(f"Reduce {i+1} did not terminate gracefully and was killed.")
                process.kill()

if __name__ == "__main__":
    try:
        num_mapper = int(input("Enter the number of mappers: "))
        num_reducers = int(input("Enter the number of reducers: "))
        run_reducers(num_mapper,num_reducers)
    except ValueError:
        print("Please enter valid integers for the number of reducers.")
    except Exception as e:
        print(f"An error occurred: {e}")