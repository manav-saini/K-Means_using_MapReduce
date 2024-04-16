import subprocess
import signal

def run_mappers(num_mappers, num_reducers):
    processes = []

    for i in range(num_mappers):
        cmd = f"python mapper.py {i+1} {num_reducers}"
        print(f"Starting mapper {i+1} with command: {cmd}")
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append(process)

    try:
        print("Mappers are running. Press Ctrl+C to terminate them.")
        while True:
            continue
    except KeyboardInterrupt:
        print("Terminating mappers...")
        for i, process in enumerate(processes, start=1):
            process.terminate()
            try:
                output, error = process.communicate(timeout=10)
                print(f"Mapper {i} terminated. Output:\n{output.decode()}")
                if error:
                    print(f"Error from mapper {i}: {error.decode()}")
            except subprocess.TimeoutExpired:
                print(f"Mapper {i} did not terminate gracefully and was killed.")
                process.kill()

if __name__ == "__main__":
    try:
        num_mappers = int(input("Enter the number of mappers: "))
        num_reducers = int(input("Enter the number of reducers: "))
        run_mappers(num_mappers, num_reducers)
    except ValueError:
        print("Please enter valid integers for the number of mappers and reducers.")
    except Exception as e:
        print(f"An error occurred: {e}")
