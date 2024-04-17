import grpc
import sys
from concurrent import futures
import numpy as np
import random
import mapreduce_pb2
import mapreduce_pb2_grpc
import concurrent.futures
import logging
import time

logging.basicConfig(filename='dump.txt', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)

class MasterServicer(mapreduce_pb2_grpc.MapReduceServiceServicer):
    def __init__(self, num_mappers, num_reducers, data_source, num_centroids, iterations, scenario, base_port=5000):
        self.num_mappers = num_mappers
        self.num_reducers = num_reducers
        self.data_source = data_source
        self.num_centroids = num_centroids
        self.iterations = iterations
        self.scenario = scenario
        self.centroids = self.initialize_centroids()
        self.base_port = base_port
        print(f"Initial centroids: {self.centroids}")
        logging.info("Initial centroids: %s", self.centroids)

    def initialize_centroids(self):
        data_points = self.load_data()
        if data_points.size == 0:
            raise ValueError("No data points loaded; check data source")
        data_points_list = [tuple(row) for row in data_points]
        if len(data_points_list) < self.num_centroids:
            raise ValueError(f"Requesting more centroids ({self.num_centroids}) than available data points ({len(data_points_list)})")
        return random.sample(data_points_list, self.num_centroids)

    def load_data(self):
        if self.scenario == 1:
            data = np.loadtxt(self.data_source, delimiter=',')
        else:
            data = np.concatenate([np.loadtxt(f, delimiter=',') for f in self.data_source])
        return data

    def map_reduce_cycle(self, iteration):
        print(f"Starting iteration {iteration}")
        data_points = self.load_data()
        if self.scenario == 1:
            indices = np.array_split(range(len(data_points)), self.num_mappers)
            chunks = [data_points[idx] for idx in indices]
        else:
            chunks = np.array_split(data_points, self.num_mappers)
        mapped = []
        reduced = []

        def map_chunk(chunk, mapper_id):
            port = self.base_port + mapper_id
            channel = grpc.insecure_channel(f'localhost:{port}')
            mapper_stub = mapreduce_pb2_grpc.MapReduceServiceStub(channel)
            point_messages = [mapreduce_pb2.Point(x=point[0], y=point[1]) for point in chunk]
            centroid_messages = [mapreduce_pb2.Centroid(id=i+1, location=mapreduce_pb2.Point(x=centroid[0], y=centroid[1])) for i, centroid in enumerate(self.centroids)]

            #Simulated probabilistic failure
            if random.random() < 0.5:
                print(f"Mapper {mapper_id} call FAILED due to simulated error")
                logging.info(f"Iteration {iteration}: Mapper {mapper_id} simulated failure")
                time.sleep(2)  # To allow time for potential manual interruption
                return None  # Simulate failure
            
            response = mapper_stub.Map(mapreduce_pb2.MapRequest(indices=indices[mapper_id-1], centroids=centroid_messages))
            print(f"Mapper {mapper_id} call SUCCESS")
            logging.info("Iteration %d: Mapper %d response: SUCCESS", iteration, mapper_id)
            return response.assignments

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_mappers) as executor:
            futures_to_chunk = {executor.submit(map_chunk, chunk, i+1): i+1 for i, chunk in enumerate(chunks)}
            while futures_to_chunk:
                for future in concurrent.futures.as_completed(futures_to_chunk):
                    mapper_id = futures_to_chunk.pop(future)
                    try:
                        chunk_result = future.result()
                        if chunk_result is None:  # Failure simulation or other error
                            print(f"Retrying mapper {mapper_id} due to failure")
                            logging.error(f"Iteration {iteration}: Mapper {mapper_id} retrying")
                            futures_to_chunk[executor.submit(map_chunk, chunks[mapper_id-1], mapper_id)] = mapper_id
                        else:
                            mapped.extend(chunk_result)
                    except Exception as e:
                        print(f"Exception during mapper {mapper_id} execution, retrying. Error: {str(e)}")
                        logging.error(f"Iteration {iteration}: Mapper {mapper_id} error {str(e)} - retrying")
                        futures_to_chunk[executor.submit(map_chunk, chunks[mapper_id-1], mapper_id)] = mapper_id

        assignments = np.array_split(mapped, self.num_reducers)

        def reducer_assignments(assignment_group, reducer_id):
            port = self.base_port + 50 + reducer_id
            channel = grpc.insecure_channel(f'localhost:{port}')
            reducer_stub = mapreduce_pb2_grpc.MapReduceServiceStub(channel)

            # Simulated probabilistic failure
            if random.random() < 0.5:
                print(f"Reducer {reducer_id} call FAILED due to simulated error")
                logging.info(f"Iteration {iteration}: Reducer {reducer_id} simulated failure")
                time.sleep(1)  # To allow time for potential manual interruption
                return None  # Simulate failure

            response = reducer_stub.Reduce(mapreduce_pb2.ReduceRequest(reducerId=reducer_id))
            print(f"Reducer {reducer_id} call SUCCESS")
            logging.info("Iteration %d: Reducer %d response: SUCCESS", iteration, reducer_id)
            return response.centroids

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_reducers) as executor:
            futures_to_assignment_group = {executor.submit(reducer_assignments, assignment_group, i+1): i+1 for i, assignment_group in enumerate(assignments)}
            while futures_to_assignment_group:
                for future in concurrent.futures.as_completed(futures_to_assignment_group):
                    reducer_id = futures_to_assignment_group.pop(future)
                    try:
                        assignment_group_result = future.result()
                        if assignment_group_result is None:  # Failure simulation or other error
                            print(f"Retrying reducer {reducer_id} due to failure")
                            logging.error(f"Iteration {iteration}: Reducer {reducer_id} retrying")
                            futures_to_assignment_group[executor.submit(reducer_assignments, assignments[reducer_id-1], reducer_id)] = reducer_id
                        else:
                            reduced.extend(assignment_group_result)
                    except Exception as e:
                        print(f"Exception during reducer {reducer_id} execution, retrying. Error: {str(e)}")
                        logging.error(f"Iteration {iteration}: Reducer {reducer_id} error {str(e)} - retrying")
                        futures_to_assignment_group[executor.submit(reducer_assignments, assignments[reducer_id-1], reducer_id)] = reducer_id

        return reduced
    
    def convert_new_centroids(self,protobuf_centroids):
        new_centroids = []
        for centroid in protobuf_centroids:
            x = centroid.location.x
            y = centroid.location.y
            new_centroids.append([x, y])
        return np.array(new_centroids)

    def run(self):
        for i in range(self.iterations):
            new_centroids = self.map_reduce_cycle(i+1)
            new_centroids = self.convert_new_centroids(new_centroids)
            print(f"Iteration {i+1}: New centroids: {new_centroids}")
            logging.info("Iteration %d: New centroids: %s", i+1, new_centroids)
            if np.allclose(self.centroids, new_centroids, atol=1e-4):
                print("Convergence reached.")
                logging.info("Convergence reached after %d iterations.", i+1)
                break
            self.centroids = new_centroids
        print("Final centroids:", new_centroids)
        logging.info("Final centroids: %s", new_centroids)

if __name__ == "__main__":
    num_mappers = int(sys.argv[1])
    num_reducers = int(sys.argv[2])
    data_source = "/Users/manavsaini/Documents/DSCD/Assignment 3/A3/points.txt"  
    num_centroids = int(sys.argv[3])
    iterations = int(sys.argv[4])
    scenario = 1 
    master = MasterServicer(num_mappers, num_reducers, data_source, num_centroids, iterations, scenario)
    master.run()