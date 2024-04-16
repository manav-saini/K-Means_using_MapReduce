import grpc
import sys
from concurrent import futures
import numpy as np
import random
import mapreduce_pb2
import mapreduce_pb2_grpc

class MasterServicer(mapreduce_pb2_grpc.MapReduceServiceServicer):
    def __init__(self, num_mappers, num_reducers, data_source, num_centroids, iterations, scenario,base_port=5000):
        self.num_mappers = num_mappers
        self.num_reducers = num_reducers
        self.data_source = data_source  
        self.num_centroids = num_centroids
        self.iterations = iterations
        self.scenario = scenario
        self.centroids = self.initialize_centroids()
        self.base_port = base_port

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

    def map_reduce_cycle(self):
        data_points = self.load_data()
        if self.scenario == 1:
            indices = np.array_split(range(len(data_points)), self.num_mappers)
            chunks = [data_points[idx] for idx in indices]
        else:
            chunks = np.array_split(data_points, self.num_mappers)
        mapped = []
        reduced = []
        channel = grpc.insecure_channel('localhost:50051')
        mapper_stub = mapreduce_pb2_grpc.MapReduceServiceStub(channel)

        for chunk in chunks:
            point_messages = [mapreduce_pb2.Point(x=point[0], y=point[1]) for point in chunk]
            centroid_messages = [mapreduce_pb2.Centroid(id=i, location=mapreduce_pb2.Point(x=centroid[0], y=centroid[1])) for i, centroid in enumerate(self.centroids)]
            try:
                response = mapper_stub.Map(mapreduce_pb2.MapRequest(data=point_messages, centroids=centroid_messages))
                mapped.extend(response.assignments)
            except grpc.RpcError as e:
                print(f"Error: {str(e)} - Retrying...")
                continue

        assignments = np.array_split(mapped, self.num_reducers)
        channel = grpc.insecure_channel('localhost:50052')
        reducer_stub = mapreduce_pb2_grpc.MapReduceServiceStub(channel)
        for i, assignment_group in enumerate(assignments):
            try:
                response = reducer_stub.Reduce(mapreduce_pb2.ReduceRequest(reducerId=i))
                reduced.extend(response.centroids)
            except grpc.RpcError as e:
                print(f"Error: {str(e)} - Retrying...")
                continue

        return reduced
    def convert_new_centroids(self,protobuf_centroids):
        new_centroids = []
        for centroid in protobuf_centroids:
            x = centroid.location.x
            y = centroid.location.y
            new_centroids.append([x, y])
        return np.array(new_centroids)

    def run(self):
        old_centroids = np.array(self.centroids)
        for i in range(self.iterations):
            new_centroids = self.map_reduce_cycle()
            new_centroids = self.convert_new_centroids(new_centroids)
            if np.allclose(old_centroids, new_centroids, atol=1e-4):
                print("Convergence reached.")
                break
            old_centroids = new_centroids
        print("Final centroids:", new_centroids)

if __name__ == "__main__":
    num_mappers = int(sys.argv[1])
    num_reducers = int(sys.argv[2])
    data_source = "/Users/manavsaini/Documents/DSCD/Assignment 3/A3/points3.txt"  # Could be a single file path or list of file paths
    num_centroids = int(sys.argv[3])
    iterations = int(sys.argv[4])
    scenario = 1  # 1 for single big file, 2 for multiple files
    master = MasterServicer(num_mappers, num_reducers, data_source, num_centroids, iterations, scenario)
    master.run()