import grpc
from concurrent import futures
import numpy as np
import random
import mapreduce_pb2
import mapreduce_pb2_grpc
import os
import json
import sys

class MapperServicer(mapreduce_pb2_grpc.MapReduceServiceServicer):
    def __init__(self, mapper_id, reducer_count=4, base_directory='Data/Mappers'):
        self.mapper_id = mapper_id
        self.reducer_count = reducer_count
        self.base_directory = base_directory
        self.output_dir = os.path.join(base_directory, f'M{mapper_id}')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.partition_files = [open(os.path.join(self.output_dir, f'partition_{i+1}.txt'), 'w') for i in range(reducer_count)]

    def __del__(self):
        for file in self.partition_files:
            file.close()

    def Map(self, request, context):
        #Check for simulated failure
        # if random.random() < 0.5:
        #     # Simulate a failure with a clear error message or status
        #     context.set_code(grpc.StatusCode.ABORTED)
        #     context.set_details('Simulated failure condition triggered.')
        #     return mapreduce_pb2.MapResponse()  # Returning an empty response to indicate failure
        self.partition_files = [open(os.path.join(self.output_dir, f'partition_{i+1}.txt'), 'w') for i in range(self.reducer_count)]
        try:
            # Printing first data point and centroid location to check correct data types
            print("First data point:", request.data[0])
            print("First centroid location:", request.centroids[0].location)

            # Converting Protobuf messages to Python tuples for easier manipulation
            data_points = [(point.x, point.y) for point in request.data]
            centroids = [centroid for centroid in request.centroids]
            print(centroids)

            assignments = []
            for point in data_points:
                closest_centroid_id, closest_distance = self.find_closest_centroid(point, centroids)
                self.write_to_partition(closest_centroid_id, point) 
                assignments.append(mapreduce_pb2.CentroidAssignment(
                    centroidId=closest_centroid_id,
                    point=mapreduce_pb2.Point(x=point[0], y=point[1])
                ))
            return mapreduce_pb2.MapResponse(assignments=assignments)
        except Exception as e:
            # Log the exception and abort the RPC with an error.
            print(f"Error processing Map request: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details('Error processing the request.')
            return mapreduce_pb2.MapResponse()
    
    def find_closest_centroid(self, point, centroids):
        closest_centroid_id = None
        closest_distance = float('inf')
        for centroid in centroids:
            distance = np.sqrt((centroid.location.x - point[0]) ** 2 + (centroid.location.y - point[1]) ** 2)
            if distance < closest_distance:
                closest_distance = distance
                closest_centroid_id = centroid.id
        return closest_centroid_id, closest_distance

    def write_to_partition(self, centroid_id, point):
        reducer_index = centroid_id % self.reducer_count
        with open(os.path.join(self.output_dir, f'partition_{reducer_index+1}.txt'), 'a') as file:
            file.write(f"{centroid_id},{point[0]},{point[1]}\n")

    def read_data(self, data_indices_or_files):
        if isinstance(data_indices_or_files, list) and all(isinstance(x, int) for x in data_indices_or_files):
            return self.read_data_from_indices(data_indices_or_files)
        elif isinstance(data_indices_or_files, list) and all(isinstance(x, str) for x in data_indices_or_files):
            return self.read_data_from_files(data_indices_or_files)
        else:
            raise ValueError("Invalid input format")

    def read_data_from_indices(self, indices):
        file_path = os.path.join(self.base_directory, 'large_dataset.txt')
        data = np.loadtxt(file_path)
        selected_data = data[indices]
        return selected_data

    def read_data_from_files(self, file_names):
        data = []
        for file_name in file_names:
            file_path = os.path.join(self.base_directory, file_name)
            data.extend(np.loadtxt(file_path,delimiter=','))
        return data
    
    def FetchData(self, request, context):
        reducer_id = request.reducerId
        data = self.read_data_from_files([f'M{self.mapper_id}/partition_{reducer_id}.txt'])
        print(data)
        assignments = []
        print(data)
        for parts in data:
            assignments.append(mapreduce_pb2.CentroidAssignment(
                centroidId=int(parts[0]),
                point=mapreduce_pb2.Point(x=float(parts[1]), y=float(parts[2]))
            ))
        return mapreduce_pb2.FetchDataResponse(assignments=assignments)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    mapper_id = int(sys.argv[1])
    redcer_count = int(sys.argv[2])
    mapreduce_pb2_grpc.add_MapReduceServiceServicer_to_server(MapperServicer(mapper_id=mapper_id, reducer_count=redcer_count), server)
    server.add_insecure_port(f'[::]:{5000+mapper_id}')
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
