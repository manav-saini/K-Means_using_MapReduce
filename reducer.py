import grpc
from concurrent import futures
import numpy as np
import os
import mapreduce_pb2
import mapreduce_pb2_grpc
import sys
import concurrent.futures

class ReducerServicer(mapreduce_pb2_grpc.MapReduceServiceServicer):
    def __init__(self,reducer_id,base_port=5000, num_mappers=4):
        self.output_dir = "Data/Reducers"
        self.base_port = base_port
        self.num_mappers = num_mappers
        self.reducer_id = reducer_id
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def fetch_data_from_mapper(self, mapper_id):
        channel = grpc.insecure_channel(f'localhost:{self.base_port + mapper_id}')
        mapper_stub = mapreduce_pb2_grpc.MapReduceServiceStub(channel)
        try:
            response = mapper_stub.FetchData(mapreduce_pb2.FetchDataRequest(reducerId=self.reducer_id))  # Assuming all reducers fetch from all mappers
            return response.assignments
        except grpc.RpcError as e:
            print(f"Failed to fetch data from mapper {mapper_id} on port {self.base_port + mapper_id}: {str(e)}")
            return []

    def Reduce(self, request, context):
        # Parallel fetch data from all mappers
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_mappers) as executor:
            future_to_mapper = {executor.submit(self.fetch_data_from_mapper, i+1): i for i in range(self.num_mappers)}
            centroid_points = {}
            for future in concurrent.futures.as_completed(future_to_mapper):
                assignments = future.result()
                for assignment in assignments:
                    if assignment.centroidId not in centroid_points:
                        centroid_points[assignment.centroidId] = []
                    centroid_points[assignment.centroidId].append((assignment.point.x, assignment.point.y))

        # Calculate new centroids
        new_centroids = []
        for centroid_id, points in centroid_points.items():
            points_array = np.array(points)
            mean_point = np.mean(points_array, axis=0)
            new_centroid = mapreduce_pb2.Centroid(
                id=centroid_id,
                location=mapreduce_pb2.Point(x=mean_point[0], y=mean_point[1])
            )
            new_centroids.append(new_centroid)
            self.write_to_file(centroid_id, mean_point)

        return mapreduce_pb2.ReduceResponse(centroids=new_centroids)

    def write_to_file(self, centroid_id, mean_point):
        with open(os.path.join(self.output_dir, f'reducer_{centroid_id+1}_output.txt'), 'w') as file:
            file.write(f"Centroid ID: {centroid_id}, New Centroid: [{mean_point[0]}, {mean_point[1]}]\n")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    reducer_id = int(sys.argv[1])
    mapper_count = int(sys.argv[2])
    mapreduce_pb2_grpc.add_MapReduceServiceServicer_to_server(ReducerServicer(reducer_id=reducer_id,num_mappers=mapper_count), server)
    server.add_insecure_port(f'[::]:{5050+reducer_id}')
    server.start()
    print(f"Reducer server started on port {5050+reducer_id}.")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
