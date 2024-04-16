import grpc
from concurrent import futures
import numpy as np
import os
import mapreduce_pb2
import mapreduce_pb2_grpc

class ReducerServicer(mapreduce_pb2_grpc.MapReduceServiceServicer):
    def __init__(self):
        self.output_dir = "reducer_output"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def Reduce(self, request, context):
        # Request data from Mapper
        channel = grpc.insecure_channel('localhost:50051')
        mapper_stub = mapreduce_pb2_grpc.MapReduceServiceStub(channel)
        mapper_response = mapper_stub.FetchData(mapreduce_pb2.FetchDataRequest(reducerId=request.reducerId))

        centroid_points = {}
        for assignment in mapper_response.assignments:
            if assignment.centroidId not in centroid_points:
                centroid_points[assignment.centroidId] = []
            centroid_points[assignment.centroidId].append((assignment.point.x, assignment.point.y))

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
        with open(os.path.join(self.output_dir, f'reducer_{centroid_id}_output.txt'), 'w') as file:
            file.write(f"Centroid ID: {centroid_id}, New Centroid: [{mean_point[0]}, {mean_point[1]}]\n")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    mapreduce_pb2_grpc.add_MapReduceServiceServicer_to_server(ReducerServicer(), server)
    server.add_insecure_port('[::]:50052')
    server.start()
    print("Reducer server started on port 50052.")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()