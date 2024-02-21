from pathlib import Path
import sys

HERE = Path(__file__).parent
ROOT_DIR = HERE.parent

sys.path.append(str(ROOT_DIR))

from protos.python.feature_extraction import feature_extraction_pb2_grpc, feature_extraction_pb2
import torchreid

# =============================================================================

import grpc
import pickle
from concurrent import futures
import argparse
import socket
import torch
import numpy as np
from torchreid.utils import FeatureExtractor
import torchvision.transforms as T


class FeatureExtractionServicer(feature_extraction_pb2_grpc.FeatureExtractorServicer):

    def __init__(
            self, 
            **kwargs
    ):
        self.device = kwargs["device"]
        checkpoint_path = kwargs["checkpoint_path"]

        self.model = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path=checkpoint_path,
            device=self.device,
            image_size=[256, 128]
        )

        self.val_transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def predict(self, request, context):
        img_batch = pickle.loads(request.imgs_pkl)

        with torch.no_grad():
            img_batch = (img_batch.permute(0, 2, 3, 1).numpy()[:,:,:,::-1] * 255).astype(np.uint8)

            print(f'[{np.random.randint(1, 10)} :-D] Processing batch of shape', img_batch.shape)
            img_batch = torch.tensor([self.val_transforms(img).tolist() for img in img_batch])
            features = self.model(img_batch)

        features_pkl = pickle.dumps(features)
        return feature_extraction_pb2.FeatureBatch(features_pkl=features_pkl)


def parse_kwargs():
    
    ap = argparse.ArgumentParser()

    ap.add_argument("--port", required=True, type=int, help="Port number")
    ap.add_argument("--device", required=True, type=str)
    ap.add_argument("--checkpoint_path", required=True, type=str)

    kwargs = vars(ap.parse_args())

    return kwargs


if __name__ == '__main__':

    kwargs = parse_kwargs()

    port = kwargs["port"]

    # check if the port is not used
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(('localhost', port)) == 0:
            raise ValueError(f"Port {port} is already in use. Please check if the server is already running.")
        
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    feature_extraction_pb2_grpc.add_FeatureExtractorServicer_to_server(FeatureExtractionServicer(**kwargs), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"Server started at port {port}")
    server.wait_for_termination()