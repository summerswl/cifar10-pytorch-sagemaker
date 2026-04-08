import logging
logging.getLogger("sagemaker.config").setLevel(logging.WARNING)
logging.getLogger("sagemaker").setLevel(logging.WARNING)

import sagemaker
from sagemaker.pytorch.model import PyTorchModel

role_arn = "arn:aws:iam::991776827101:role/SageMakerExecutionRole-MNIST"
bucket_name = "summerswl-pytorch-project"

model = PyTorchModel(
    model_data=f"s3://{bucket_name}/cifar10/model.tar.gz",
    role=role_arn,
    framework_version="2.1",
    py_version="py310",
    entry_point="inference.py",
    source_dir="model_package",
)

print("🚀 Deploying CIFAR-10 CNN model to SageMaker...")

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name="cifar10-endpoint",
    wait=True
)

print("\n✅ DEPLOYMENT SUCCESSFUL!")
print(f"Endpoint Name: {predictor.endpoint_name}")
