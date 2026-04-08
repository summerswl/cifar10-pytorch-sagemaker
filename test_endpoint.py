"""
test_endpoint.py - Test the deployed CIFAR-10 model on AWS SageMaker
"""

import boto3
import json
from PIL import Image
import io

# ==================== Configuration ====================
ENDPOINT_NAME = "cifar10-endpoint"   # Make sure this matches your deployed endpoint name

class_names = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')

# ==================== Test Function ====================
def test_endpoint(image_path="test_image.jpg"):
    runtime_client = boto3.client('sagemaker-runtime')
    
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        print(f"Sending '{image_path}' to endpoint '{ENDPOINT_NAME}'...")
        
        response = runtime_client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="image/png",
            Body=image_bytes
        )
        
        result = json.loads(response["Body"].read().decode("utf-8"))
        
        print("\n" + "="*60)
        print("✅ PREDICTION RESULT")
        print("="*60)
        print(f"Predicted Class : {result['prediction']}")
        print(f"Confidence      : {result['confidence']*100:.1f}%")
        print("="*60)
        
        return result
        
    except FileNotFoundError:
        print(f"❌ Error: Image file '{image_path}' not found.")
        print("Please place a test image in the project folder as 'test_image.png'")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

# ==================== Run Test ====================
if __name__ == "__main__":
    test_endpoint()
