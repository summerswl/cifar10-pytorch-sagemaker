"""
Gradio Web Interface for CIFAR-10 CNN Classifier with Examples
"""

import gradio as gr
import boto3
import json
import io

# Configuration
ENDPOINT_NAME = "cifar10-endpoint"

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Initialize SageMaker Runtime client
runtime_client = boto3.client('sagemaker-runtime')

def predict_image(image):
    """Send image to SageMaker endpoint and return prediction."""
    try:
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        image_bytes = img_byte_arr.getvalue()

        response = runtime_client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="image/png",
            Body=image_bytes
        )

        result = json.loads(response["Body"].read().decode("utf-8"))
        
        prediction = result.get("prediction", "Unknown")
        confidence = result.get("confidence", 0) * 100
        
        return f"**Predicted:** {prediction}\n**Confidence:** {confidence:.1f}%"
    
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Create Gradio Interface with Examples
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Markdown(label="Prediction Result"),
    title="CIFAR-10 Image Classifier",
    description="Upload a photo or click an example below. The model will predict one of 10 classes.",
    examples=[
        ["examples/dog.jpg"],
        ["examples/cat.jpg"],
        ["examples/airplane.jpg"],
        ["examples/car.jpg"],
        ["examples/bird.jpg"],
    ],
    flagging_mode="never",
    cache_examples=False
)

if __name__ == "__main__":
    print("Starting Gradio web interface for CIFAR-10...")
    print("Open the link shown below in your browser")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False   # Change to True if you want a temporary public link
    )
