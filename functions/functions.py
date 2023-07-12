import requests

# Function to generate AI based images using OpenAI Dall-E
def generate_images(text):
    response = openai.Image.create(prompt= text, n=1, size="512x512")
    image_url = response['data'][0]['url']
    return image_url


api_functions = {
    "generate_images": generate_images
}