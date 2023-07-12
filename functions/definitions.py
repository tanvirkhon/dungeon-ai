function_definitions[
    {
        "name": "generate_images",
        "description": "Generate AI based images using OpenAI Dall-E if the user asks for an image.",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to generate the image from."
            }
        },
        "returns": {
            "image_url": {
                "type": "string",
                "description": "The URL of the generated image."
            }
        },
        "required": ["text"]
    }
]