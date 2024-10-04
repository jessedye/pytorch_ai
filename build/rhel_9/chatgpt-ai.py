import openai
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Set your OpenAI API key (replace with your actual API key)
openai.api_key = "your-openai-api-key"

# Define input model for FastAPI
class PromptRequest(BaseModel):
    prompt: str

# Define the generate endpoint
@app.post("/generate/")
async def generate_text(request: PromptRequest):
    try:
        # Call the OpenAI API with the provided prompt using GPT-4
        response = openai.ChatCompletion.create(
            model="gpt-4",  # You can specify "gpt-3.5-turbo" for a cheaper version
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": request.prompt}
            ],
            max_tokens=500,  # Adjust the max tokens for response length
            temperature=0.7,  # Adjust temperature for randomness
        )

        # Extract and return the response from the GPT-4 model
        response_text = response['choices'][0]['message']['content']
        return {"response": response_text}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
