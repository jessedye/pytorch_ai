import requests
import argparse

def send_prompt(api_url, prompt):
    try:
        # Prepare the JSON payload with the prompt
        payload = {
            "prompt": prompt
        }

        # Send POST request to the API
        response = requests.post(api_url, json=payload)

        # Raise an exception if the request was unsuccessful
        response.raise_for_status()

        # Parse the response JSON
        result = response.json()

        # Check for OpenAI-style text response structure
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0].get("text", "No text found")
        # Check for image response
        elif "image_path" in result:
            image_url = result["image_path"]
            return f"Image URL: {image_url}"
        else:
            return "No valid response generated."

    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Send prompts to an AI API.")
    parser.add_argument('--api-url', type=str, default="DOMAIN.com", 
                        help="The API URL to send the prompt to")
    parser.add_argument('--prompt', type=str, default=None, help="The prompt to send to the API")

    args = parser.parse_args()

    if args.prompt:
        # If prompt is passed as an argument, send it to the AI and display the response
        response = send_prompt(args.api_url, args.prompt)
        print(f"AI Response: {response}")
    else:
        # Otherwise, enter interactive mode
        while True:
            prompt = input("Enter your prompt (or 'exit' to quit): ")

            if prompt.lower() == 'exit':
                print("Exiting...")
                break

            # Send prompt to the AI and get the response
            response = send_prompt(args.api_url, prompt)

            # Display the response from the AI
            print(f"AI Response: {response}\n")
