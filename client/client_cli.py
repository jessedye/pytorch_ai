import requests
import argparse

def send_prompt(api_url, prompt):
    try:
        # Correct JSON format with the prompt field
        payload = {
            "prompt": prompt
        }

        # Send POST request to the API
        response = requests.post(api_url, json=payload)

        # Raise an exception if the request was unsuccessful
        response.raise_for_status()

        # Parse the response JSON
        result = response.json()

        # Return the generated text
        return result.get("response", "No response generated.")
    
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Send prompts to an AI API.")
    parser.add_argument('--api-url', type=str, default="https://elder-brain-api.dev.aquia-k8s-lab.net/direct/", 
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
