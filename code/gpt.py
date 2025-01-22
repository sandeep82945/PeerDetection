from openai import OpenAI
import time

from openai import OpenAI
# client = OpenAI(api_key ='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')





def generate_openai(prompt, level=1):
    instructions = {
        1: "Paraphrase the following text with minimal changes. Keep the meaning as close to the original as possible (Only return the output and strictly nothing else):",
        2: "Paraphrase the following text moderately. Change some sentence structures and wording but keep the overall meaning (Only return the output and strictly nothing else):",
        3: "Paraphrase the following text significantly. Rephrase it completely while maintaining the meaning (Only return the output and strictly nothing else):",
    }
    instruction = instructions.get(level, instructions[1])
    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": f'{instruction}'},
        {"role": "user", "content": f"{prompt}"}
    ]
    )
    return completion.choices[0].message.content

if __name__ == "__main__":
    print(generate_openai("I am a god boy who goes to the class"))
