import requests
import json
import shutil
from datetime import datetime

COLUMNS = shutil.get_terminal_size().columns
OLLAMA_HOST = "http://localhost:11434"
TEST_SYSTEM_PROMPT = "" 
TESTER_PROMPT = ""
ADA_SYSTEM_PROMPT= ""
REWERITTER_SYSTEM_PROMPT=""
def query_ollama_chat(messages, model, stream=False, host=OLLAMA_HOST):
    url = f"{host}/api/chat"
    headers = {"Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "stream": stream}

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        result = response.json()
        return result.get("message", {}).get("content", "")
    else:
        raise Exception(f"Request failed with {response.status_code}: {response.text}")


def run_two_model_conversation(previous_question, previous_answer):
    model_a="FreeQwen"
    model_b="FreeQwen"
    model_c="FreeQwen"
    # Step 1: Model A generates a question
    system_a = TEST_SYSTEM_PROMPT
    user_a = TESTER_PROMPT + "Your previous question:" + previous_question + "Ada's answer to the previous question" + previous_answer

    question = query_ollama_chat(
        messages=[
            {"role": "system", "content": system_a},
            {"role": "user", "content": user_a},
        ],
        model=model_a,
    )
    #print("\n" + "="*COLUMNS)
    #print(f"Tester thinking: {question.partition('</think>\n\n')[0]}\n\n")

    question=question.partition('</think>\n\n')[2]
    
    # Step 2: Model B answers the question
    system_b = ADA_SYSTEM_PROMPT
    answer = query_ollama_chat(
        messages=[
            {"role": "system", "content": system_b},
            {"role": "user", "content": question},
        ],
        model=model_b,
    )
    trimmed_answer = answer.partition('</think>\n\n')[2]

    system_c = REWERITTER_SYSTEM_PROMPT
    user_c = answer.partition('</think>\n\n')[0]
    user_c = "** Internal thinking section to rewrite:**\n" + user_c.partition('<think>\n')[2] + "End of Internal thinking section to rewrite.**\n"
    revised_thinking = query_ollama_chat(
        messages=[
            {"role": "system", "content": system_c},
            {"role": "user", "content": user_c},
        ],
        model=model_c,
    )
    
    revised_thinking = revised_thinking.replace("the user", "Drew")

    revised_thinking = revised_thinking.partition('</think>\n\n')[2]
    print("\n" + "="*COLUMNS)
    print(f"🤔\033[96mTester's Question:\n{question}\n\n\033[0m")
    print(f"🧠\033[93mRevised thinking:\n {revised_thinking}\n\n\033[0m")
    print(f"🙋🏻‍♀️\033[92mAda's Answer:\n {trimmed_answer}\n\n\033[0m")
    print("\n" + "="*COLUMNS)
    # Step 3: Record conversation in training JSON format
    final_answer = revised_thinking + "</think>\n\n" + trimmed_answer
    conversation = {
        "conversations": [
            {"role": "system", "content": system_b},
            {"role": "user", "content": question},
            {"role": "assistant", "content": final_answer},
        ],
    }

    return conversation

if __name__ == "__main__":
    previous_answer = "None."
    previous_question = "None."
    # Loop for 10 conversations
    num_lines = sum(1 for _ in open("generatedTrainingData.json"))
    while(num_lines < 200000):
    # Read in the various prompts
        with open("tester_system_prompt.txt") as tester_system_prompt:
            TEST_SYSTEM_PROMPT = tester_system_prompt.read()

        with open("tester_prompt.txt") as test_prompt:
            TESTER_PROMPT = test_prompt.read()

        with open("ada_system_prompt.txt") as ada_system_prompt:
            ADA_SYSTEM_PROMPT = ada_system_prompt.read()
        with open("reweritter_prompt.ptxt") as rewritter_prompt:
            REWERITTER_SYSTEM_PROMPT = rewritter_prompt.read()
    # Get converstion 
        conversation = run_two_model_conversation(previous_question, previous_answer)
        previous_question = conversation['conversations'][1]['value']
        previous_answer = conversation['conversations'][2]['value']
        # Save to json for future training data use
        with open("generatedTrainingData.json", "a") as f:
            json.dump(conversation['conversations'], f)
            f.write(",\n")
        num_lines = sum(1 for _ in open("generatedTrainingData.json"))