import openai
import chainlit as cl

openai.api_key_path = "openaikey.txt"
model_name = "gpt-3.5-turbo"
settings = {
    "temperature": 0.7,
    "max_tokens": 500,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}

system_prompt = """You are an helpfull assistant.
"""

@cl.on_chat_start
def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": system_prompt}],
    )


@cl.on_message
def main(message: str):
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message})  
    msg = cl.Message(content="")

    response = openai.ChatCompletion.create(
        model=model_name, messages=message_history, stream=True, **settings
    )
    for resp in response:
        token = resp.choices[0]["delta"].get("content", "")
        msg.stream_token(token)

    message_history.append({"role": "assistant", "content": msg.content})
    msg.send()

"""@cl.on_chat_start
def main():
    res = cl.AskUserMessage(content="What is your OPENAI api key?", timeout=30).send()
    if res:
        openai.api_key = res['content']
        cl.Message(
            content=f"Your key correctly added aske me everything: {res['content']}",
        ).send()"""