from openai import AsyncOpenAI
import time
import asyncio
import pandas as pd

class Request:
    def __init__(self,model,prompt,client):
        self.model = model
        self.prompt = prompt
        self.t_start = None
        self.t_first_token = None
        self.last_token_time = None
        self.t_end = None
        self.num_tokens = 0
        self.num_gaps = 0
        self.total_gap_time = 0.0
        self.ttft = None
        self.e2e = None
        self.tbt = None
        self.client = client

class Response:
    def __init__(self,request):
        self.response = ""
        self.ttft = request.ttft
        self.e2e = request.e2e
        self.tbt = request.tbt

async def run(req:Request):
    prompt = req.prompt
    client = req.client
    res = Response(req)
    req.t_start = time.perf_counter()
    stream = await client.chat.completions.create(
        model="with_special_tokens",
        messages=[
            {"role": "system", "content": "You are a helpful assistant answer each question briefly."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024,
        stream=True
    )
    
    async for chunk in stream:
        
        choice = chunk.choices[0]
        delta = choice.delta

        if delta.content:
            req.num_tokens += 1

            if req.t_first_token is None:
                req.t_first_token = time.perf_counter()
                req.last_token_time = req.t_first_token
            else:
                gap_time = time.perf_counter() - req.last_token_time
                req.total_gap_time += gap_time
                req.num_gaps += 1
                req.last_token_time = time.perf_counter()
            res.response += delta.content
    req.t_end = time.perf_counter()
    res.ttft = req.t_first_token - req.t_start
    res.e2e = req.t_end - req.t_start
    res.tbt = req.total_gap_time / req.num_gaps if req.num_gaps > 0 else 0

    return res
    




async def router(request,special_tokens):

    # Logic of which model it belongs to
    model = request.model
    if model == "base":
        special_token = special_tokens["base"]
    else :
        special_token = special_tokens["instruct"]
    
    # Prepend the special token to the input
    request.prompt = special_token + " " + request.prompt
    return await run(request)


async def main():
    client = AsyncOpenAI(base_url="http://localhost:8080/v1", api_key="asdad")
    special_tokens = {"base": "<|use_base|>", "instruct": "<|use_instruct|>"}

    requests = [
        Request(model="instruct", prompt="Tell me a joke about cats.", client=client),
        Request(model="base", prompt="What is capital of India?", client=client),
        Request(model="instruct", prompt="What is the capital of France?", client=client),
        Request(model="base", prompt="What is opposite of die?", client=client),
        Request(model="instruct", prompt="How many wives does Lord Shree Krishna have?", client=client),
        Request(model="base", prompt="What is other name for mitosis.", client=client),
        Request(model="instruct", prompt="What are the benefits of exercise?", client=client),
        Request(model="base", prompt="How many number of stages in water cycle.", client=client),
        Request(model="instruct", prompt="Tell me a fun fact about space.", client=client),
        Request(model="base", prompt="How many years did Covid last?", client=client),]

    tasks = [asyncio.create_task(router(request, special_tokens)) for request in requests]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    
    df = pd.DataFrame([{
        "ttft": res.ttft,
        "e2e": res.e2e,
        "tbt": res.tbt
    } for res in results])

    df.to_csv("metrics.csv", index=False)

if __name__ == "__main__":
    asyncio.run(main())