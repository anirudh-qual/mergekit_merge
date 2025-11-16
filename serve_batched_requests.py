from openai import AsyncOpenAI
import time
import asyncio
import pandas as pd
import random

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
    rows = []
    for batch in [2,4,8,16,32,64,128]:
        print(f"Running batch size: {batch}")
        requests = create_batch_requests(batch,client)
        tasks = [asyncio.create_task(router(request, special_tokens)) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        avg_ttft = sum([res.ttft for res in results]) / len(results)
        avg_e2e = sum([res.e2e for res in results]) / len(results)
        avg_tbt = sum([res.tbt for res in results]) / len(results)
        rows.append({
            "batch_size": batch,
            "avg_ttft": avg_ttft,
            "avg_e2e": avg_e2e,
            "avg_tbt": avg_tbt
        })

    df = pd.DataFrame(rows)
    df.to_csv("metrics.csv", index=False)

def create_batch_requests(batch_size,client):
        requests = []
        
        # Synthetic prompts for instruct model
        instruct_prompts = [
            "Tell me a joke about dogs.",
            "Write a short poem about the ocean.",
            "Explain quantum computing in simple terms.",
            "What are the benefits of exercise?",
            "How do I make a perfect cup of coffee?",
            "Describe the water cycle.",
            "What is the difference between HTML and CSS?",
            "Give me 3 tips for better time management.",
            "What causes the seasons to change?",
            "Explain photosynthesis to a 10-year-old.",
            "How does a car engine work?",
            "What are some healthy breakfast ideas?",
            "Describe the process of making bread.",
            "What is machine learning?",
            "Give me a fun fact about space.",
        ]
        
        # Synthetic prompts for base model
        base_prompts = [
            "What is capital of Germany?",
            "The tallest mountain in the world is",
            "Python was created by",
            "The speed of light is approximately",
            "Water boils at",
            "The largest planet in our solar system is",
            "The capital of France is",
            "DNA stands for",
            "The human body has",
            "The Great Wall of China was built to",
            "E=mc² was proposed by",
            "The primary colors are",
            "The Earth orbits around the",
            "The smallest unit of life is",
            "Oxygen has the chemical symbol",
        ]
        
        # Generate half instruct and half base requests
        for i in range(batch_size // 2):
            prompt = random.choice(instruct_prompts)
            requests.append(Request(model="instruct", prompt=prompt, client=client))
        
        for i in range(batch_size // 2):
            prompt = random.choice(base_prompts)
            requests.append(Request(model="base", prompt=prompt, client=client))
        
        # Shuffle to mix instruct and base requests
        random.shuffle(requests)
        
        return requests

if __name__ == "__main__":
    asyncio.run(main())
