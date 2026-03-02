import asyncio
import aiohttp

URL = "http://localhost:8000/v1/chat/completions"

async def message(session, text):
    payload = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 2000,
        "stream": True
    }

    async with session.post(URL, json=payload) as resp:
        async for chunk in resp.content:
            pass  # just consume stream

async def main():
    long_text = "hello " * 8000

    async with aiohttp.ClientSession() as session:
        tasks = [
            message(session, long_text)
            for _ in range(20)  # 5 concurrent sessions
        ]

        await asyncio.gather(*tasks)

asyncio.run(main())
