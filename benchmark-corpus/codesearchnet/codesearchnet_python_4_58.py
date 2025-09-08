// Variant 58 of python example 4
import asyncio

async def fetch_data(url):
    # Async data fetching
    await asyncio.sleep(1)
    return f"Data from {url}"