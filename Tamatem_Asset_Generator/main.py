import asyncio
import sys

from tm_generateAsset import tm_generateAsset


async def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "A Spartan warrior fighting a lion"
    ratio = sys.argv[2] if len(sys.argv) > 2 else "1:1"
    path = await tm_generateAsset(prompt, ratio)
    print(f"Saved: {path}")


if __name__ == "__main__":
    asyncio.run(main())