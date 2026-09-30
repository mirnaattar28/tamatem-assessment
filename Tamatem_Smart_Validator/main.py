import asyncio
import json
import sys
from pathlib import Path

from tm_validateAndTag import tm_validateAndTag


async def main():
    default = Path(__file__).parent / "frames" / "frame (1).png"
    frame = sys.argv[1] if len(sys.argv) > 1 else str(default)
    report = await tm_validateAndTag(frame)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())