from nanovllm_voxcpm.models.voxcpm.server import AsyncVoxCPMServer
import numpy as np
import soundfile as sf
from tqdm.asyncio import tqdm

async def main():
    print("Loading...")
    server = AsyncVoxCPMServer(model="~/VoxCPM-0.5B")
    await server.wait_for_ready()
    print("Ready")

    buf = []
    async for data in tqdm(server.generate(target_text="I have a dream that my four little children will one day live in a nation where they will not be judged by the color of their skin but by the content of their character. I have a dream today! I have a dream that one day, down in Alabama, with its vicious racists, with its governor having his lips dripping with the words of interposition and nullification; one day right down in Alabama little black boys and black girls will be able to join hands with little white boys and white girls as sisters and brothers. I have a dream today! I have a dream that one day every valley shall be exalted, and every hill and mountain shall be made low, the rough places will be made plain, and the crooked places will be made straight, and the glory of the Lord shall be revealed and all flesh shall see it together.")):
        buf.append(data)
    wav = np.concatenate(buf, axis=0)
    sf.write("test.wav", wav, 16000)

    await server.stop()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
