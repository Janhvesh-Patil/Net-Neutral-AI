# Net-Neutral AI: Current Blocks & Limitations

Before exploring the future roadmap, it's important to understand the technical limitations of our current prototype and why certain architectural decisions were made.

## 1. Cloud Memory Limits (OOM Crashes)
The current cloud host for our Coordinator backend is Render's free tier, which imposes a strict 512 MB memory limit on the Python server container.
- **The Block:** When evaluating a massive global model on a 50 MB dataset (like the full IMDB set), Python `csv` parsing and PyTorch tensor allocations quickly exceed 512 MB. When this happens, the Linux operating system forcefully terminates the server process (OOM Kill), causing clients to receive `502 Bad Gateway` errors.
- **Current Workaround:** To prevent crashes, we eliminated the hardcoded 50 MB datasets and implemented a dynamic 80/20 data split on much smaller uploaded `.csv` files. The Coordinator evaluates exclusively on the tiny 20% test split, keeping memory usage well under the 512 MB limit.

## 2. Strict HTTP Timeouts
Render enforces a strict 60-second limit for any single HTTP request to complete. If a response takes longer than 60 seconds, Render cuts the connection and returns a `504 Gateway Timeout` or `Read timed out` error.
- **The Block:** When a client submits its trained weights at the end of a round, the Coordinator previously attempted to run `FedAvg` and evaluate the entire global model before responding to the client. On Render's slow CPUs, this process took several minutes, causing the client's HTTP POST request to time out and crash the client script.
- **Current Workaround:** We refactored the Coordinator to offload the heavy evaluation step into an asynchronous background thread. Now, the `/submit` endpoint instantly replies with a `200 OK` the millisecond it receives the weights, keeping Render happy while it crunches the numbers out-of-band.

## 3. Asynchronous UI & Data Disparity
Currently, there is a stark difference between the data visible to the developer in the command terminal versus what is displayed on the web dashboard.
- **The Block:** The web frontend relies on basic HTTP polling (sending a `GET` request every few seconds) to check for updates. This is inefficient, prone to missed state changes, and doesn't support live streaming of complex data arrays. As a result, granular data like per-epoch loss, local accuracy, and processing times remain trapped in the local terminal.
- **Current Workaround:** The Web UI serves as a high-level state tracker (Round 1, Round 2, Leaderboard), while the actual debugging and model telemetry are entirely reliant on reading the raw terminal stdout.
