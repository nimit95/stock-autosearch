# Stock AutoSearch

Autonomous ML experimentation for the Indian stock market (Nifty 100).

See `program.md` for the full experiment protocol.

## Setup

1. Install TA-Lib C library:
   ```bash
   # macOS
   brew install ta-lib

   # Ubuntu/Debian
   sudo apt-get install libta-lib-dev
   ```

2. Create virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Fetch and cache market data:
   ```bash
   python prepare.py
   ```

4. Run baseline:
   ```bash
   python train.py
   ```
