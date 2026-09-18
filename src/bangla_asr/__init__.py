"""
Bangla ASR: Inference-Only Rescoring and Candidate Optimization Framework.
Provides modular components for:
  - Audio chunking (Silero VAD) & sequential ground-truth fuzzy alignment
  - Autoregressive diverse beam candidate generation (Whisper-medium)
  - Minimum Bayes Risk (MBR) consensus selection
  - Discriminative candidate reranking (BanglaBERT regression / quality estimation)
"""

__version__ = "1.0.0"
