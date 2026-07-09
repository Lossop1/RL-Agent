"""locomotion console - live operations frontend over the RL-locomotion management surfaces.

Backend is a thin FastAPI skin over the existing autotuner/tools modules. It runs locally
and reaches the remote GPU box via the same paramiko SSH the rest of the autotuner uses.
"""
