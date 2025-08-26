# AIStaticReasoningExtractor
This project aims to extract reasoning capabilities of larger models via static analysis for enhancing smaller models capabilities.

In my view, an 8B model should be able to 1:1 match the performance of extremely large parameter models.

The intent of this project is as follows: Perform static analysis on larger models (over 400B), statically analyze them in batches on consumer level GPUs with 12 GB of VRAM by chunking the analysis, and then caching their reasoning capability (domain agnostic), allowing smaller models to achieve bolt-on 1:1 performance, or even improved performance by combining both models capabilities, where the 8B can potentiall fill in the larger models gaps.
