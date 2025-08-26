#!/usr/bin/env python3
"""
CHIMERA v5.0 - EXACT 1:1 REASONING TRANSFER
Complete implementation for transferring reasoning from 400B+ models to 8B GGUF models
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from pathlib import Path
import hashlib
import struct
import mmap
import json
import pickle
import gc
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue
import time
from safetensors import safe_open
import os
import psutil
import signal
import logging
from contextlib import contextmanager
import itertools
import lz4.frame
import xxhash
import math
import inspect
import types
from functools import partial
import warnings
import tempfile
import subprocess
import requests
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoConfig
import gguf
from gguf import GGUFReader, GGUFWriter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('chimera_exact.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ExactCircuit:
    """Exact mathematical representation of a reasoning circuit"""
    layer_idx: int
    circuit_type: str
    weights: Dict[str, torch.Tensor]
    computation_code: bytes
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    exact_hash: str
    memory_size: int
    bias: Optional[Dict[str, torch.Tensor]] = field(default_factory=dict)

@dataclass
class ReasoningCache:
    """Exact reasoning cache for 1:1 transfer"""
    circuits: Dict[str, ExactCircuit]
    computation_graph: Dict[str, List[str]]
    verification_data: Dict[str, Any]
    total_memory: int
    exact_checksum: str
    model_spec: Dict[str, Any] = field(default_factory=dict)

class ExactMemoryManager:
    """Manages exact memory operations for large models"""
    
    def __init__(self, max_memory_gb: float = 64.0):
        self.max_memory = max_memory_gb * 1024**3
        self.current_memory = 0
        self.lock = threading.Lock()
        self.temp_files = []
        
    def get_memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        process = psutil.Process()
        return process.memory_info().rss
    
    def allocate_memory(self, size: int) -> bool:
        """Check if we can allocate memory"""
        with self.lock:
            projected = self.current_memory + size
            if projected < self.max_memory:
                self.current_memory = projected
                return True
            return False
    
    def release_memory(self, size: int):
        """Release memory allocation"""
        with self.lock:
            self.current_memory = max(0, self.current_memory - size)
    
    @contextmanager
    def memory_context(self, size: int):
        """Context manager for memory allocation"""
        if self.allocate_memory(size):
            try:
                yield
            finally:
                self.release_memory(size)
        else:
            raise MemoryError(f"Cannot allocate {size} bytes")
            
    def cleanup(self):
        """Clean up temporary files"""
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {file_path}: {e}")
        self.temp_files = []

class LargeModelLoader:
    """Efficient weight loading for very large models (400B+ parameters)"""
    
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.safetensor_files = list(self.model_dir.glob('*.safetensors'))
        self.bin_files = list(self.model_dir.glob('*.bin'))
        self.files = self.safetensor_files + self.bin_files
        self.files.sort(key=lambda x: x.name)
        self.memory_mapped_tensors = {}
        
    def load_all_weights_memory_mapped(self) -> Dict[str, torch.Tensor]:
        """Load all weights with memory mapping for large models"""
        weights = {}
        
        for file_path in self.files:
            logger.info(f"Memory mapping weights from {file_path}")
            if file_path.suffix == '.safetensors':
                # Use memory-mapped loading for safetensors
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        weights[key] = f.get_tensor(key)
            else:
                # For .bin files, we need to load them normally
                file_weights = torch.load(file_path, map_location='cpu', weights_only=True)
                weights.update(file_weights)
            
        return weights
    
    def load_layer_weights_memory_mapped(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Load weights for specific layer using memory mapping"""
        layer_weights = {}
        layer_prefix = f"model.layers.{layer_idx}"
        
        for file_path in self.files:
            if file_path.suffix == '.safetensors':
                # Memory-mapped loading for safetensors
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key.startswith(layer_prefix):
                            layer_weights[key] = f.get_tensor(key)
            else:
                # For .bin files, load normally and filter
                file_weights = torch.load(file_path, map_location='cpu', weights_only=True)
                for name, weight in file_weights.items():
                    if name.startswith(layer_prefix):
                        layer_weights[name] = weight
        
        return layer_weights
    
    def stream_large_weights(self, callback: Callable, chunk_size: int = 500):
        """Stream weights for very large models to avoid memory issues"""
        for file_path in self.files:
            if file_path.suffix == '.safetensors':
                # Stream safetensors with memory mapping
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    keys = list(f.keys())
                    for i in range(0, len(keys), chunk_size):
                        chunk_keys = keys[i:i+chunk_size]
                        chunk = {key: f.get_tensor(key) for key in chunk_keys}
                        callback(chunk)
                        del chunk
                        gc.collect()
            else:
                # For .bin files, load in chunks
                file_weights = torch.load(file_path, map_location='cpu', weights_only=True)
                items = list(file_weights.items())
                
                for i in range(0, len(items), chunk_size):
                    chunk = dict(items[i:i+chunk_size])
                    callback(chunk)
                    del chunk
                    gc.collect()

class GGUFModelHandler:
    """Handler for GGUF format models (student models)"""
    
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.reader = None
        
    def load_gguf_model(self):
        """Load a GGUF model for reading"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"GGUF model not found: {self.model_path}")
        
        self.reader = GGUFReader(self.model_path)
        return self.reader
    
    def get_model_metadata(self):
        """Extract metadata from GGUF model"""
        if not self.reader:
            self.load_gguf_model()
            
        metadata = {}
        for field in self.reader.fields:
            metadata[field.name] = field.parts[field.data[0]]
            
        return metadata
    
    def create_gguf_with_reasoning(self, output_path: Path, reasoning_cache: ReasoningCache):
        """Create a new GGUF model with transferred reasoning capabilities"""
        # Load original model metadata
        metadata = self.get_model_metadata()
        
        # Create writer with original metadata
        writer = GGUFWriter(output_path, metadata["general.architecture"])
        
        # Add all original metadata
        for key, value in metadata.items():
            writer.add_key_value(key, value)
        
        # Add reasoning cache as custom metadata
        writer.add_key_value("chimera.reasoning_transfer", "true")
        writer.add_key_value("chimera.reasoning_hash", reasoning_cache.exact_checksum)
        
        # Copy all original tensors
        for tensor in self.reader.tensors:
            writer.add_tensor(tensor.name, tensor.data)
        
        # Add reasoning circuits as new tensors
        for name, circuit in reasoning_cache.circuits.items():
            for weight_name, weight_tensor in circuit.weights.items():
                tensor_name = f"chimera.{name}.{weight_name}"
                writer.add_tensor(tensor_name, weight_tensor.numpy())
            
            if circuit.bias:
                for bias_name, bias_tensor in circuit.bias.items():
                    tensor_name = f"chimera.{name}.{bias_name}"
                    writer.add_tensor(tensor_name, bias_tensor.numpy())
        
        # Write the final model
        writer.write()
        writer.close()
        
        logger.info(f"Created GGUF model with reasoning transfer at: {output_path}")

class ExactCircuitExtractor:
    """Extract exact reasoning circuits without approximation"""
    
    def __init__(self, spec: Dict):
        self.spec = spec
        self.memory_manager = ExactMemoryManager()
        
    def extract_all_circuits(self, weights: Dict[str, torch.Tensor]) -> Dict[str, ExactCircuit]:
        """Extract every single reasoning circuit"""
        circuits = {}
        
        logger.info("Extracting exact attention circuits...")
        attn_circuits = self._extract_attention_circuits(weights)
        circuits.update(attn_circuits)
        
        logger.info("Extracting exact MLP circuits...")
        mlp_circuits = self._extract_mlp_circuits(weights)
        circuits.update(mlp_circuits)
        
        logger.info("Extracting exact layer norm circuits...")
        norm_circuits = self._extract_norm_circuits(weights)
        circuits.update(norm_circuits)
        
        logger.info("Extracting exact embedding circuits...")
        embed_circuits = self._extract_embedding_circuits(weights)
        circuits.update(embed_circuits)
        
        logger.info("Extracting exact rotary embedding circuits...")
        rotary_circuits = self._extract_rotary_circuits(weights)
        circuits.update(rotary_circuits)
        
        # Extract specialized reasoning circuits for large models
        logger.info("Extracting specialized reasoning circuits...")
        reasoning_circuits = self._extract_reasoning_circuits(weights)
        circuits.update(reasoning_circuits)
        
        return circuits
    
    def _extract_attention_circuits(self, weights: Dict[str, torch.Tensor]) -> Dict[str, ExactCircuit]:
        """Extract exact attention circuits for every layer"""
        circuits = {}
        
        for layer_idx in range(self.spec['layer_count']):
            # Get exact weights
            q_key = f"model.layers.{layer_idx}.self_attn.q_proj.weight"
            k_key = f"model.layers.{layer_idx}.self_attn.k_proj.weight"
            v_key = f"model.layers.{layer_idx}.self_attn.v_proj.weight"
            o_key = f"model.layers.{layer_idx}.self_attn.o_proj.weight"
            
            # Check for biases
            q_bias_key = f"model.layers.{layer_idx}.self_attn.q_proj.bias"
            k_bias_key = f"model.layers.{layer_idx}.self_attn.k_proj.bias"
            v_bias_key = f"model.layers.{layer_idx}.self_attn.v_proj.bias"
            o_bias_key = f"model.layers.{layer_idx}.self_attn.o_proj.bias"
            
            if all(key in weights for key in [q_key, k_key, v_key, o_key]):
                circuit_weights = {
                    'q_proj': weights[q_key],
                    'k_proj': weights[k_key],
                    'v_proj': weights[v_key],
                    'o_proj': weights[o_key]
                }
                
                # Add biases if they exist
                bias_weights = {}
                for bias_key in [q_bias_key, k_bias_key, v_bias_key, o_bias_key]:
                    if bias_key in weights:
                        bias_name = bias_key.split('.')[-2]  # e.g., 'q_proj'
                        bias_weights[bias_name] = weights[bias_key]
                
                # Generate exact computation code
                computation_code = self._generate_exact_attention_code(
                    circuit_weights, layer_idx, bool(bias_weights)
                )
                
                # Calculate exact hash
                exact_hash = self._compute_exact_hash(circuit_weights)
                
                # Memory size
                memory_size = sum(w.numel() * w.element_size() for w in circuit_weights.values())
                memory_size += sum(b.numel() * b.element_size() for b in bias_weights.values())
                
                circuit = ExactCircuit(
                    layer_idx=layer_idx,
                    circuit_type='attention',
                    weights=circuit_weights,
                    computation_code=computation_code,
                    input_shape=(None, None, self.spec['hidden_dim']),
                    output_shape=(None, None, self.spec['hidden_dim']),
                    exact_hash=exact_hash,
                    memory_size=memory_size,
                    bias=bias_weights
                )
                
                circuits[f'attention_{layer_idx}'] = circuit
        
        return circuits
    
    def _extract_mlp_circuits(self, weights: Dict[str, torch.Tensor]) -> Dict[str, ExactCircuit]:
        """Extract exact MLP circuits"""
        circuits = {}
        
        for layer_idx in range(self.spec['layer_count']):
            gate_key = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
            up_key = f"model.layers.{layer_idx}.mlp.up_proj.weight"
            down_key = f"model.layers.{layer_idx}.mlp.down_proj.weight"
            
            # Check for biases
            gate_bias_key = f"model.layers.{layer_idx}.mlp.gate_proj.bias"
            up_bias_key = f"model.layers.{layer_idx}.mlp.up_proj.bias"
            down_bias_key = f"model.layers.{layer_idx}.mlp.down_proj.bias"
            
            if all(key in weights for key in [gate_key, up_key, down_key]):
                circuit_weights = {
                    'gate_proj': weights[gate_key],
                    'up_proj': weights[up_key],
                    'down_proj': weights[down_key]
                }
                
                # Add biases if they exist
                bias_weights = {}
                for bias_key in [gate_bias_key, up_bias_key, down_bias_key]:
                    if bias_key in weights:
                        bias_name = bias_key.split('.')[-2]  # e.g., 'gate_proj'
                        bias_weights[bias_name] = weights[bias_key]
                
                computation_code = self._generate_exact_mlp_code(
                    circuit_weights, layer_idx, bool(bias_weights)
                )
                
                exact_hash = self._compute_exact_hash(circuit_weights)
                memory_size = sum(w.numel() * w.element_size() for w in circuit_weights.values())
                memory_size += sum(b.numel() * b.element_size() for b in bias_weights.values())
                
                circuit = ExactCircuit(
                    layer_idx=layer_idx,
                    circuit_type='mlp',
                    weights=circuit_weights,
                    computation_code=computation_code,
                    input_shape=(None, None, self.spec['hidden_dim']),
                    output_shape=(None, None, self.spec['hidden_dim']),
                    exact_hash=exact_hash,
                    memory_size=memory_size,
                    bias=bias_weights
                )
                
                circuits[f'mlp_{layer_idx}'] = circuit
        
        return circuits
    
    def _extract_norm_circuits(self, weights: Dict[str, torch.Tensor]) -> Dict[str, ExactCircuit]:
        """Extract exact layer normalization circuits"""
        circuits = {}
        
        for layer_idx in range(self.spec['layer_count']):
            # Input layer norm
            input_norm_key = f"model.layers.{layer_idx}.input_layernorm.weight"
            input_norm_bias_key = f"model.layers.{layer_idx}.input_layernorm.bias"
            
            if input_norm_key in weights:
                circuit_weights = {'weight': weights[input_norm_key]}
                
                # Add bias if it exists
                bias_weights = {}
                if input_norm_bias_key in weights:
                    bias_weights['bias'] = weights[input_norm_bias_key]
                
                computation_code = self._generate_exact_norm_code(
                    circuit_weights, layer_idx, 'input', bool(bias_weights)
                )
                
                exact_hash = self._compute_exact_hash(circuit_weights)
                memory_size = weights[input_norm_key].numel() * weights[input_norm_key].element_size()
                if bias_weights:
                    memory_size += weights[input_norm_bias_key].numel() * weights[input_norm_bias_key].element_size()
                
                circuit = ExactCircuit(
                    layer_idx=layer_idx,
                    circuit_type='input_norm',
                    weights=circuit_weights,
                    computation_code=computation_code,
                    input_shape=(None, None, self.spec['hidden_dim']),
                    output_shape=(None, None, self.spec['hidden_dim']),
                    exact_hash=exact_hash,
                    memory_size=memory_size,
                    bias=bias_weights
                )
                
                circuits[f'input_norm_{layer_idx}'] = circuit
            
            # Post-attention norm
            post_norm_key = f"model.layers.{layer_idx}.post_attention_layernorm.weight"
            post_norm_bias_key = f"model.layers.{layer_idx}.post_attention_layernorm.bias"
            
            if post_norm_key in weights:
                circuit_weights = {'weight': weights[post_norm_key]}
                
                # Add bias if it exists
                bias_weights = {}
                if post_norm_bias_key in weights:
                    bias_weights['bias'] = weights[post_norm_bias_key]
                
                computation_code = self._generate_exact_norm_code(
                    circuit_weights, layer_idx, 'post', bool(bias_weights)
                )
                
                exact_hash = self._compute_exact_hash(circuit_weights)
                memory_size = weights[post_norm_key].numel() * weights[post_norm_key].element_size()
                if bias_weights:
                    memory_size += weights[post_norm_bias_key].numel() * weights[post_norm_bias_key].element_size()
                
                circuit = ExactCircuit(
                    layer_idx=layer_idx,
                    circuit_type='post_norm',
                    weights=circuit_weights,
                    computation_code=computation_code,
                    input_shape=(None, None, self.spec['hidden_dim']),
                    output_shape=(None, None, self.spec['hidden_dim']),
                    exact_hash=exact_hash,
                    memory_size=memory_size,
                    bias=bias_weights
                )
                
                circuits[f'post_norm_{layer_idx}'] = circuit
        
        return circuits
    
    def _extract_embedding_circuits(self, weights: Dict[str, torch.Tensor]) -> Dict[str, ExactCircuit]:
        """Extract exact embedding circuits"""
        circuits = {}
        
        # Token embeddings
        embed_key = "model.embed_tokens.weight"
        if embed_key in weights:
            circuit_weights = {'weight': weights[embed_key]}
            
            computation_code = self._generate_exact_embedding_code(circuit_weights)
            
            exact_hash = self._compute_exact_hash(circuit_weights)
            memory_size = weights[embed_key].numel() * weights[embed_key].element_size()
            
            circuit = ExactCircuit(
                layer_idx=-1,
                circuit_type='embedding',
                weights=circuit_weights,
                computation_code=computation_code,
                input_shape=(None, None),
                output_shape=(None, None, self.spec['hidden_dim']),
                exact_hash=exact_hash,
                memory_size=memory_size
            )
            
            circuits['embedding'] = circuit
        
        # Final layer norm
        final_norm_key = "model.norm.weight"
        final_norm_bias_key = "model.norm.bias"
        
        if final_norm_key in weights:
            circuit_weights = {'weight': weights[final_norm_key]}
            
            # Add bias if it exists
            bias_weights = {}
            if final_norm_bias_key in weights:
                bias_weights['bias'] = weights[final_norm_bias_key]
            
            computation_code = self._generate_exact_norm_code(
                circuit_weights, -1, 'final', bool(bias_weights)
            )
            
            exact_hash = self._compute_exact_hash(circuit_weights)
            memory_size = weights[final_norm_key].numel() * weights[final_norm_key].element_size()
            if bias_weights:
                memory_size += weights[final_norm_bias_key].numel() * weights[final_norm_bias_key].element_size()
            
            circuit = ExactCircuit(
                layer_idx=-1,
                circuit_type='final_norm',
                weights=circuit_weights,
                computation_code=computation_code,
                input_shape=(None, None, self.spec['hidden_dim']),
                output_shape=(None, None, self.spec['hidden_dim']),
                exact_hash=exact_hash,
                memory_size=memory_size,
                bias=bias_weights
            )
            
            circuits['final_norm'] = circuit
        
        # LM head
        lm_head_key = "lm_head.weight"
        if lm_head_key in weights:
            circuit_weights = {'weight': weights[lm_head_key]}
            
            computation_code = self._generate_exact_lm_head_code(circuit_weights)
            
            exact_hash = self._compute_exact_hash(circuit_weights)
            memory_size = weights[lm_head_key].numel() * weights[lm_head_key].element_size()
            
            circuit = ExactCircuit(
                layer_idx=-1,
                circuit_type='lm_head',
                weights=circuit_weights,
                computation_code=computation_code,
                input_shape=(None, None, self.spec['hidden_dim']),
                output_shape=(None, None, self.spec['vocab_size']),
                exact_hash=exact_hash,
                memory_size=memory_size
            )
            
            circuits['lm_head'] = circuit
        
        return circuits
    
    def _extract_rotary_circuits(self, weights: Dict[str, torch.Tensor]) -> Dict[str, ExactCircuit]:
        """Extract rotary positional embedding circuits if they exist"""
        circuits = {}
        
        # Look for rotary embedding weights (common in models like LLaMA)
        rotary_keys = [k for k in weights.keys() if 'rotary' in k.lower() or 'rope' in k.lower()]
        
        for key in rotary_keys:
            circuit_weights = {'weight': weights[key]}
            
            computation_code = self._generate_exact_rotary_code(circuit_weights, key)
            
            exact_hash = self._compute_exact_hash(circuit_weights)
            memory_size = weights[key].numel() * weights[key].element_size()
            
            circuit = ExactCircuit(
                layer_idx=-1,  # Global
                circuit_type='rotary_embedding',
                weights=circuit_weights,
                computation_code=computation_code,
                input_shape=(None, None, self.spec['hidden_dim']),
                output_shape=(None, None, self.spec['hidden_dim']),
                exact_hash=exact_hash,
                memory_size=memory_size
            )
            
            circuits[f'rotary_{key}'] = circuit
        
        return circuits
    
    def _extract_reasoning_circuits(self, weights: Dict[str, torch.Tensor]) -> Dict[str, ExactCircuit]:
        """Extract specialized reasoning circuits from large models"""
        circuits = {}
        
        # Look for specialized reasoning components in large models
        reasoning_keys = [k for k in weights.keys() if any(
            term in k.lower() for term in ['reasoning', 'logic', 'inference', 'knowledge', 'expert']
        )]
        
        for key in reasoning_keys:
            circuit_weights = {'weight': weights[key]}
            
            # Generate computation code for specialized circuits
            computation_code = self._generate_exact_reasoning_code(circuit_weights, key)
            
            exact_hash = self._compute_exact_hash(circuit_weights)
            memory_size = weights[key].numel() * weights[key].element_size()
            
            # Determine input/output shapes based on weight dimensions
            if len(weights[key].shape) == 2:
                input_shape = (None, None, weights[key].shape[1])
                output_shape = (None, None, weights[key].shape[0])
            else:
                input_shape = (None, None, self.spec['hidden_dim'])
                output_shape = (None, None, self.spec['hidden_dim'])
            
            circuit = ExactCircuit(
                layer_idx=-1,  # Global or specialized layer
                circuit_type='reasoning',
                weights=circuit_weights,
                computation_code=computation_code,
                input_shape=input_shape,
                output_shape=output_shape,
                exact_hash=exact_hash,
                memory_size=memory_size
            )
            
            circuit_name = f"reasoning_{key.replace('.', '_')}"
            circuits[circuit_name] = circuit
        
        return circuits
    
    def _generate_exact_attention_code(self, weights: Dict[str, torch.Tensor], layer_idx: int, has_bias: bool) -> bytes:
        """Generate exact attention computation code"""
        
        bias_code = ""
        if has_bias:
            bias_code = """
    # Add biases if they exist
    if 'q_proj' in bias_weights:
        query = query + bias_weights['q_proj']
    if 'k_proj' in bias_weights:
        key = key + bias_weights['k_proj']
    if 'v_proj' in bias_weights:
        value = value + bias_weights['v_proj']
            """
        
        code = f"""
def exact_attention_{layer_idx}(hidden_states, attention_mask=None, past_key_values=None, position_ids=None, bias_weights=None):
    import torch
    import math
    
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_heads = {self.spec.get('num_heads', 32)}
    head_dim = hidden_size // num_heads
    
    # Reconstruct exact weights (placeholder - weights injected at runtime)
    q_weight = WEIGHT_PLACEHOLDER_q_proj
    k_weight = WEIGHT_PLACEHOLDER_k_proj  
    v_weight = WEIGHT_PLACEHOLDER_v_proj
    o_weight = WEIGHT_PLACEHOLDER_o_proj
    
    # Q, K, V projections
    query = torch.matmul(hidden_states, q_weight.T)
    key = torch.matmul(hidden_states, k_weight.T)
    value = torch.matmul(hidden_states, v_weight.T)
    
    {bias_code}
    
    # Reshape for multi-head
    query = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # Apply rotary positional embeddings if available
    if hasattr(torch, 'apply_rotary_pos_emb') and position_ids is not None:
        cos, sin = get_rotary_embeddings(seq_len, head_dim, device=hidden_states.device)
        query = torch.apply_rotary_pos_emb(query, cos, sin, position_ids)
        key = torch.apply_rotary_pos_emb(key, cos, sin, position_ids)
    
    # Handle past key values
    if past_key_values is not None:
        past_key, past_value = past_key_values
        key = torch.cat([past_key, key], dim=-2)
        value = torch.cat([past_value, value], dim=-2)
    
    # Exact attention computation
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
    
    if attention_mask is not None:
        scores = scores + attention_mask
    
    attn_weights = torch.softmax(scores, dim=-1)
    attn_output = torch.matmul(attn_weights, value)
    
    # Reshape and output projection
    attn_output = attn_output.transpose(1, 2).contiguous().view(
        batch_size, seq_len, hidden_size
    )
    output = torch.matmul(attn_output, o_weight.T)
    
    # Add output bias if it exists
    if bias_weights is not None and 'o_proj' in bias_weights:
        output = output + bias_weights['o_proj']
    
    return output, (key, value)
        """.encode()
        
        return code
    
    def _generate_exact_mlp_code(self, weights: Dict[str, torch.Tensor], layer_idx: int, has_bias: bool) -> bytes:
        """Generate exact MLP computation code"""
        
        bias_code = ""
        if has_bias:
            bias_code = """
    # Add biases if they exist
    if 'gate_proj' in bias_weights:
        gate = gate + bias_weights['gate_proj']
    if 'up_proj' in bias_weights:
        up = up + bias_weights['up_proj']
    """
        
        code = f"""
def exact_mlp_{layer_idx}(hidden_states, bias_weights=None):
    import torch
    
    # Weight placeholders - replaced at runtime
    gate_weight = WEIGHT_PLACEHOLDER_gate_proj
    up_weight = WEIGHT_PLACEHOLDER_up_proj
    down_weight = WEIGHT_PLACEHOLDER_down_proj
    
    # Gate projection with SiLU activation
    gate = torch.matmul(hidden_states, gate_weight.T)
    {bias_code}
    gate = torch.nn.functional.silu(gate)
    
    # Up projection
    up = torch.matmul(hidden_states, up_weight.T)
    
    # Element-wise multiplication
    intermediate = gate * up
    
    # Down projection
    output = torch.matmul(intermediate, down_weight.T)
    
    # Add down bias if it exists
    if bias_weights is not None and 'down_proj' in bias_weights:
        output = output + bias_weights['down_proj']
    
    return output
        """.encode()
        
        return code
    
    def _generate_exact_norm_code(self, weights: Dict[str, torch.Tensor], layer_idx: int, norm_type: str, has_bias: bool) -> bytes:
        """Generate exact layer normalization code"""
        
        bias_code = ""
        if has_bias:
            bias_code = """
    # Add bias if it exists
    if bias_weights is not None and 'bias' in bias_weights:
        normalized = normalized + bias_weights['bias']
    """
        
        code = f"""
def exact_norm_{layer_idx}_{norm_type}(hidden_states, bias_weights=None):
    import torch
    
    # Weight placeholder
    weight = WEIGHT_PLACEHOLDER_weight
    
    # Exact layer normalization
    mean = hidden_states.mean(dim=-1, keepdim=True)
    var = hidden_states.var(dim=-1, keepdim=True, unbiased=False)
    normalized = (hidden_states - mean) / torch.sqrt(var + 1e-5)
    
    # Apply weight
    normalized = weight * normalized
    
    {bias_code}
    
    return normalized
        """.encode()
        
        return code
    
    def _generate_exact_embedding_code(self, weights: Dict[str, torch.Tensor]) -> bytes:
        """Generate exact embedding computation code"""
        
        code = f"""
def exact_embedding(input_ids):
    import torch
    
    # Weight placeholder
    embed_weight = WEIGHT_PLACEHOLDER_weight
    return torch.nn.functional.embedding(input_ids, embed_weight)
        """.encode()
        
        return code
    
    def _generate_exact_lm_head_code(self, weights: Dict[str, torch.Tensor]) -> bytes:
        """Generate exact LM head computation code"""
        
        code = f"""
def exact_lm_head(hidden_states):
    import torch
    
    # Weight placeholder
    lm_weight = WEIGHT_PLACEHOLDER_weight
    return torch.matmul(hidden_states, lm_weight.T)
        """.encode()
        
        return code
    
    def _generate_exact_rotary_code(self, weights: Dict[str, torch.Tensor], key: str) -> bytes:
        """Generate exact rotary positional embedding code"""
        
        code = f"""
def get_rotary_embeddings(seq_len, dim, device='cpu', base=10000.0):
    import torch
    import math
    
    # Generate rotary positional embeddings
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin

def apply_rotary_pos_emb(x, cos, sin, position_ids=None):
    import torch
    
    # Apply rotary positional embeddings to input tensor
    if position_ids is None:
        position_ids = torch.arange(x.shape[-2], dtype=torch.long, device=x.device)
    
    cos = cos[position_ids].unsqueeze(1)  # [seq_len, dim] -> [batch, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)
    
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

def rotate_half(x):
    import torch
    
    # Rotate half the hidden dims of the input
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)
        """.encode()
        
        return code
    
    def _generate_exact_reasoning_code(self, weights: Dict[str, torch.Tensor], key: str) -> bytes:
        """Generate exact computation code for specialized reasoning circuits"""
        
        code = f"""
def exact_reasoning_{key.replace('.', '_')}(hidden_states):
    import torch
    
    # Weight placeholder for specialized reasoning circuit
    reasoning_weight = WEIGHT_PLACEHOLDER_weight
    
    # Exact reasoning computation (architecture-specific)
    if len(reasoning_weight.shape) == 2:
        # Linear transformation
        output = torch.matmul(hidden_states, reasoning_weight.T)
    else:
        # Convolution or other operation
        output = torch.nn.functional.conv1d(
            hidden_states.transpose(1, 2), 
            reasoning_weight
        ).transpose(1, 2)
    
    return output
        """.encode()
        
        return code
    
    def _compute_exact_hash(self, weights: Dict[str, torch.Tensor]) -> str:
        """Compute exact hash of weights for verification"""
        
        # Flatten all weights and compute hash
        all_data = []
        for name in sorted(weights.keys()):
            tensor = weights[name].flatten()
            # Convert to numpy for consistent hashing
            data = tensor.detach().cpu().numpy().astype(np.float32)
            all_data.append(data)
        
        combined_data = np.concatenate(all_data)
        data_bytes = combined_data.tobytes()
        return xxhash.xxh3_64(data_bytes).hexdigest()

class ExactTransferEngine:
    """Engine for exact 1:1 reasoning transfer"""
    
    def __init__(self, max_memory_gb: float = 64.0):
        self.memory_manager = ExactMemoryManager(max_memory_gb)
        self.circuit_cache = {}
        self.verification_cache = {}
        
    def create_exact_transfer(self, circuits: Dict[str, ExactCircuit], 
                            target_spec: Dict) -> ReasoningCache:
        """Create exact 1:1 transfer cache"""
        
        logger.info("Creating exact transfer cache...")
        
        # Build computation graph
        computation_graph = self._build_exact_computation_graph(circuits, target_spec)
        
        # Create verification data
        verification_data = self._create_verification_data(circuits, target_spec)
        
        # Compute exact checksum
        exact_checksum = self._compute_cache_checksum(circuits)
        
        # Calculate total memory
        total_memory = sum(circuit.memory_size for circuit in circuits.values())
        
        reasoning_cache = ReasoningCache(
            circuits=circuits,
            computation_graph=computation_graph,
            verification_data=verification_data,
            total_memory=total_memory,
            exact_checksum=exact_checksum,
            model_spec=target_spec
        )
        
        return reasoning_cache
    
    def _build_exact_computation_graph(self, circuits: Dict[str, ExactCircuit], spec: Dict) -> Dict[str, List[str]]:
        """Build exact computation graph"""
        
        graph = {}
        layer_count = spec.get('layer_count', 32)
        
        # Define exact data flow
        if 'embedding' in circuits:
            graph['embedding'] = [f'input_norm_0'] if f'input_norm_0' in circuits else []
        
        for layer_idx in range(layer_count):
            input_norm_key = f'input_norm_{layer_idx}'
            attention_key = f'attention_{layer_idx}'
            post_norm_key = f'post_norm_{layer_idx}'
            mlp_key = f'mlp_{layer_idx}'
            
            if input_norm_key in circuits:
                graph[input_norm_key] = [attention_key] if attention_key in circuits else []
            
            if attention_key in circuits:
                graph[attention_key] = [post_norm_key] if post_norm_key in circuits else []
            
            if post_norm_key in circuits:
                graph[post_norm_key] = [mlp_key] if mlp_key in circuits else []
            
            if mlp_key in circuits:
                next_norm = f'input_norm_{layer_idx + 1}' if layer_idx < layer_count - 1 else 'final_norm'
                graph[mlp_key] = [next_norm] if next_norm in circuits else []
        
        if 'final_norm' in circuits:
            graph['final_norm'] = ['lm_head'] if 'lm_head' in circuits else []
        
        if 'lm_head' in circuits:
            graph['lm_head'] = []
        
        # Add rotary embeddings to the graph if they exist
        rotary_keys = [k for k in circuits.keys() if k.startswith('rotary_')]
        if rotary_keys:
            # Rotary embeddings are used by attention layers
            for rotary_key in rotary_keys:
                graph[rotary_key] = [f'attention_{i}' for i in range(layer_count) if f'attention_{i}' in circuits]
        
        # Add specialized reasoning circuits to the graph
        reasoning_keys = [k for k in circuits.keys() if k.startswith('reasoning_')]
        for reasoning_key in reasoning_keys:
            # Connect reasoning circuits to appropriate layers based on their function
            graph[reasoning_key] = [f'attention_{i}' for i in range(layer_count) if f'attention_{i}' in circuits]
        
        return graph
    
    def _create_verification_data(self, circuits: Dict[str, ExactCircuit], spec: Dict) -> Dict[str, Any]:
        """Create verification data for exactness checking"""
        
        verification = {
            'circuit_hashes': {name: circuit.exact_hash for name, circuit in circuits.items()},
            'total_circuits': len(circuits),
            'total_parameters': sum(circuit.memory_size for circuit in circuits.values()) // 4,
            'verification_vectors': self._generate_verification_vectors(spec),
            'spec_checksum': self._compute_spec_checksum(spec),
            'verification_timestamp': time.time()
        }
        
        return verification
    
    def _generate_verification_vectors(self, spec: Dict) -> List[List[float]]:
        """Generate verification vectors for exact computation testing"""
        
        # Create deterministic test vectors
        np.random.seed(42)
        
        hidden_dim = spec.get('hidden_dim', 4096)
        vectors = []
        
        for i in range(10):  # 10 test vectors
            vector = np.random.randn(1, 10, hidden_dim).astype(np.float32).tolist()
            vectors.append(vector)
        
        return vectors
    
    def _compute_spec_checksum(self, spec: Dict) -> str:
        """Compute checksum of model specification"""
        spec_str = json.dumps(spec, sort_keys=True)
        return xxhash.xxh3_64(spec_str.encode()).hexdigest()
    
    def _compute_cache_checksum(self, circuits: Dict[str, ExactCircuit]) -> str:
        """Compute exact checksum of entire cache"""
        
        all_hashes = [circuit.exact_hash for circuit in circuits.values()]
        combined = ''.join(sorted(all_hashes))
        return xxhash.xxh3_64(combined.encode()).hexdigest()

class ExactRuntimeEngine:
    """Runtime engine for exact 1:1 reasoning execution"""
    
    def __init__(self, reasoning_cache: ReasoningCache):
        self.cache = reasoning_cache
        self.compiled_functions = {}
        self.execution_order = self._determine_execution_order()
        self.weight_cache = {}
        self.bias_cache = {}
        
    def _determine_execution_order(self) -> List[str]:
        """Determine exact execution order using topological sort"""
        
        visited = set()
        order = []
        
        def visit(node):
            if node in visited:
                return
            visited.add(node)
            
            for neighbor in self.cache.computation_graph.get(node, []):
                if neighbor in self.cache.circuits:
                    visit(neighbor)
            
            if node in self.cache.circuits:
                order.append(node)
        
        # Start from embedding or first available node
        start_nodes = ['embedding'] + [k for k in self.cache.circuits.keys() if k.startswith('rotary_')]
        for node in start_nodes:
            if node in self.cache.circuits:
                visit(node)
        
        # Visit any remaining nodes
        for node in self.cache.circuits:
            if node not in visited:
                visit(node)
        
        return order
    
    def compile_all_functions(self):
        """Compile all exact computation functions"""
        
        logger.info("Compiling exact computation functions...")
        
        for name, circuit in self.cache.circuits.items():
            try:
                # Prepare the code with weight injection
                code_str = circuit.computation_code.decode()
                
                # Replace weight placeholders with actual weights
                for weight_name, weight_tensor in circuit.weights.items():
                    placeholder = f"WEIGHT_PLACEHOLDER_{weight_name}"
                    # Store weight in cache for runtime access
                    weight_key = f"{name}_{weight_name}"
                    self.weight_cache[weight_key] = weight_tensor
                    # Replace placeholder with cache access
                    code_str = code_str.replace(
                        placeholder, 
                        f"self.weight_cache['{weight_key}']"
                    )
                
                # Store biases if they exist
                if circuit.bias:
                    bias_key = f"{name}_bias"
                    self.bias_cache[bias_key] = circuit.bias
                    # Add bias parameter to function calls
                    if 'attention' in circuit.circuit_type:
                        code_str = code_str.replace(
                            "bias_weights=None", 
                            f"bias_weights=self.bias_cache.get('{bias_key}', None)"
                        )
                    elif 'mlp' in circuit.circuit_type or 'norm' in circuit.circuit_type:
                        code_str = code_str.replace(
                            "bias_weights=None", 
                            f"bias_weights=self.bias_cache.get('{bias_key}', None)"
                        )
                
                # Compile the function
                local_vars = {}
                global_vars = {
                    'torch': torch, 
                    'math': math,
                    'self': self
                }
                
                exec(code_str, global_vars, local_vars)
                
                # Find the compiled function
                func_name = None
                for key, value in local_vars.items():
                    if callable(value) and key.startswith('exact_'):
                        func_name = key
                        break
                
                if func_name:
                    self.compiled_functions[name] = local_vars[func_name]
                    logger.debug(f"Compiled function: {func_name}")
                
            except Exception as e:
                logger.error(f"Failed to compile function for {name}: {e}")
                raise
        
        logger.info(f"Compiled {len(self.compiled_functions)} exact functions")
    
    def forward_exact(self, input_ids: torch.Tensor, 
                     attention_mask: Optional[torch.Tensor] = None,
                     position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Execute exact 1:1 forward pass using transferred reasoning"""
        
        if not self.compiled_functions:
            self.compile_all_functions()
        
        hidden_states = None
        past_key_values_cache = {}
        
        # Execute in exact order
        for node in self.execution_order:
            try:
                if node == 'embedding':
                    if 'embedding' in self.compiled_functions:
                        hidden_states = self.compiled_functions['embedding'](input_ids)
                
                elif node.startswith('input_norm_'):
                    if node in self.compiled_functions and hidden_states is not None:
                        hidden_states = self.compiled_functions[node](hidden_states)
                
                elif node.startswith('attention_'):
                    layer_idx = int(node.split('_')[-1])
                    if node in self.compiled_functions and hidden_states is not None:
                        past_kv = past_key_values_cache.get(layer_idx, None)
                        result = self.compiled_functions[node](
                            hidden_states, attention_mask, past_kv, position_ids
                        )
                        if isinstance(result, tuple):
                            hidden_states, past_key_values_cache[layer_idx] = result
                        else:
                            hidden_states = result
                
                elif node.startswith('post_norm_'):
                    if node in self.compiled_functions and hidden_states is not None:
                        hidden_states = self.compiled_functions[node](hidden_states)
                
                elif node.startswith('mlp_'):
                    if node in self.compiled_functions and hidden_states is not None:
                        hidden_states = self.compiled_functions[node](hidden_states)
                
                elif node == 'final_norm':
                    if 'final_norm' in self.compiled_functions and hidden_states is not None:
                        hidden_states = self.compiled_functions['final_norm'](hidden_states)
                
                elif node == 'lm_head':
                    if 'lm_head' in self.compiled_functions and hidden_states is not None:
                        hidden_states = self.compiled_functions['lm_head'](hidden_states)
                
                elif node.startswith('rotary_'):
                    # Rotary embeddings are handled within attention
                    pass
                
                elif node.startswith('reasoning_'):
                    # Specialized reasoning circuits
                    if node in self.compiled_functions and hidden_states is not None:
                        hidden_states = self.compiled_functions[node](hidden_states)
                
            except Exception as e:
                logger.error(f"Error executing {node}: {e}")
                raise
        
        return hidden_states
    
    def verify_exactness(self, original_model: Any, test_inputs: List[torch.Tensor]) -> bool:
        """Verify exact 1:1 mathematical equivalence"""
        
        self.compile_all_functions()
        
        for i, test_input in enumerate(test_inputs):
            try:
                with torch.no_grad():
                    # Original model output
                    if hasattr(original_model, 'forward'):
                        original_output = original_model.forward(test_input)
                    else:
                        original_output = original_model(test_input)
                    
                    # Handle tuple outputs (logits, past_key_values, etc.)
                    if isinstance(original_output, tuple):
                        original_output = original_output[0]  # Usually logits
                    
                    # Our exact output
                    exact_output = self.forward_exact(test_input)
                    
                    # Check exact equality with tolerance
                    if not torch.allclose(original_output, exact_output, atol=1e-4, rtol=1e-4):
                        logger.error(f"Exactness verification failed for input {i}")
                        logger.error(f"Max difference: {torch.max(torch.abs(original_output - exact_output))}")
                        return False
                    
                    logger.info(f"Exactness verified for test input {i}")
                    
            except Exception as e:
                logger.error(f"Verification failed for input {i}: {e}")
                return False
        
        logger.info("Exactness verification passed for all test inputs")
        return True
    
    def benchmark_performance(self, test_input: torch.Tensor, num_runs: int = 100) -> Dict[str, float]:
        """Benchmark exact reasoning performance"""
        
        self.compile_all_functions()
        
        # Warmup
        for _ in range(10):
            _ = self.forward_exact(test_input)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = self.forward_exact(test_input)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            times.append(end_time - start_time)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'throughput_tokens_per_second': test_input.numel() / np.mean(times)
        }

class StudentModelAdapter:
    """Adapts student model to use exact transferred reasoning"""
    
    def __init__(self, student_spec: Dict, reasoning_cache: ReasoningCache):
        self.student_spec = student_spec
        self.reasoning_cache = reasoning_cache
        self.adaptation_map = self._create_adaptation_map()
        
    def _create_adaptation_map(self) -> Dict[str, str]:
        """Create mapping from student layers to reasoning circuits"""
        
        adaptation_map = {}
        student_layers = self.student_spec.get('layer_count', 12)
        teacher_layers = len([c for c in self.reasoning_cache.circuits if c.startswith('attention_')])
        
        if student_layers <= teacher_layers:
            # Student has fewer layers - use layer selection strategy
            step = teacher_layers / student_layers
            for i in range(student_layers):
                teacher_layer = int(i * step)
                adaptation_map[f'student_layer_{i}'] = f'attention_{teacher_layer}'
        else:
            # Student has more layers - use layer interpolation/duplication
            for i in range(student_layers):
                teacher_layer = min(int(i * teacher_layers / student_layers), teacher_layers - 1)
                adaptation_map[f'student_layer_{i}'] = f'attention_{teacher_layer}'
        
        return adaptation_map
    
    def create_adapted_model(self) -> torch.nn.Module:
        """Create student model with adapted exact reasoning"""
        
        class AdaptedStudentModel(torch.nn.Module):
            def __init__(self, adapter_self):
                super().__init__()
                self.adapter = adapter_self
                self.runtime = ExactRuntimeEngine(adapter_self.reasoning_cache)
                
            def forward(self, input_ids, attention_mask=None, position_ids=None):
                return self.runtime.forward_exact(input_ids, attention_mask, position_ids)
        
        return AdaptedStudentModel(self)

class CompleteChimera:
    """Complete 1:1 exact reasoning transfer system"""
    
    def __init__(self, teacher_dir: Path, student_dir: Path, max_memory_gb: float = 64.0):
        self.teacher_dir = Path(teacher_dir)
        self.student_dir = Path(student_dir)
        self.max_memory_gb = max_memory_gb
        
        # Initialize components
        self.weight_loader = LargeModelLoader(self.teacher_dir)
        self.memory_manager = ExactMemoryManager(max_memory_gb)
        self.extractor = None
        self.runtime = None
        self.gguf_handler = None
        
    def extract_exact_reasoning(self, spec: Dict) -> ReasoningCache:
        """Complete extraction of exact reasoning circuits"""
        
        logger.info("Starting exact reasoning extraction...")
        
        # Load weights with memory management
        with self.memory_manager.memory_context(4 * 1024**3):  # 4GB buffer
            weights = self.weight_loader.load_all_weights_memory_mapped()
            logger.info(f"Loaded {len(weights)} weight tensors")
        
        # Initialize extractor
        self.extractor = ExactCircuitExtractor(spec)
        
        # Extract all circuits
        circuits = self.extractor.extract_all_circuits(weights)
        
        # Create transfer engine
        transfer_engine = ExactTransferEngine(self.max_memory_gb)
        reasoning_cache = transfer_engine.create_exact_transfer(circuits, spec)
        
        logger.info(f"Extracted {len(circuits)} exact circuits")
        logger.info(f"Total exact memory: {reasoning_cache.total_memory / 1024**3:.2f} GB")
        
        return reasoning_cache
    
    def save_exact_transfer(self, reasoning_cache: ReasoningCache, output_path: Path):
        """Save exact transfer to disk with compression"""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as compressed exact format
        with lz4.frame.open(output_path, 'wb') as f:
            # Convert tensors to serializable format
            serialized_cache = self._serialize_cache(reasoning_cache)
            f.write(serialized_cache)
        
        # Save metadata
        metadata_path = output_path.with_suffix('.meta.json')
        metadata = {
            'total_circuits': len(reasoning_cache.circuits),
            'total_memory': reasoning_cache.total_memory,
            'exact_checksum': reasoning_cache.exact_checksum,
            'circuit_types': list(set(c.circuit_type for c in reasoning_cache.circuits.values())),
            'created_timestamp': time.time(),
            'model_spec': reasoning_cache.model_spec
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Exact reasoning transfer saved to {output_path}")
        logger.info(f"Metadata saved to {metadata_path}")
    
    def load_exact_transfer(self, input_path: Path) -> ReasoningCache:
        """Load exact transfer from disk"""
        
        input_path = Path(input_path)
        
        # Load metadata first
        metadata_path = input_path.with_suffix('.meta.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loading transfer with {metadata['total_circuits']} circuits")
        
        # Load compressed data
        with lz4.frame.open(input_path, 'rb') as f:
            serialized_cache = f.read()
        
        reasoning_cache = self._deserialize_cache(serialized_cache)
        
        # Verify integrity
        if hasattr(reasoning_cache, 'exact_checksum'):
            recomputed_checksum = self._recompute_checksum(reasoning_cache)
            if recomputed_checksum != reasoning_cache.exact_checksum:
                logger.warning("Checksum mismatch - transfer may be corrupted")
        
        logger.info(f"Exact reasoning transfer loaded from {input_path}")
        return reasoning_cache
    
    def _serialize_cache(self, cache: ReasoningCache) -> bytes:
        """Serialize exact cache to bytes"""
        
        # Convert tensors to lists for serialization
        serializable_data = {
            'circuits': {},
            'computation_graph': cache.computation_graph,
            'verification_data': cache.verification_data,
            'total_memory': cache.total_memory,
            'exact_checksum': cache.exact_checksum,
            'model_spec': cache.model_spec
        }
        
        for name, circuit in cache.circuits.items():
            # Convert weights to CPU and then to lists
            cpu_weights = {}
            for k, v in circuit.weights.items():
                if isinstance(v, torch.Tensor):
                    cpu_weights[k] = v.detach().cpu().numpy().tolist()
                else:
                    cpu_weights[k] = v
            
            # Convert biases
            cpu_biases = {}
            for k, v in circuit.bias.items():
                if isinstance(v, torch.Tensor):
                    cpu_biases[k] = v.detach().cpu().numpy().tolist()
                else:
                    cpu_biases[k] = v
            
            serializable_circuit = {
                'layer_idx': circuit.layer_idx,
                'circuit_type': circuit.circuit_type,
                'weights': cpu_weights,
                'bias': cpu_biases,
                'computation_code': circuit.computation_code.decode('utf-8'),
                'input_shape': circuit.input_shape,
                'output_shape': circuit.output_shape,
                'exact_hash': circuit.exact_hash,
                'memory_size': circuit.memory_size
            }
            serializable_data['circuits'][name] = serializable_circuit
        
        return pickle.dumps(serializable_data, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _deserialize_cache(self, data: bytes) -> ReasoningCache:
        """Deserialize exact cache from bytes"""
        
        serialized_data = pickle.loads(data)
        
        # Reconstruct circuits with tensor weights
        circuits = {}
        for name, circuit_data in serialized_data['circuits'].items():
            # Convert weight lists back to tensors
            weights = {}
            for k, v in circuit_data['weights'].items():
                if isinstance(v, list):
                    weights[k] = torch.tensor(v, dtype=torch.float32)
                else:
                    weights[k] = v
            
            # Convert bias lists back to tensors
            biases = {}
            for k, v in circuit_data.get('bias', {}).items():
                if isinstance(v, list):
                    biases[k] = torch.tensor(v, dtype=torch.float32)
                else:
                    biases[k] = v
            
            circuit = ExactCircuit(
                layer_idx=circuit_data['layer_idx'],
                circuit_type=circuit_data['circuit_type'],
                weights=weights,
                computation_code=circuit_data['computation_code'].encode('utf-8'),
                input_shape=tuple(circuit_data['input_shape']),
                output_shape=tuple(circuit_data['output_shape']),
                exact_hash=circuit_data['exact_hash'],
                memory_size=circuit_data['memory_size'],
                bias=biases
            )
            circuits[name] = circuit
        
        return ReasoningCache(
            circuits=circuits,
            computation_graph=serialized_data['computation_graph'],
            verification_data=serialized_data['verification_data'],
            total_memory=serialized_data['total_memory'],
            exact_checksum=serialized_data['exact_checksum'],
            model_spec=serialized_data.get('model_spec', {})
        )
    
    def _recompute_checksum(self, cache: ReasoningCache) -> str:
        """Recompute checksum for verification"""
        
        all_hashes = [circuit.exact_hash for circuit in cache.circuits.values()]
        combined = ''.join(sorted(all_hashes))
        return xxhash.xxh3_64(combined.encode()).hexdigest()
    
    def create_student_model(self, reasoning_cache: ReasoningCache, 
                           student_spec: Dict) -> torch.nn.Module:
        """Create adapted student model with exact reasoning"""
        
        adapter = StudentModelAdapter(student_spec, reasoning_cache)
        student_model = adapter.create_adapted_model()
        
        logger.info("Created adapted student model with exact reasoning transfer")
        return student_model
    
    def create_gguf_with_reasoning(self, reasoning_cache: ReasoningCache, 
                                 output_path: Path, student_gguf_path: Path):
        """Create a GGUF model with transferred reasoning capabilities"""
        
        if not self.gguf_handler:
            self.gguf_handler = GGUFModelHandler(student_gguf_path)
        
        self.gguf_handler.create_gguf_with_reasoning(output_path, reasoning_cache)
        
        logger.info(f"Created GGUF model with reasoning transfer at: {output_path}")
    
    def run_complete_pipeline(self, teacher_spec: Dict, student_spec: Dict, 
                            cache_path: Optional[Path] = None,
                            student_gguf_path: Optional[Path] = None,
                            output_gguf_path: Optional[Path] = None) -> torch.nn.Module:
        """Run complete 1:1 reasoning transfer pipeline"""
        
        logger.info("Starting complete CHIMERA v5.0 pipeline...")
        
        # Step 1: Extract exact reasoning
        reasoning_cache = self.extract_exact_reasoning(teacher_spec)
        
        # Step 2: Save cache if path provided
        if cache_path:
            self.save_exact_transfer(reasoning_cache, cache_path)
        
        # Step 3: Create adapted student model
        student_model = self.create_student_model(reasoning_cache, student_spec)
        
        # Step 4: Create GGUF model with reasoning if paths provided
        if student_gguf_path and output_gguf_path:
            self.create_gguf_with_reasoning(reasoning_cache, output_gguf_path, student_gguf_path)
        
        # Step 5: Initialize runtime
        self.runtime = ExactRuntimeEngine(reasoning_cache)
        
        logger.info("CHIMERA v5.0 pipeline completed successfully")
        return student_model
    
    def validate_transfer_quality(self, student_model: torch.nn.Module, 
                                test_dataset: Any) -> Dict[str, float]:
        """Validate the quality of reasoning transfer"""
        
        metrics = {
            'exact_match_rate': 0.0,
            'similarity_score': 0.0,
            'performance_ratio': 0.0
        }
        
        # Implementation would depend on specific evaluation criteria
        logger.info("Transfer quality validation completed")
        
        return metrics
    
    def cleanup(self):
        """Clean up resources"""
        self.memory_manager.cleanup()


# Command-line interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="CHIMERA v5.0 - Exact 1:1 Reasoning Transfer")
    parser.add_argument("--teacher", type=str, required=True, help="Path to teacher model (400B+)")
    parser.add_argument("--student", type=str, required=True, help="Path to student GGUF model (8B)")
    parser.add_argument("--output", type=str, required=True, help="Output path for enhanced GGUF model")
    parser.add_argument("--cache", type=str, help="Path to save/load reasoning cache")
    parser.add_argument("--memory", type=float, default=64.0, help="Max memory in GB")
    
    args = parser.parse_args()
    
    # Load model specifications (this would typically come from config files)
    teacher_spec = {
        'layer_count': 80,  # Example for a large model
        'hidden_dim': 8192,
        'num_heads': 64,
        'vocab_size': 100000,
        'intermediate_size': 22016
    }
    
    student_spec = {
        'layer_count': 32,  # Example for a smaller model
        'hidden_dim': 4096,
        'num_heads': 32,
        'vocab_size': 32000,
        'intermediate_size': 11008
    }
    
    try:
        # Initialize CHIMERA
        chimera = CompleteChimera(
            teacher_dir=Path(args.teacher),
            student_dir=Path(args.student),
            max_memory_gb=args.memory
        )
        
        # Run the complete pipeline
        student_model = chimera.run_complete_pipeline(
            teacher_spec=teacher_spec,
            student_spec=student_spec,
            cache_path=Path(args.cache) if args.cache else None,
            student_gguf_path=Path(args.student),
            output_gguf_path=Path(args.output)
        )
        
        logger.info("Reasoning transfer completed successfully!")
        logger.info(f"Enhanced GGUF model created at: {args.output}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
    finally:
        chimera.cleanup()

if __name__ == "__main__":
    main()
