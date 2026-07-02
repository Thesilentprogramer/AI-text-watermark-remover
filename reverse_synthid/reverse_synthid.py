#!/usr/bin/env python3
"""
SynthID Watermark Reversal/Removal Tool

This script provides multiple methods to remove or reduce SynthID watermarks
from AI-generated text. The watermark is embedded in the statistical pattern
of token choices, so we can break it by:

1. Paraphrasing - Using a non-watermarked model to rewrite the text
2. Token substitution - Replacing tokens with synonyms
3. Text shuffling - Reordering sentences/phrases where semantically valid
4. Insertion attacks - Adding/removing filler words

The most effective method is paraphrasing, as it completely regenerates the
token sequence while preserving meaning.

Based on analysis of: https://github.com/google-deepmind/synthid-text
Paper: https://doi.org/10.1038/s41586-024-08025-4

Author: Reverse engineering tool for research purposes only
"""

import argparse
import hashlib
import random
import re
import os
from typing import Optional, List, Tuple
import torch
import numpy as np
import transformers
from reverse_synthid.src.synthid_text import logits_processing

def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')


DEVICE = get_device()


# ============================================================================
# METHOD 1: Paraphrasing Attack (Most Effective)
# ============================================================================

class ParaphrasingAttack:
    """
    Uses a non-watermarked model to paraphrase text, breaking the watermark.
    
    The SynthID watermark works by biasing token selection based on n-gram
    context hashes. By regenerating the text with a different model (or the
    same model without watermarking), we break the statistical pattern.
    """
    
    def __init__(self, model_name: str = "google/gemma-2b-it", device: torch.device = DEVICE):
        """
        Initialize with a non-watermarked model.
        
        Args:
            model_name: HuggingFace model to use for paraphrasing (default: gemma-2b-it)
            device: Torch device
        """
        
        self.device = device
        self.model_name = model_name
        self.is_gemma = 'gemma' in model_name.lower()
        
        print(f"Loading model {model_name} on {device}...")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load the BASE model without any watermarking
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            dtype=torch.bfloat16 if self.is_gemma else (torch.float16 if torch.cuda.is_available() else torch.float32),
        )
        
        print("Model loaded successfully!")
    
    def paraphrase(
        self,
        text: str,
        instruction: str = "Paraphrase the following text while keeping the same meaning. Output only the paraphrased version, nothing else.",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
    ) -> str:
        """
        Paraphrase text to remove watermark.
        
        Args:
            text: Watermarked text to paraphrase
            instruction: Prompt prefix for paraphrasing
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Paraphrased text without watermark
        """
        # Use chat template for Gemma models
        if self.is_gemma:
            messages = [
                {"role": "user", "content": f"{instruction}\n\nText to paraphrase:\n{text}"}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"{instruction}\n\n{text}\n\nParaphrased version:"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode and extract only the generated part
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the paraphrased part (after the prompt)
        if self.is_gemma:
            # For Gemma, the response comes after the model turn
            if "model" in full_text.lower():
                parts = full_text.split("model")
                if len(parts) > 1:
                    paraphrased = parts[-1].strip()
                else:
                    paraphrased = full_text[len(prompt):].strip()
            else:
                paraphrased = full_text[len(prompt):].strip()
        else:
            if "Paraphrased version:" in full_text:
                paraphrased = full_text.split("Paraphrased version:")[-1].strip()
            else:
                paraphrased = full_text[len(prompt):].strip()
        
        return paraphrased
    
    def chunk_and_paraphrase(
        self,
        text: str,
        chunk_size: int = 200,
        overlap: int = 50,
        **kwargs
    ) -> str:
        """
        Paraphrase long text in chunks.
        
        Args:
            text: Full watermarked text
            chunk_size: Words per chunk
            overlap: Overlapping words between chunks
            **kwargs: Additional args for paraphrase()
            
        Returns:
            Full paraphrased text
        """
        words = text.split()
        if len(words) <= chunk_size:
            return self.paraphrase(text, **kwargs)
        
        chunks = []
        i = 0
        while i < len(words):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            i += chunk_size - overlap
        
        paraphrased_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"Paraphrasing chunk {i+1}/{len(chunks)}...")
            paraphrased = self.paraphrase(chunk, **kwargs)
            paraphrased_chunks.append(paraphrased)
        
        # Simple concatenation - could be improved with better merging
        return ' '.join(paraphrased_chunks)


# ============================================================================
# METHOD 2: Token-Level Perturbation Attack
# ============================================================================

class TokenPerturbationAttack:
    """
    Perturbs tokens at specific positions to break n-gram patterns.
    
    SynthID uses n-gram contexts (default ngram_len=5) to compute hash values.
    By inserting, deleting, or substituting tokens, we shift the n-gram
    boundaries and break the watermark signal.
    """
    
    # Common synonym pairs for substitution
    SYNONYMS = {
        'big': ['large', 'huge', 'enormous', 'massive'],
        'small': ['tiny', 'little', 'miniature', 'compact'],
        'good': ['great', 'excellent', 'wonderful', 'fantastic'],
        'bad': ['poor', 'terrible', 'awful', 'horrible'],
        'happy': ['joyful', 'pleased', 'delighted', 'content'],
        'sad': ['unhappy', 'sorrowful', 'melancholic', 'gloomy'],
        'fast': ['quick', 'rapid', 'swift', 'speedy'],
        'slow': ['gradual', 'unhurried', 'leisurely', 'sluggish'],
        'important': ['significant', 'crucial', 'vital', 'essential'],
        'interesting': ['fascinating', 'intriguing', 'engaging', 'compelling'],
        'beautiful': ['lovely', 'gorgeous', 'stunning', 'attractive'],
        'difficult': ['hard', 'challenging', 'tough', 'demanding'],
        'easy': ['simple', 'straightforward', 'effortless', 'uncomplicated'],
        'help': ['assist', 'aid', 'support', 'facilitate'],
        'use': ['utilize', 'employ', 'apply', 'leverage'],
        'make': ['create', 'produce', 'generate', 'construct'],
        'show': ['demonstrate', 'display', 'exhibit', 'reveal'],
        'think': ['believe', 'consider', 'assume', 'suppose'],
        'say': ['state', 'mention', 'express', 'declare'],
        'get': ['obtain', 'acquire', 'receive', 'gain'],
        'the': ['this', 'that'],  # Context-dependent
        'a': ['one', 'some'],
        'is': ['remains', 'appears', 'seems'],
        'are': ['remain', 'appear', 'seem'],
        'was': ['had been', 'appeared'],
        'were': ['had been', 'appeared'],
        'very': ['extremely', 'quite', 'particularly', 'especially'],
        'really': ['truly', 'genuinely', 'certainly', 'definitely'],
        'however': ['nevertheless', 'nonetheless', 'yet', 'still'],
        'also': ['additionally', 'furthermore', 'moreover', 'too'],
        'because': ['since', 'as', 'given that', 'due to the fact that'],
    }
    
    # Filler words that can be inserted or removed
    FILLERS = [
        'actually', 'basically', 'certainly', 'clearly', 'definitely',
        'essentially', 'evidently', 'frankly', 'generally', 'honestly',
        'indeed', 'naturally', 'obviously', 'perhaps', 'possibly',
        'presumably', 'probably', 'seemingly', 'surely', 'typically',
        'undoubtedly', 'usually', 'well',
    ]
    
    def __init__(self, ngram_len: int = 5):
        """
        Initialize perturbation attack.
        
        Args:
            ngram_len: The n-gram length used by SynthID (default 5)
        """
        self.ngram_len = ngram_len
        # Build reverse lookup
        self.reverse_synonyms = {}
        for word, syns in self.SYNONYMS.items():
            for syn in syns:
                if syn not in self.reverse_synonyms:
                    self.reverse_synonyms[syn] = []
                self.reverse_synonyms[syn].append(word)
    
    def substitute_synonyms(
        self,
        text: str,
        substitution_rate: float = 0.3,
        seed: Optional[int] = None,
    ) -> str:
        """
        Replace words with synonyms to break n-gram patterns.
        
        Args:
            text: Input text
            substitution_rate: Fraction of substitutable words to replace
            seed: Random seed for reproducibility
            
        Returns:
            Text with synonym substitutions
        """
        if seed is not None:
            random.seed(seed)
        
        words = text.split()
        result = []
        
        for word in words:
            word_lower = word.lower().strip('.,!?;:')
            
            # Check if we should substitute
            should_substitute = random.random() < substitution_rate
            
            if should_substitute:
                # Check direct synonyms
                if word_lower in self.SYNONYMS:
                    synonym = random.choice(self.SYNONYMS[word_lower])
                    # Preserve original capitalization
                    if word[0].isupper():
                        synonym = synonym.capitalize()
                    # Preserve trailing punctuation
                    trailing = ''
                    for char in word[::-1]:
                        if char in '.,!?;:':
                            trailing = char + trailing
                        else:
                            break
                    result.append(synonym + trailing)
                    continue
                
                # Check reverse synonyms
                elif word_lower in self.reverse_synonyms:
                    synonym = random.choice(self.reverse_synonyms[word_lower])
                    if word[0].isupper():
                        synonym = synonym.capitalize()
                    trailing = ''
                    for char in word[::-1]:
                        if char in '.,!?;:':
                            trailing = char + trailing
                        else:
                            break
                    result.append(synonym + trailing)
                    continue
            
            result.append(word)
        
        return ' '.join(result)
    
    def insert_fillers(
        self,
        text: str,
        insertion_rate: float = 0.1,
        seed: Optional[int] = None,
    ) -> str:
        """
        Insert filler words to shift n-gram boundaries.
        
        Args:
            text: Input text
            insertion_rate: Probability of inserting after each word
            seed: Random seed
            
        Returns:
            Text with inserted filler words
        """
        if seed is not None:
            random.seed(seed)
        
        words = text.split()
        result = []
        
        for i, word in enumerate(words):
            result.append(word)
            
            # Insert after sentence boundaries or at random positions
            if random.random() < insertion_rate:
                # Prefer insertion after certain word patterns
                if word.endswith((',', ';')) or (i > 0 and i < len(words) - 1):
                    filler = random.choice(self.FILLERS)
                    result.append(filler)
        
        return ' '.join(result)
    
    def remove_fillers(self, text: str) -> str:
        """
        Remove common filler words to shift n-gram boundaries.
        
        Args:
            text: Input text
            
        Returns:
            Text with fillers removed
        """
        words = text.split()
        result = []
        
        for word in words:
            word_lower = word.lower().strip('.,!?;:')
            if word_lower not in self.FILLERS:
                result.append(word)
        
        return ' '.join(result)
    
    def perturb(
        self,
        text: str,
        substitution_rate: float = 0.3,
        insertion_rate: float = 0.05,
        remove_fillers: bool = False,
        seed: Optional[int] = None,
    ) -> str:
        """
        Apply multiple perturbation strategies.
        
        Args:
            text: Input text
            substitution_rate: Rate of synonym substitution
            insertion_rate: Rate of filler insertion
            remove_fillers: Whether to remove existing fillers
            seed: Random seed
            
        Returns:
            Perturbed text
        """
        result = text
        
        if remove_fillers:
            result = self.remove_fillers(result)
        
        result = self.substitute_synonyms(result, substitution_rate, seed)
        
        if insertion_rate > 0:
            result = self.insert_fillers(result, insertion_rate, seed)
        
        return result


# ============================================================================
# METHOD 3: Sentence-Level Shuffling Attack
# ============================================================================

class SentenceShufflingAttack:
    """
    Shuffles sentences or clauses to break n-gram patterns at boundaries.
    
    While this changes the document structure, it can be effective when
    combined with other methods. The cross-sentence n-grams are broken.
    """
    
    def __init__(self):
        pass
    
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def shuffle_within_paragraphs(
        self,
        text: str,
        seed: Optional[int] = None,
    ) -> str:
        """
        Shuffle sentences within each paragraph.
        
        Args:
            text: Input text
            seed: Random seed
            
        Returns:
            Text with shuffled sentences
        """
        if seed is not None:
            random.seed(seed)
        
        paragraphs = text.split('\n\n')
        result_paragraphs = []
        
        for para in paragraphs:
            if not para.strip():
                continue
            sentences = self.split_sentences(para)
            if len(sentences) > 2:
                # Keep first and last, shuffle middle
                middle = sentences[1:-1]
                random.shuffle(middle)
                sentences = [sentences[0]] + middle + [sentences[-1]]
            result_paragraphs.append(' '.join(sentences))
        
        return '\n\n'.join(result_paragraphs)
    
    def reverse_clauses(self, text: str) -> str:
        """
        Reverse the order of clauses in complex sentences.
        
        Args:
            text: Input text
            
        Returns:
            Text with reversed clauses
        """
        sentences = self.split_sentences(text)
        result = []
        
        for sentence in sentences:
            # Check for complex sentences with commas
            if ',' in sentence or ';' in sentence:
                # Split on conjunctions and punctuation
                clauses = re.split(r'(,\s*(?:and|but|or|so|yet)\s*|;\s*)', sentence)
                if len(clauses) > 1:
                    # Reverse clause order while preserving punctuation
                    reversed_clauses = []
                    for i in range(len(clauses) - 1, -1, -1):
                        reversed_clauses.append(clauses[i])
                    sentence = ''.join(reversed_clauses).strip()
            result.append(sentence)
        
        return ' '.join(result)


# ============================================================================
# METHOD 4: Unicode/Homoglyph Attack
# ============================================================================

class HomoglyphAttack:
    """
    Replaces characters with visually similar Unicode characters.
    
    This breaks the exact byte sequence while keeping text readable.
    Very effective against hash-based watermarks since different bytes
    produce different hashes.
    """
    
    # Mapping of ASCII characters to similar-looking Unicode characters
    HOMOGLYPHS = {
        'a': ['а', 'ɑ', 'α'],  # Cyrillic, Latin alpha, Greek
        'e': ['е', 'ə', 'ε'],  # Cyrillic, schwa, Greek epsilon
        'i': ['і', 'ι', 'ı'],  # Cyrillic, Greek iota, dotless i
        'o': ['о', 'ο', 'ᴏ'],  # Cyrillic, Greek omicron, small caps
        'u': ['υ', 'ս'],       # Greek upsilon, Armenian
        'c': ['с', 'ϲ'],       # Cyrillic, Greek lunate sigma
        'p': ['р', 'ρ'],       # Cyrillic, Greek rho
        's': ['ѕ', 'ꜱ'],       # Cyrillic, small caps
        'x': ['х', 'χ'],       # Cyrillic, Greek chi
        'y': ['у', 'γ'],       # Cyrillic, Greek gamma
        'A': ['А', 'Α'],       # Cyrillic, Greek
        'B': ['В', 'Β'],       # Cyrillic, Greek
        'C': ['С', 'Ϲ'],       # Cyrillic, Greek
        'E': ['Е', 'Ε'],       # Cyrillic, Greek
        'H': ['Н', 'Η'],       # Cyrillic, Greek
        'I': ['І', 'Ι'],       # Cyrillic, Greek
        'K': ['К', 'Κ'],       # Cyrillic, Greek
        'M': ['М', 'Μ'],       # Cyrillic, Greek
        'N': ['Ν'],            # Greek
        'O': ['О', 'Ο'],       # Cyrillic, Greek
        'P': ['Р', 'Ρ'],       # Cyrillic, Greek
        'S': ['Ѕ'],            # Cyrillic
        'T': ['Т', 'Τ'],       # Cyrillic, Greek
        'X': ['Х', 'Χ'],       # Cyrillic, Greek
        'Y': ['Υ'],            # Greek
        'Z': ['Ζ'],            # Greek
    }
    
    def __init__(self):
        pass
    
    def apply_homoglyphs(
        self,
        text: str,
        replacement_rate: float = 0.1,
        seed: Optional[int] = None,
    ) -> str:
        """
        Replace some characters with homoglyphs.
        
        Args:
            text: Input text
            replacement_rate: Fraction of replaceable chars to replace
            seed: Random seed
            
        Returns:
            Text with homoglyph substitutions
        """
        if seed is not None:
            random.seed(seed)
        
        result = []
        for char in text:
            if char in self.HOMOGLYPHS and random.random() < replacement_rate:
                result.append(random.choice(self.HOMOGLYPHS[char]))
            else:
                result.append(char)
        
        return ''.join(result)


# ============================================================================
# METHOD 5: Whitespace Attack
# ============================================================================

class WhitespaceAttack:
    """
    Manipulates whitespace to break tokenization patterns.
    
    Since watermarking depends on token sequences, changing how text
    is tokenized can affect the watermark.
    """
    
    def __init__(self):
        pass
    
    def add_zero_width_chars(
        self,
        text: str,
        insertion_rate: float = 0.1,
        seed: Optional[int] = None,
    ) -> str:
        """
        Insert zero-width characters between tokens.
        
        Args:
            text: Input text
            insertion_rate: Probability of insertion between words
            seed: Random seed
            
        Returns:
            Text with zero-width characters
        """
        if seed is not None:
            random.seed(seed)
        
        # Zero-width characters
        zwchars = [
            '\u200b',  # Zero-width space
            '\u200c',  # Zero-width non-joiner
            '\u200d',  # Zero-width joiner
            '\ufeff',  # Zero-width no-break space
        ]
        
        words = text.split(' ')
        result = []
        
        for word in words:
            result.append(word)
            if random.random() < insertion_rate:
                result.append(random.choice(zwchars))
        
        return ' '.join(result)
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize all whitespace to standard spaces.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        # Replace various whitespace chars with standard space
        import unicodedata
        result = []
        for char in text:
            if unicodedata.category(char).startswith('Z'):
                result.append(' ')
            else:
                result.append(char)
        
        # Collapse multiple spaces
        return re.sub(r' +', ' ', ''.join(result))


# ============================================================================
# Combined Attack
# ============================================================================

class CombinedAttack:
    """
    Combines multiple attack methods for maximum effectiveness.
    """
    
    def __init__(
        self,
        use_paraphrasing: bool = True,
        model_name: str = "google/gemma-2b-it",
        device: torch.device = DEVICE,
    ):
        """
        Initialize combined attack.
        
        Args:
            use_paraphrasing: Whether to use ML-based paraphrasing
            model_name: Model for paraphrasing
            device: Torch device
        """
        self.perturbation = TokenPerturbationAttack()
        self.shuffling = SentenceShufflingAttack()
        self.homoglyph = HomoglyphAttack()
        self.whitespace = WhitespaceAttack()
        
        if use_paraphrasing:
            try:
                self.paraphrasing = ParaphrasingAttack(model_name, device)
            except Exception as e:
                print(f"Warning: Could not load paraphrasing model: {e}")
                self.paraphrasing = None
        else:
            self.paraphrasing = None
    
    def attack(
        self,
        text: str,
        use_paraphrasing: bool = True,
        use_perturbation: bool = True,
        use_homoglyphs: bool = False,  # May break text display
        perturbation_rate: float = 0.2,
        seed: Optional[int] = None,
    ) -> Tuple[str, dict]:
        """
        Apply combined attack to remove watermark.
        
        Args:
            text: Watermarked text
            use_paraphrasing: Use ML paraphrasing (most effective)
            use_perturbation: Use token perturbation
            use_homoglyphs: Use homoglyph substitution (may affect display)
            perturbation_rate: Rate for token perturbation
            seed: Random seed
            
        Returns:
            Tuple of (attacked_text, attack_info_dict)
        """
        result = text
        attack_info = {'original_length': len(text)}
        
        # Step 1: Paraphrasing (most effective)
        if use_paraphrasing and self.paraphrasing is not None:
            print("Applying paraphrasing attack...")
            result = self.paraphrasing.paraphrase(result)
            attack_info['paraphrased'] = True
        else:
            attack_info['paraphrased'] = False
        
        # Step 2: Token perturbation
        if use_perturbation:
            print("Applying token perturbation...")
            result = self.perturbation.perturb(
                result,
                substitution_rate=perturbation_rate,
                insertion_rate=perturbation_rate / 2,
                seed=seed,
            )
            attack_info['perturbed'] = True
        else:
            attack_info['perturbed'] = False
        
        # Step 3: Homoglyph (optional, can break text)
        if use_homoglyphs:
            print("Applying homoglyph attack...")
            result = self.homoglyph.apply_homoglyphs(
                result,
                replacement_rate=0.05,
                seed=seed,
            )
            attack_info['homoglyphs'] = True
        else:
            attack_info['homoglyphs'] = False
        
        attack_info['final_length'] = len(result)
        
        return result, attack_info


# ============================================================================
# Watermark Detection (to verify attack effectiveness)
# ============================================================================

class WatermarkDetector:
    """
    Simple watermark detector to verify attack effectiveness.
    Uses the same detection logic as SynthID.
    """
    
    def __init__(
        self,
        tokenizer_name: str = "gpt2",
        ngram_len: int = 5,
        keys: List[int] = None,
    ):
        """
        Initialize detector.
        
        Args:
            tokenizer_name: Tokenizer to use
            ngram_len: N-gram length for detection
            keys: Watermarking keys (default: SynthID default keys)
        """
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if keys is None:
            # Default SynthID keys
            keys = [
                654, 400, 836, 123, 340, 443, 597, 160, 57, 29,
                590, 639, 13, 715, 468, 990, 966, 226, 324, 585,
                118, 504, 421, 521, 129, 669, 732, 225, 90, 960,
            ]
        
        self.logits_processor = logits_processing.SynthIDLogitsProcessor(
            ngram_len=ngram_len,
            keys=keys,
            context_history_size=1024,
            temperature=1.0,
            top_k=40,
            device=DEVICE,
        )
    
    def compute_score(self, text: str) -> Tuple[float, dict]:
        """
        Compute watermark detection score.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (mean_g_value, detection_info)
        """
        # Tokenize
        tokens = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=1024,
        ).input_ids.to(DEVICE)
        
        # Compute g-values
        g_values = self.logits_processor.compute_g_values(tokens)
        
        # Compute mean score
        mean_g = g_values.float().mean().item()
        
        # For watermarked text, mean should be > 0.5
        # For non-watermarked text, mean should be ~0.5
        info = {
            'mean_g_value': mean_g,
            'num_tokens': tokens.shape[1],
            'g_values_shape': list(g_values.shape),
            'likely_watermarked': mean_g > 0.55,  # Threshold
        }
        
        return mean_g, info


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Remove SynthID watermarks from AI-generated text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Paraphrase attack with Gemma (most effective)
  python reverse_synthid.py --input text.txt --output clean.txt --method paraphrase
  
  # Use a different model (e.g., GPT-2 or larger Gemma)
  python reverse_synthid.py --input text.txt --output clean.txt --method paraphrase --model google/gemma-7b-it
  
  # Token perturbation (no ML model needed)
  python reverse_synthid.py --input text.txt --output clean.txt --method perturb
  
  # Combined attack
  python reverse_synthid.py --input text.txt --output clean.txt --method combined
  
  # Just detect watermark
  python reverse_synthid.py --input text.txt --detect-only
"""
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input file with watermarked text')
    parser.add_argument('--output', '-o', type=str,
                        help='Output file for cleaned text')
    parser.add_argument('--method', '-m', type=str, default='combined',
                        choices=['paraphrase', 'perturb', 'shuffle', 
                                'homoglyph', 'combined'],
                        help='Attack method to use')
    parser.add_argument('--model', type=str, default='google/gemma-2b-it',
                        help='Model for paraphrasing (default: google/gemma-2b-it)')
    parser.add_argument('--rate', type=float, default=0.2,
                        help='Perturbation rate (0.0-1.0)')
    parser.add_argument('--detect-only', action='store_true',
                        help='Only detect watermark, do not remove')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--no-viz', action='store_true',
                        help='Disable visualization generation')
    
    args = parser.parse_args()
    
    # Read input
    with open(args.input, 'r') as f:
        text = f.read().strip()
    
    print(f"Input text length: {len(text)} characters")
    
    # Detect watermark
    if args.detect_only or args.method in ['combined', 'paraphrase']:
        try:
            detector = WatermarkDetector()
            score, info = detector.compute_score(text)
            print(f"\n=== Watermark Detection ===")
            print(f"Mean G-value: {score:.4f}")
            print(f"Number of tokens: {info['num_tokens']}")
            print(f"Likely watermarked: {info['likely_watermarked']}")
            
            if args.detect_only:
                return
        except Exception as e:
            print(f"Warning: Could not run detection: {e}")
    
    # Apply attack
    print(f"\n=== Applying {args.method} attack ===")
    
    if args.method == 'paraphrase':
        attack = ParaphrasingAttack(args.model, DEVICE)
        result = attack.chunk_and_paraphrase(text)
    
    elif args.method == 'perturb':
        attack = TokenPerturbationAttack()
        result = attack.perturb(text, substitution_rate=args.rate, seed=args.seed)
    
    elif args.method == 'shuffle':
        attack = SentenceShufflingAttack()
        result = attack.shuffle_within_paragraphs(text, seed=args.seed)
    
    elif args.method == 'homoglyph':
        attack = HomoglyphAttack()
        result = attack.apply_homoglyphs(text, replacement_rate=args.rate, seed=args.seed)
    
    elif args.method == 'combined':
        attack = CombinedAttack(use_paraphrasing=True, model_name=args.model, device=DEVICE)
        result, info = attack.attack(
            text,
            use_paraphrasing=True,
            use_perturbation=True,
            perturbation_rate=args.rate,
            seed=args.seed,
        )
        print(f"Attack info: {info}")
    
    print(f"\nOutput text length: {len(result)} characters")
    
    # Verify attack effectiveness
    try:
        detector = WatermarkDetector()
        new_score, new_info = detector.compute_score(result)
        print(f"\n=== Post-Attack Detection ===")
        print(f"Mean G-value: {new_score:.4f}")
        print(f"Likely watermarked: {new_info['likely_watermarked']}")
    except Exception as e:
        print(f"Warning: Could not verify attack: {e}")
    
    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(result)
        print(f"\nCleaned text written to: {args.output}")
        
        # Generate visualization comparing watermarked and clean text
        if not args.no_viz:
            print(f"\n=== Generating Visualization ===")
            try:
                from visualizations.visualize_watermark import EnhancedWatermarkVisualizer
                
                # Create visualizations directory if needed
                viz_dir = os.path.join(os.path.dirname(args.output) or '.', 'visualizations')
                os.makedirs(viz_dir, exist_ok=True)
                
                # Initialize visualizer with the model
                visualizer = EnhancedWatermarkVisualizer(tokenizer_name=args.model)
                
                # Generate comparison visualization
                output_base = os.path.splitext(os.path.basename(args.output))[0]
                comparison_path = os.path.join(viz_dir, f"{output_base}_comparison.png")
                
                visualizer.create_comparison(
                    watermarked_text=text,
                    clean_text=result,
                    output_dir=comparison_path
                )
                print(f"Comparison visualization saved to: {comparison_path}")
                
                # Also save individual visualizations
                watermarked_path = os.path.join(viz_dir, f"{output_base}_watermarked.png")
                clean_path = os.path.join(viz_dir, f"{output_base}_clean.png")
                
                visualizer.create_visualization(
                    text, 
                    watermarked_path, 
                    title="Watermarked Text"
                )
                visualizer.create_visualization(
                    result, 
                    clean_path, 
                    title="Cleaned Text"
                )
                print(f"Individual visualizations saved to: {watermarked_path}, {clean_path}")
                
            except Exception as e:
                print(f"Warning: Could not generate visualization: {e}")
    else:
        print(f"\n=== Cleaned Text ===\n{result}")


if __name__ == '__main__':
    main()
