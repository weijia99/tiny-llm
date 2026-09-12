import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper

from .qwen3_week1 import Qwen3ModelWeek1
from .qwen3_week2 import Qwen3ModelWeek2
from typing import Callable


def _release_kv_cache(kv_cache):
    if kv_cache is None:
        return
    for layer in kv_cache:
        layer.release()


def simple_generate(
    model: Qwen3ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
    max_tokens: int = 256,
) -> None:
    def _step(model, y):
        
        logits = model(y[None])
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        if sampler is not None:
            y = sampler(logprobs)
        else:
            y = mx.argmax(logprobs, axis=-1)
        return y
    
    tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()

    
    while True:
            token = _step(model, tokens)
            mx.eval(token)
            tokens = mx.concat([tokens, token])
            if token.item() == tokenizer.eos_token_id:
                break
            detokenizer.add_token(token.item())
            print(detokenizer.last_segment, end="", flush=True)
    

def simple_generate_with_kv_cache(
    model: Qwen3ModelWeek2,
    tokenizer: TokenizerWrapper,
    prompt: str,
    max_tokens: int = 256,
) -> str:
    def _step(model, y, offset, kv_cache):
        # 通过产生的logists接着传进去
        logits = model(y[None], offset, kv_cache)
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        
        y = mx.argmax(logprobs, axis=-1)
        return y

    tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    kv_cache = model.create_kv_cache()
    offset = len(tokens)
    print(offset)
    token = _step(model, tokens, offset=0, kv_cache=kv_cache)
    mx.eval(token)
    tokens = mx.concat([tokens, token])
    if token.item() == tokenizer.eos_token_id:
            return
    detokenizer.add_token(token.item())
    print(detokenizer.last_segment, end="", flush=True)
    # prefill代码，先缓存kv cache，之后再进行decode


    while True:
        token = _step(model, token, offset=offset, kv_cache=kv_cache)
        mx.eval(token)
        offset += 1
        tokens = mx.concat([tokens, token])
        if token.item() == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token.item())
        print(detokenizer.last_segment, end="", flush=True)

        # 后续的来进行offset
    _release_kv_cache(kv_cache)
    


def speculative_generate(
    draft_model: Qwen3ModelWeek2,
    model: Qwen3ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
    proposal_length: int = 4,
    max_tokens: int = 256,
) -> str:
    pass
