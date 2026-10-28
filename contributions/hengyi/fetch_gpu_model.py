"""Fetch pinned public Qwen weights to project-local cache, verify LFS SHA256."""
import hashlib
import json
from pathlib import Path
import urllib.request

MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'
WEIGHTS_SHA256 = 'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'
MODEL_DIR = Path('models/Qwen2.5-0.5B-Instruct')


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in ['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
                 'merges.txt', 'vocab.json', 'LICENSE', 'README.md', 'model.safetensors']:
        destination = MODEL_DIR/name
        if not destination.exists():
            print(f'Downloading {name}', flush=True)
            url = f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}'
            temporary = destination.with_suffix(destination.suffix+'.part')
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as output:
                while block := response.read(1024*1024):
                    output.write(block)
            temporary.replace(destination)
        digest = hashlib.file_digest(destination.open('rb'), 'sha256').hexdigest()
        if name == 'model.safetensors' and digest != WEIGHTS_SHA256:
            raise ValueError('Model weight hash mismatch')
        hashes[name] = digest
    payload = dict(model_id=MODEL_ID, revision=REVISION, files_sha256=hashes,
                   source=f'https://huggingface.co/{MODEL_ID}', license='Apache-2.0')
    (MODEL_DIR/'source.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    Path('results/gpu').mkdir(parents=True, exist_ok=True)
    Path('results/gpu/model_source.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print('Pinned model download verified', flush=True)


if __name__ == '__main__':
    main()
