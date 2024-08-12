# ofen

## Work In Progress

This project is currently under active development. Features and documentation may be incomplete or subject to change.

## About

Ofen is a toolkit aimed at making transformer models production-ready. API included (in the future).

## Usage

```python
from ofen.models import TextEncoder, CrossEncoder

encoder = TextEncoder.from_pretrained("mixedbread-ai/mxbai-embed-large-v1")
cross_encoder = CrossEncoder.from_pretrained("mixedbread-ai/mxbai-rerank-large-v1")

from ofen.models.onnx import ORTTextEncoder, ORTCrossEncoder
encoder = ORTTextEncoder.from_pretrained("mixedbread-ai/mxbai-embed-large-v1")
cross_encoder = ORTCrossEncoder.from_pretrained("mixedbread-ai/mxbai-rerank-large-v1")


encoder.encode("Hello world")
cross_encoder.rerank("Python", ["print('Hello world')"])
```

## Contributing

As this project is in its early stages, contributions are welcome. Please check the issues page for current tasks or suggest improvements.

## License

[To be determined]
