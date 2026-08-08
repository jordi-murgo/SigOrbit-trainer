# Threat model and safe execution

The trainer treats configurations, manifests, images, and recovery checkpoints
as untrusted local inputs. It rejects unknown configuration keys, escaping or
symlinked image paths, unsupported image formats, excessive dimensions, digest
mismatches, and incompatible checkpoint metadata. Authorized image bytes are
re-read into a bounded snapshot and rehashed on every training access, preventing
post-preflight replacement from silently changing the corpus. Recovery tensor
bytes are likewise hashed and deserialized from the same immutable snapshot. Native recovery state uses
safetensors plus strict JSON; no pickle is loaded for resume.

Run training as a non-root user in a network-disabled container with a read-only
root filesystem, the dataset mounted read-only, a dedicated output volume, and
memory/PID/disk/time limits. Image decoders contain native code; stronger threat
models should decode in a sandboxed preprocessing job.

The current SigOrbit 0.1 deployable format is a `.pt` weights-only dictionary.
It is created locally at export time, never accepted as trainer resume state,
and SigOrbit verifies a caller-supplied SHA-256 before loading with
`torch.load(weights_only=True)`.
