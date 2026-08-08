# Model release policy

Training completion never authorizes weight publication. By default, exported
artifacts are labelled `private-research-only`.

A public or commercial model release requires, at minimum:

- written dataset-by-dataset permission covering training and weight release;
- biometric/privacy and organizational ownership review;
- approved model card, limitations, evaluation protocol, and intended use;
- held-out and rotation robustness gates;
- SigOrbit runtime reload and embedding-parity verification;
- artifact SHA-256, provenance attestation, and an explicit approver reference.

The Python package release workflow rejects weight and data extensions.
