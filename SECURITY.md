# Security Policy

## Project Status & Philosophy

`lerobot` has so far been primarily a research and prototyping tool, which is why deployment security hasn’t been a strong focus until now. As `lerobot` continues to be adopted and deployed in production, we are paying much closer attention to these kinds of issues.

Fortunately, being an open-source project, the community can also help by reporting and fixing vulnerabilities. We appreciate your efforts to responsibly disclose your findings and will make every effort to acknowledge your contributions.

## Reporting a Vulnerability

To report a security issue in this fork, please use the GitHub Security Advisory ["Report a Vulnerability"](https://github.com/capstone-indory/indory_lerobot/security/advisories/new) tab.

The maintainers will send a response indicating the next steps in handling your report. After the initial reply to your report, we will keep you informed of the progress towards a fix and full announcement, and may ask for additional information or guidance.

#### Upstream LeRobot and Hugging Face Infrastructure

For vulnerabilities in upstream LeRobot or Hugging Face Hub infrastructure,
please follow the upstream Hugging Face security reporting process.

#### Open Source Disclosures

If reporting a vulnerability specific to the open-source codebase (and not the underlying Hub infrastructure), you may also use [Huntr](https://huntr.com), a vulnerability disclosure program for open source software.

## Supported Versions

Currently, we treat `lerobot` as a rolling release. We prioritize security updates for the latest available version (`main` branch).

| Version  | Supported |
| -------- | --------- |
| Latest   | ✅        |
| < Latest | ❌        |

## Secure Usage Guidelines

`lerobot` is tightly coupled to the Hugging Face Hub for sharing data and pretrained policies. When downloading artifacts uploaded by others, you expose yourself to risks. Please read below for recommendations to keep your runtime and robot environment safe.

### Remote Artefacts (Weights & Policies)

Models and policies uploaded to the Hugging Face Hub come in different formats. We heavily recommend uploading and downloading models in the [`safetensors`](https://github.com/huggingface/safetensors) format.

`safetensors` was developed specifically to prevent arbitrary code execution on your system, which is critical when running software on physical hardware/robots.

To avoid loading models from unsafe formats (e.g., `pickle`), you should ensure you are prioritizing `safetensors` files.

### Remote Code

Some models or environments on the Hub may require `trust_remote_code=True` to run custom architecture code.

Please **always** verify the content of the modeling files when using this argument. We recommend setting a specific `revision` (commit hash) when loading remote code to ensure you protect yourself from unverified updates to the repository.
