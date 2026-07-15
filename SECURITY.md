# Security policy

Please report vulnerabilities privately through GitHub Security Advisories rather than a public
issue. Include affected versions, reproduction steps, impact, and any suggested remediation.

The supported line is the latest release. The parser is designed for small local model files and
enforces a 4 MiB input limit; it is not a hardened service endpoint. Applications accepting
remote models should add process isolation, resource quotas, and authentication at their own
boundary.

Never include API keys, access tokens, credentials, or private model data in reports or fixtures.
