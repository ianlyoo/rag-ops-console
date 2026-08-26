# Owner actions — rag-ops-console

Manual steps that cannot be automated via API.

## 1. GitHub repository social preview (manual upload)

Pages `og:image` (`https://ianlyoo.github.io/rag-ops-console/assets/social-preview.png`) is automatic after push. Repository card requires manual upload:

- Path: Settings → General → Social preview → Edit → Upload an image
- File: `docs/assets/social-preview.png` (1280×640, <1 MiB, #0f172a)

Verification:

```bash
gh api graphql -f query='query{repository(owner:"ianlyoo",name:"rag-ops-console"){openGraphImageUrl}}'
# before: https://opengraph.githubassets.com/... (default)
# after:  user-images / repository-images URL
```

Live Pages preview:

```bash
curl -fsSL https://ianlyoo.github.io/rag-ops-console/ | grep -o 'og:image[^>]*content="[^"]*"'
curl -fsSL https://ianlyoo.github.io/rag-ops-console/assets/social-preview.png -o /tmp/p.png && ls -lh /tmp/p.png
```

## 2. Optional profile pin

- Profile → Customize your pins → select `ianlyoo/rag-ops-console`

## 3. Do not attempt

- No API upload for repository social preview.
- No force push.
