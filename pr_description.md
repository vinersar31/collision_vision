🔒 Fix Server-Side Request Forgery (SSRF) / Local File Read in download_examples.py

🎯 **What:** The `download` function in `scripts/download_examples.py` previously accepted any URL scheme and passed it directly to `urllib.request.urlopen`. This PR adds URL scheme validation to ensure only `http` and `https` schemes are permitted.

⚠️ **Risk:** Without scheme validation, an attacker or malformed input could exploit the script to access internal resources or read local files using schemes like `file://` or `ftp://`. This constitutes a Server-Side Request Forgery (SSRF) and Local File Read vulnerability, potentially exposing sensitive data.

🛡️ **Solution:** The fix utilizes `urllib.parse.urlparse` to extract the scheme from the provided URL. If the scheme is not `http` or `https`, the function rejects the URL, prints a warning message, and returns `False`, thereby neutralizing the vulnerability. A new test suite `tests/test_download_examples.py` has also been added to verify this behavior.
