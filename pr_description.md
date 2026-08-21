🔒 Fix Denial of Service vulnerability in image upload

🎯 **What:** Fixed a vulnerability where users could upload arbitrarily large images, leading to excessive memory consumption during conversion to NumPy arrays.
⚠️ **Risk:** A Denial of Service (DoS) attack could easily be triggered by an attacker uploading a very large image file (e.g., a "Decompression Bomb" or exceptionally high-resolution image). This would cause the Streamlit application to exhaust system memory and crash, taking down the service for all users.
🛡️ **Solution:** Enforced a maximum image pixel limit (`Image.MAX_IMAGE_PIXELS = 25000000`) before image processing. Wrapped the image conversion logic in a `try...except Image.DecompressionBombError` block to gracefully catch the exception, stop processing, and display an error message to the user instead of crashing the server.
