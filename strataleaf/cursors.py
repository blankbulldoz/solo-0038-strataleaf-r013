import base64
import hashlib
import hmac
from .codec import dumps, loads
from .errors import InvalidCursor

# Legacy builds accidentally used the 64 ASCII hex characters of the secret
# as the HMAC key instead of the 32 decoded bytes, so every database shared
# the same effective signing key. Cursors minted with that key stay verifiable
# via the legacy key until their (unpinned, short-lived) traversals finish;
# newly minted cursors always use the per-database derived key.
_CURRENT_KEY_DOMAIN = b"strataleaf-cursor-key-v2:"


def _secret_keys(store):
    """Return (current per-database key, legacy shared key) for this store."""
    secret = store.get("cursor_secret")
    try:
        decoded = bytes.fromhex(secret)
        if len(decoded) != 32:
            raise ValueError("secret length")
    except ValueError:
        decoded = hashlib.sha256(secret.encode()).digest()
    current = hmac.new(_CURRENT_KEY_DOMAIN + decoded,
                       store.get("database_id").encode(),
                       hashlib.sha256).digest()
    # Exact key the old encode_cursor used: the hex text itself as bytes.
    legacy = secret.encode()
    return current, legacy


def encode_cursor(store, payload):
    raw = dumps(payload).encode()
    key, _ = _secret_keys(store)
    signature = hmac.new(key, raw, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(signature + raw).decode().rstrip("=")


def decode_cursor(store, token):
    try:
        if not isinstance(token, str) or len(token) > 8192:
            raise ValueError("invalid token length")
        data = base64.b64decode(token + "=" * (-len(token) % 4), altchars=b"-_", validate=True)
        signature, raw = data[:32], data[32:]
        if not any(hmac.compare_digest(
                hmac.new(key, raw, hashlib.sha256).digest(), signature)
                for key in _secret_keys(store)):
            raise ValueError("signature mismatch")
        payload = loads(raw)
        if not isinstance(payload, dict) or payload.get("database") != store.get("database_id"):
            raise ValueError("wrong database")
        return payload
    except (ValueError, TypeError, KeyError, UnicodeDecodeError) as exc:
        raise InvalidCursor("invalid or foreign continuation cursor") from exc
