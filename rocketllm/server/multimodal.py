"""Images off the wire, and into the tensors a vision-language model reads.

OpenAI's multimodal chat payload is a message whose ``content`` is a list of parts rather than a
string::

    [{"type": "text", "text": "what is in this picture?"},
     {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KG..."}}]

Two things happen to that here. The parts are rewritten into the ``{"type": "image"}`` form that
chat templates actually branch on, so the template puts the model's own image placeholders in the
right place; and the payloads are decoded into images for the processor, which is the only thing
that knows how many tokens one image expands to for a given checkpoint.

**What is accepted, and what is not.** ``data:`` URLs and paths on this machine are decoded. An
``http://`` or ``https://`` URL is refused, and that is deliberate rather than unfinished: fetching
a URL a request supplied would make this server issue arbitrary outbound requests on a client's
behalf, from wherever it is deployed, against whatever it can reach -- including addresses a client
cannot reach itself. A client that wants a remote image can fetch it and inline it, which costs it
one request and costs this server none of that.

Pillow is an optional dependency. Without it every text path is unaffected and an image request is
refused with the command that fixes it, which is the same shape every other optional package takes
here.
"""
import base64
import binascii
import io
from pathlib import Path
from urllib.parse import unquote, urlparse

from .protocol import RequestError

#: What to type when the decoder is missing. The extra rather than a bare `pip install pillow`, so
#: one instruction covers whatever else the image path grows to need.
PILLOW_HINT = "pip install 'rocketllm[vision]'"

#: Part types that carry a picture. `image_url` is OpenAI's; `image` is what most chat templates
#: and several clients use directly. Both are accepted, because refusing one of them would be a
#: compatibility break with no upside.
IMAGE_PART_TYPES = ("image_url", "image", "input_image")

#: The rewritten form. Chat templates overwhelmingly branch on `content['type'] == 'image'`, and
#: the pixels travel to the processor separately, so the part that reaches the template carries no
#: payload at all.
TEMPLATE_IMAGE_PART = {"type": "image"}


def _pillow():
    """The imaging library, or a RequestError naming what to install."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RequestError(
            f"this request carries an image, and decoding one needs Pillow, which is an optional "
            f"dependency ({exc}). Install it with:  {PILLOW_HINT}",
            param="messages", status_code=400, code="image_decoding_unavailable") from exc
    return Image


def part_image_source(part):
    """The URL or path a content part points at, or None when the part is not an image.

    Every shape clients actually send: OpenAI's nested ``{"image_url": {"url": ...}}``, the flat
    ``{"image_url": "..."}`` several libraries emit, and the ``{"type": "image", "image": ...}``
    form chat templates use.
    """
    if not isinstance(part, dict):
        return None
    kind = part.get("type")
    if kind is not None and kind not in IMAGE_PART_TYPES:
        return None
    for key in ("image_url", "image", "url", "source"):
        value = part.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict):
            nested = value.get("url") or value.get("path") or value.get("data")
            if isinstance(nested, str) and nested:
                return nested
    return None


def _decode_data_url(source):
    """The bytes behind a ``data:`` URL.

    The payload is required to be base64: a ``data:`` URL may also be percent-encoded text, and an
    image never is, so accepting one would only turn a malformed request into a confusing decode
    failure further in.
    """
    header, _, payload = source.partition(",")
    if not payload:
        raise RequestError("an image data: URL carried no payload after the comma",
                           param="messages")
    if ";base64" not in header:
        raise RequestError(
            "an image data: URL must be base64-encoded (data:image/png;base64,...). A "
            "percent-encoded data: URL is text, and this is where pictures go.",
            param="messages")
    try:
        # validate=False on purpose: real clients wrap base64 at 76 columns, and the newlines that
        # produces are not an error in anything else that reads these.
        return base64.b64decode(payload)
    except (binascii.Error, ValueError) as exc:
        raise RequestError(f"an image data: URL could not be base64-decoded ({exc})",
                           param="messages") from exc


def _read_local(source):
    """The bytes of a local file, given a path or a file:// URL."""
    if source.startswith("file://"):
        parsed = urlparse(source)
        # A Windows path arrives as file:///C:/x, which leaves a leading slash in front of the
        # drive letter that Path cannot use.
        path = unquote(parsed.path)
        if parsed.netloc:
            path = f"//{parsed.netloc}{path}"
        elif len(path) > 2 and path[0] == "/" and path[2] == ":":
            path = path[1:]
    else:
        path = source
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        raise RequestError(f"could not read the image at {path!r} ({exc})",
                           param="messages") from exc


def decode_image(source):
    """One image part's payload as a PIL image, in RGB.

    RGB rather than whatever the file happened to be: a palette or greyscale image reaching an
    image processor that expects three channels fails inside the processor with a shape error that
    says nothing about the picture it came from.
    """
    Image = _pillow()
    if source.startswith("data:"):
        raw = _decode_data_url(source)
    elif source.startswith(("http://", "https://")):
        raise RequestError(
            "this server does not fetch image URLs. Sending one would make it issue outbound "
            "requests on a client's behalf, against whatever the machine it runs on can reach. "
            "Inline the image instead, as data:image/<type>;base64,<payload>, or pass a path to a "
            "file on the server.",
            param="messages", code="image_url_not_fetched")
    else:
        raw = _read_local(source)

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as exc:  # noqa: BLE001 - every decoder failure is the same client mistake
        raise RequestError(f"an image in this request could not be decoded ({exc}). Supported "
                           f"formats are whatever Pillow reads on this machine.",
                           param="messages") from exc
    return image.convert("RGB") if image.mode != "RGB" else image


def message_images(message):
    """Every image source in one request message, in the order it appeared."""
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return []
    found = []
    for part in content:
        source = part_image_source(part)
        if source is not None:
            found.append(source)
    return found


def has_images(messages):
    return any(message_images(message) for message in messages)


def collect_images(messages):
    """Decode every image in a conversation, in the order the model will meet them.

    Order is the whole contract: the processor pairs the Nth decoded image with the Nth placeholder
    the template rendered, so a list built in any other order silently describes the wrong picture.
    """
    return [decode_image(source)
            for message in messages
            for source in message_images(message)]


def template_content(message):
    """A message's content in the shape a chat template expects, images included.

    Text parts keep their text and image parts collapse to a bare ``{"type": "image"}``: the
    template's job is to put the model's placeholder tokens in the right place, and it needs to know
    that an image is there and nothing else about it. A message with no images at all comes back as
    a plain string, which is what every template handles and what this server has always sent.
    """
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return message.text()

    parts = []
    for part in content:
        if part_image_source(part) is not None:
            parts.append(dict(TEMPLATE_IMAGE_PART))
        elif isinstance(part, dict) and part.get("type") in (None, "text"):
            text = part.get("text")
            if isinstance(text, str) and text:
                parts.append({"type": "text", "text": text})
    if not any(part["type"] == "image" for part in parts):
        return message.text()
    return parts
