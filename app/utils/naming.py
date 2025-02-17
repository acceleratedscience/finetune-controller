import uuid


def generate_short_uuid():
    # Generate a UUID and truncate it to 8 characters
    return str(uuid.uuid4())[:8].lower()
