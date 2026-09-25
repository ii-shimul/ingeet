"""Sign Language to Bengali Translation Dictionary and Vocabulary.

Maps hand gestures, letters, and conversational signs to Bengali script translations.
"""

SIGN_DICTIONARY = {
    # Conversational Gestures & Words
    "HELLO": {
        "bn": "হ্যালো",
        "en": "Hello",
        "desc": "অভিবাদন (Greeting)",
        "color": (0, 255, 128),
    },
    "THANK_YOU": {
        "bn": "ধন্যবাদ",
        "en": "Thank You",
        "desc": "কৃতজ্ঞতা প্রকাশ (Gratitude)",
        "color": (0, 255, 255),
    },
    "YES": {
        "bn": "হ্যাঁ",
        "en": "Yes",
        "desc": "সম্মতি (Affirmative)",
        "color": (0, 255, 128),
    },
    "NO": {
        "bn": "না",
        "en": "No",
        "desc": "অসম্মতি (Negative)",
        "color": (0, 100, 255),
    },
    "I_LOVE_YOU": {
        "bn": "আমি তোমাকে ভালোবাসি",
        "en": "I Love You",
        "desc": "ভালোবাসার ইশারা (Affection)",
        "color": (180, 105, 255),
    },
    "PEACE": {
        "bn": "শান্তি ও বিজয়",
        "en": "Peace / Victory",
        "desc": "শান্তির প্রতীক (Victory/Peace)",
        "color": (255, 215, 0),
    },
    "GOOD": {
        "bn": "ভালো, চমৎকার",
        "en": "Good / Awesome",
        "desc": "প্রশংসা (Appreciation)",
        "color": (0, 255, 128),
    },
    "OKAY": {
        "bn": "ঠিক আছে",
        "en": "Okay",
        "desc": "ঠিক আছে (Agreement)",
        "color": (0, 255, 200),
    },
    "PLEASE": {
        "bn": "দয়া করে",
        "en": "Please",
        "desc": "অনুরোধ (Request)",
        "color": (255, 191, 0),
    },
    "SORRY": {
        "bn": "দুঃখিত",
        "en": "Sorry",
        "desc": "ক্ষমা প্রার্থনা (Apology)",
        "color": (200, 200, 255),
    },
    "HELP": {
        "bn": "সাহায্য চাই",
        "en": "Help",
        "desc": "সাহায্য (Emergency/Assistance)",
        "color": (0, 69, 255),
    },
    "STOP": {
        "bn": "থামুন",
        "en": "Stop",
        "desc": "থামুন (Halt)",
        "color": (0, 0, 255),
    },
    "WATER": {
        "bn": "পানি",
        "en": "Water",
        "desc": "পানির ইশারা (Thirst)",
        "color": (255, 200, 0),
    },
    "CALL_ME": {
        "bn": "যোগাযোগ করুন",
        "en": "Call Me",
        "desc": "যোগাযোগ (Call gesture)",
        "color": (255, 165, 0),
    },
    "ROCK": {
        "bn": "দুর্দান্ত",
        "en": "Rock On",
        "desc": "উৎসাহ (Rock on)",
        "color": (255, 0, 255),
    },
}

# Alphabet A-Z clean Bengali representations
ALPHABET_MAP = {
    "A": "এ",
    "B": "বি",
    "C": "সি",
    "D": "ডি",
    "E": "ই",
    "F": "এফ",
    "G": "জি",
    "H": "এইচ",
    "I": "আই",
    "J": "জে",
    "K": "কে",
    "L": "এল",
    "M": "এম",
    "N": "এন",
    "O": "ও",
    "P": "পি",
    "Q": "কিউ",
    "R": "আর",
    "S": "এস",
    "T": "টি",
    "U": "ইউ",
    "V": "ভি",
    "W": "ডাব্লিউ",
    "X": "এক্স",
    "Y": "ওয়াই",
    "Z": "জেড",
}

for letter, bn_text in ALPHABET_MAP.items():
    SIGN_DICTIONARY[letter] = {
        "bn": bn_text,
        "en": f"Letter '{letter}'",
        "desc": f"বর্ণমালা {letter}",
        "color": (255, 255, 255),
    }


def get_translation(sign_name: str) -> dict:
    """Retrieve Bengali translation metadata for a sign or letter."""
    cleaned = sign_name.strip().upper().replace(" ", "_")
    if cleaned in SIGN_DICTIONARY:
        return SIGN_DICTIONARY[cleaned]
    return {
        "bn": sign_name,
        "en": sign_name,
        "desc": "অজানা ইশারা",
        "color": (200, 200, 200),
    }


# Aliases
get_bengali_translation = get_translation
ASL_TO_BENGALI = SIGN_DICTIONARY
