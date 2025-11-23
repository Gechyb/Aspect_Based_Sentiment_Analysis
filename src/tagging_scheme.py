TAGS = ["PAD", "O", "B-POS", "I-POS", "B-NEG", "I-NEG", "B-NEU", "I-NEU"]

TAG2ID = {t: i for i, t in enumerate(TAGS)}
ID2TAG = {i: t for t, i in TAG2ID.items()}

VALID_TRANSITIONS = {
    "O": {"B-POS", "B-NEG", "B-NEU", "O"},
    "B-POS": {"I-POS", "O", "B-NEG", "B-NEU", "B-POS"},
    "I-POS": {"I-POS", "O", "B-POS", "B-NEG", "B-NEU"},
    "B-NEG": {"I-NEG", "O", "B-POS", "B-NEU", "B-NEG"},
    "I-NEG": {"I-NEG", "O", "B-POS", "B-NEG", "B-NEU"},
    "B-NEU": {"I-NEU", "O", "B-POS", "B-NEG", "B-NEU"},
    "I-NEU": {"I-NEU", "O", "B-POS", "B-NEG", "B-NEU"},
}
