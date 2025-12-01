

CARD_VALUES = {"ace": 1, "two": 2, "three": 3, "four": 4}

def parse_die_value(cls_name):
    if not isinstance(cls_name, str):
        cls_name = str(cls_name)
    parts = cls_name.split("_")
    if len(parts) != 2:
        return None, None
    return parts[0], int(parts[1])


class Scoring:
    def __init__(self):
        self.scores = {"red": 0, "blue": 0, "yellow": 0}

        # die_id → state
        self.state = {}

        self.STABLE_ON = 5      # frames needed before awarding points
        self.STABLE_OFF = 5     # frames needed before subtracting

    def update_scores(self, tracked_dice, assocoations):

        # build map to bind card class to die class (if assigned)
        class_to_card = {}

        for a in assocoations:
            class_to_card[a["die_class"]] = a["card_class"]

        # update each die state
        for die_id, (bbox, cls_name) in tracked_dice.items():
            color, die_value = parse_die_value(cls_name)

            if color is None:
                continue

            card = class_to_card.get(cls_name, None)

            if die_id not in self.state:
                self.state[die_id] = {
                    "card": card,
                    "stable_on": 0,
                    "stable_off": 0,
                    "has_scored": False
                }
            
            st = self.state[die_id]

            if card is not None:
                # die is on a card in thos frame
                st["stable_on"] += 1
                st["stable_off"] = 0

                # Only score if die is stable for  enough frames
                if not st["has_scored"] and st["stable_on"] >= self.STABLE_ON:
                    self.scores[color] += CARD_VALUES[card] * die_value
                    st["has_scored"] = True

            else:
                # die is not on a card in this frame
                st["stable_off"] += 1
                st["stable_on"] = 0

                # only subtract score when die was ON a card and remains OFF for some frames
                if st["has_scored"] and st["stable_off"] >= self.STABLE_OFF:
                    old_card = st["card"]
                    if old_card in CARD_VALUES:
                        self.scores[color] -= CARD_VALUES[card] * die_value
                    st["has_scored"] = False

            # update stored card
            st["card"] = card
        return self.scores
