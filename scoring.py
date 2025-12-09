

# CARD_VALUES = {"ace": 1, "two": 2, "three": 3, "four": 4}

# def parse_die_value(cls_name):
#     if not isinstance(cls_name, str):
#         cls_name = str(cls_name)
#     parts = cls_name.split("_")
#     if len(parts) != 2:
#         return None, None
#     return parts[0], int(parts[1])


# class Scoring:
#     def __init__(self):
#         self.scores = {"red": 0, "blue": 0, "yellow": 0}

#         # die_id → state
#         self.state = {}

#         self.STABLE_ON = 5      # frames needed before awarding points
#         self.STABLE_OFF = 5     # frames needed before subtracting

#     def update_scores(self, tracked_dice, assocoations):

#         # build map to bind card class to die class (if assigned)
#         class_to_card = {}

#         for a in assocoations:
#             class_to_card[a["die_class"]] = a["card_class"]

#         # update each die state
#         for die_id, (bbox, cls_name) in tracked_dice.items():
#             color, die_value = parse_die_value(cls_name)

#             if color is None:
#                 continue

#             card = class_to_card.get(cls_name, None)

#             if die_id not in self.state:
#                 self.state[die_id] = {
#                     "card": card,
#                     "stable_on": 0,
#                     "stable_off": 0,
#                     "has_scored": False
#                 }
            
#             st = self.state[die_id]

#             if card is not None:
#                 # die is on a card in thos frame
#                 st["stable_on"] += 1
#                 st["stable_off"] = 0

#                 # Only score if die is stable for  enough frames
#                 if not st["has_scored"] and st["stable_on"] >= self.STABLE_ON:
#                     self.scores[color] += CARD_VALUES[card] * die_value
#                     st["has_scored"] = True

#             else:
#                 # die is not on a card in this frame
#                 st["stable_off"] += 1
#                 st["stable_on"] = 0

#                 # only subtract score when die was ON a card and remains OFF for some frames
#                 if st["has_scored"] and st["stable_off"] >= self.STABLE_OFF:
#                     old_card = st["card"]
#                     if old_card in CARD_VALUES:
#                         self.scores[color] -= CARD_VALUES[card] * die_value
#                     st["has_scored"] = False

#             # update stored card
#             st["card"] = card
#         return self.scores


# CARD_VALUES = {"ace": 1, "two": 2, "three": 3, "four": 4}


# def parse_die(cls_name):
#     try:
#         color, val = cls_name.split("_")
#         return color, int(val)
#     except:
#         return None, None


# class Scoring:
#     def __init__(self):
#         self.scores = {"blue": 0, "red": 0, "yellow": 0}

#         # per die state
#         self.state = {}  # die_id → { last_card, scored, soft_exit_count, hard_exit_count }

#         # thresholds
#         self.SOFT_EXIT_FRAMES = 10    # small jitter tolerance
#         self.HARD_EXIT_FRAMES = 25    # require true exit before subtracting
#         self.STABLE_ENTER_FRAMES = 10 # require stronger stability for scoring

#     def update_scores(self, tracked_dice, associations):
#         # class → card mapping
#         class_to_card = {
#             a["die_class"]: a["card_class"]
#             for a in associations
#         }

#         for die_id, (bbox, cls_name) in tracked_dice.items():
#             color, die_val = parse_die(cls_name)
#             if color is None:
#                 continue

#             current_card = class_to_card.get(cls_name, None)

#             if die_id not in self.state:
#                 self.state[die_id] = {
#                     "last_card": None,
#                     "scored": False,
#                     "enter_count": 0,
#                     "soft_exit_count": 0,
#                     "hard_exit_count": 0,
#                 }

#             st = self.state[die_id]

#             # -----------------------------------------------------
#             # CASE 1: DIE IS ON A CARD
#             # -----------------------------------------------------
#             if current_card is not None:
#                 if current_card == st["last_card"]:
#                     st["enter_count"] += 1
#                 else:
#                     st["enter_count"] = 1
#                     st["scored"] = False

#                 # Reset exits
#                 st["soft_exit_count"] = 0
#                 st["hard_exit_count"] = 0

#                 # Score ONCE
#                 if not st["scored"] and st["enter_count"] >= self.STABLE_ENTER_FRAMES:
#                     if current_card in CARD_VALUES:
#                         self.scores[color] += CARD_VALUES[current_card] * die_val
#                     st["scored"] = True

#             # -----------------------------------------------------
#             # CASE 2: DIE IS OFF CARD
#             # -----------------------------------------------------
#             else:
#                 st["enter_count"] = 0
#                 st["soft_exit_count"] += 1
#                 st["hard_exit_count"] += 1

#                 # Do NOTHING on soft exit → avoid jitter penalty
#                 if st["soft_exit_count"] < self.SOFT_EXIT_FRAMES:
#                     pass

#                 # Hard exit → subtract score if it was scored before
#                 elif st["hard_exit_count"] >= self.HARD_EXIT_FRAMES:
#                     if st["scored"] and st["last_card"] in CARD_VALUES:
#                         self.scores[color] -= CARD_VALUES[st["last_card"]] * die_val

#                     st["scored"] = False
#                     st["last_card"] = None

#             # Update last card at end
#             st["last_card"] = current_card

#         return self.scores


# CARD_VALUES = {"ace" : 1, "two" : 2, "three" : 3, "four" : 4}

# def parse_die_value(cls_name):
#     color, value = cls_name.split("_")
#     return color, int(value)

# class Scoring:
#     def __init__(self):
#         self.scores = {"red":0, "blue":0, "yellow":0}
#         self.previous_assignments = {}

#     def update_scores(self, tracked_dice, associations):
#         new_assignments = {}

#         for die_id, (bbox, cls_name) in tracked_dice.items():
#             for a in associations:
#                 if a["die_class"] == cls_name:
#                     new_assignments[die_id] = a["card_class"]
        
#         for die_id, cls_name in tracked_dice.items():
#             color, die_value = parse_die_value(cls_name)
            
#             old_card = self.previous_assignments.get(die_id)
#             new_card = new_assignments.get(die_id)

#             if old_card != new_card:
#                 if old_card is not None:
#                     self.scores[color] -= CARD_VALUES[old_card] * die_value
                
#                 if new_card is not None:
#                     self.scores[color] += CARD_VALUES[new_card] * die_value

#         self.previous_assignments = new_assignments
#         return self.scores

# PREV WORKING

# CARD_VALUES = {"ace": 1, "two": 2, "three": 3, "four": 4}


# def parse_die_value(cls_name: str):
#     """
#     cls_name: e.g. 'blue_6' -> ('blue', 6)
#     Safely handles unexpected formats.
#     """
#     if not isinstance(cls_name, str):
#         cls_name = str(cls_name)

#     parts = cls_name.split("_")
#     if len(parts) != 2:
#         # Not a die class, fail soft
#         return None, None

#     color, value = parts
#     try:
#         return color, int(value)
#     except ValueError:
#         return None, None


# class Scoring:
#     def __init__(self):
#         self.scores = {"red": 0, "blue": 0, "yellow": 0}
#         # die_id -> card_class (or None)
#         self.previous_assignments = {}

#     def update_scores(self, tracked_dice, associations):
#         """
#         tracked_dice: dict: id -> (bbox, cls_name)
#         associations: list of dicts:
#           {
#             "die_class": "blue_6",
#             "die_centroid": (cx, cy),
#             "card_class": "two" or None
#           }
#         Returns: scores dict
#         """
#         # 1. Build new assignments: die_id -> card_class
#         new_assignments = {}

#         for die_id, (bbox, cls_name) in tracked_dice.items():
#             # Ensure cls_name is a clean string
#             if isinstance(cls_name, tuple):
#                 cls_name = "_".join(str(x) for x in cls_name)
#             elif not isinstance(cls_name, str):
#                 cls_name = str(cls_name)

#             # find best association (same class, closest centroid)
#             x1, y1, x2, y2 = bbox
#             cx_die = (x1 + x2) / 2.0
#             cy_die = (y1 + y2) / 2.0

#             best_card = None
#             best_dist2 = None

#             for a in associations:
#                 if a["die_class"] != cls_name:
#                     continue

#                 ax, ay = a["die_centroid"]
#                 dx = cx_die - ax
#                 dy = cy_die - ay
#                 d2 = dx * dx + dy * dy

#                 if best_dist2 is None or d2 < best_dist2:
#                     best_dist2 = d2
#                     best_card = a["card_class"]

#             new_assignments[die_id] = best_card

#         # 2. Update scores by comparing previous vs new
#         for die_id, (bbox, cls_name) in tracked_dice.items():
#             if isinstance(cls_name, tuple):
#                 cls_name = "_".join(str(x) for x in cls_name)
#             elif not isinstance(cls_name, str):
#                 cls_name = str(cls_name)

#             color, die_value = parse_die_value(cls_name)
#             if color is None or die_value is None:
#                 # skip non-die classes or malformed names
#                 continue

#             old_card = self.previous_assignments.get(die_id)
#             new_card = new_assignments.get(die_id)

#             if old_card != new_card:
#                 # remove old score
#                 if old_card in CARD_VALUES:
#                     self.scores[color] -= CARD_VALUES[old_card] * die_value

#                 # add new score
#                 if new_card in CARD_VALUES:
#                     self.scores[color] += CARD_VALUES[new_card] * die_value

#         self.previous_assignments = new_assignments
#         return self.scores


# scoring.py

# CARD_VALUES = {"ace": 1, "two": 2, "three": 3, "four": 4}


# def parse_die(cls_name: str):
#     try:
#         color, val = cls_name.split("_")
#         return color, int(val)
#     except Exception:
#         return None, None


# def centroid(bbox):
#     x1, y1, x2, y2 = bbox
#     return ( (x1 + x2) / 2.0, (y1 + y2) / 2.0 )


# def centroid_in_shrunken(card_box, c, margin=10):
#     x1, y1, x2, y2 = card_box
#     cx, cy = c
#     x1s = x1 + margin
#     y1s = y1 + margin
#     x2s = x2 - margin
#     y2s = y2 - margin
#     if x2s <= x1s or y2s <= y1s:
#         return False
#     return x1s <= cx <= x2s and y1s <= cy <= y2s


# class Scoring:
#     def __init__(self):
#         self.scores = {"blue": 0, "red": 0, "yellow": 0}

#         # Per-track state
#         # die_id → {
#         #   "last_card": str or None,
#         #   "on_frames": int,
#         #   "off_frames": int,
#         #   "scored": bool,
#         #   "subtracted": bool,
#         # }
#         self.state = {}

#         # thresholds in frames
#         self.STABLE_ON = 10      # ~0.3s at 30fps
#         self.STABLE_OFF = 10     # require persistent off before subtract
#         # Important design: each track can score at most once + subtract once

#     def assign_card_for_die(self, die_bbox, cards):
#         """Return the card_class this die is currently on (or None)."""
#         c = centroid(die_bbox)

#         # 1) Try shrunken bbox containment
#         for card in cards:
#             if centroid_in_shrunken(card["bbox"], c, margin=12):
#                 return card["class"]

#         # 2) If nothing, consider die off-card
#         return None

#     def update_scores(self, tracked_dice, cards):
#         """
#         tracked_dice: dict die_id → (bbox, cls_name)
#         cards: list of { 'class': str, 'bbox': [x1,y1,x2,y2] }
#         """

#         for die_id, (bbox, cls_name) in tracked_dice.items():
#             color, die_val = parse_die(cls_name)
#             if color is None:
#                 continue

#             # Which card is this die on RIGHT NOW (if any)?
#             current_card = self.assign_card_for_die(bbox, cards)

#             # Init state object if new track
#             if die_id not in self.state:
#                 self.state[die_id] = {
#                     "last_card": None,
#                     "on_frames": 0,
#                     "off_frames": 0,
#                     "scored": False,
#                     "subtracted": False,
#                 }

#             st = self.state[die_id]

#             # ---------------------- DIE ON A CARD ----------------------
#             if current_card is not None:
#                 # Same card as before
#                 if current_card == st["last_card"]:
#                     st["on_frames"] += 1
#                 else:
#                     # new card: reset counters & allow scoring again
#                     st["on_frames"] = 1
#                     st["off_frames"] = 0
#                     st["scored"] = False
#                     st["subtracted"] = False

#                 # While on a card we consider off_frames = 0
#                 st["off_frames"] = 0

#                 # Score ONCE if stable and not scored yet
#                 if (not st["scored"]
#                         and not st["subtracted"]
#                         and st["on_frames"] >= self.STABLE_ON
#                         and current_card in CARD_VALUES):

#                     self.scores[color] += CARD_VALUES[current_card] * die_val
#                     st["scored"] = True

#             # ---------------------- DIE OFF ANY CARD -------------------
#             else:
#                 st["on_frames"] = 0
#                 if st["last_card"] is not None:
#                     st["off_frames"] += 1
#                 else:
#                     st["off_frames"] = 0

#                 # We only subtract ONCE per card placement
#                 if (st["scored"]
#                         and not st["subtracted"]
#                         and st["off_frames"] >= self.STABLE_OFF
#                         and st["last_card"] in CARD_VALUES):

#                     # Knock-off / permanent removal
#                     self.scores[color] -= CARD_VALUES[st["last_card"]] * die_val
#                     st["subtracted"] = True

#             # Update last_card to what we saw this frame
#             st["last_card"] = current_card

#         return self.scores


# GEM
# scoring.py
# CARD_VALUES = {"ace": 1, "two": 2, "three": 3, "four": 4}

# def parse_die_value(cls_name):
#     if not isinstance(cls_name, str):
#         cls_name = str(cls_name)
#     parts = cls_name.split("_")
#     if len(parts) != 2:
#         return None, None
#     return parts[0], int(parts[1])


# class Scoring:
#     def __init__(self):
#         # We no longer store self.scores as an accumulator.
#         # We only store the state of every die ID seen.
#         self.die_states = {}

#         # Config: Increased stability requirements
#         self.STABLE_ON_THRESH = 5   # Frames required to latch ON (Start Scoring)
#         self.STABLE_OFF_THRESH = 10 # Frames required to latch OFF (Stop Scoring)

#     def update_scores(self, tracked_dice, associations):
#         """
#         1. Update the state of every tracked die (is it on a card? is it stable?).
#         2. Calculate the total score from scratch based on current stable states.
#         """
        
#         # 1. Map current associations (Die Class -> Card Class)
#         current_frame_associations = {}
#         for assoc in associations:
#             current_frame_associations[assoc["die_class"]] = assoc["card_class"]

#         # 2. Iterate through all currently tracked dice
#         active_ids = set(tracked_dice.keys())
        
#         for die_id, (bbox, cls_name) in tracked_dice.items():
#             color, die_val = parse_die_value(cls_name)
#             if color is None: continue

#             # Determine what card the die is on RIGHT NOW (based on the Geometric Margin check)
#             current_card = current_frame_associations.get(cls_name, None)

#             # Initialize state if new die ID
#             if die_id not in self.die_states:
#                 self.die_states[die_id] = {
#                     "color": color,
#                     "value": die_val,
#                     "confirmed_card": None, # The card we are "locked" onto (scores points)
#                     "pending_card": None,   # The card we see right now (used for debouncing)
#                     "frames_match": 0,      # Counter for stability
#                     "missing_frames": 0     # Counter for disappearance (for garbage collection)
#                 }

#             st = self.die_states[die_id]
#             st["missing_frames"] = 0 # It is present in this frame

#             # --- HYSTERESIS (DEBOUNCING) LOGIC ---
            
#             # Case A: The die is physically on the same card we saw last frame
#             if current_card == st["pending_card"]:
#                 st["frames_match"] += 1
#             else:
#                 # Case B: The die moved to a new place (or empty space)
#                 st["pending_card"] = current_card
#                 st["frames_match"] = 1 # Reset and start counting new state

#             # Case C: We have seen the same state long enough -> Update Confirmed State (Latch ON)
#             if st["pending_card"] is not None and st["frames_match"] >= self.STABLE_ON_THRESH:
#                 st["confirmed_card"] = st["pending_card"]
            
#             # Case D: If pending is None (off card), we need a longer threshold to clear the score (Latch OFF)
#             if st["pending_card"] is None and st["frames_match"] >= self.STABLE_OFF_THRESH:
#                 st["confirmed_card"] = None


#         # 3. Clean up old IDs (Garbage Collection / True Knock-Off)
#         for die_id in list(self.die_states.keys()):
#             if die_id not in active_ids:
#                 # If tracker lost the die, increase its missing count
#                 self.die_states[die_id]["missing_frames"] += 1
#                 # If missing for long enough, remove it entirely
#                 if self.die_states[die_id]["missing_frames"] > self.STABLE_OFF_THRESH:
#                     del self.die_states[die_id]

#         # 4. CALCULATE TOTALS (Snapshot)
#         # Calculates the total score from zero every frame based on confirmed_card
#         totals = {"red": 0, "blue": 0, "yellow": 0}
        
#         for die_id, st in self.die_states.items():
#             card = st["confirmed_card"]
#             if card in CARD_VALUES:
#                 points = st["value"] * CARD_VALUES[card]
#                 totals[st["color"]] += points

#         return totals

# Cl
# # scoring.py - Improved Stable Scoring System

# CARD_VALUES = {"ace": 1, "two": 2, "three": 3, "four": 4}

# def parse_die_value(cls_name):
#     """Parse die class name into color and value."""
#     if not isinstance(cls_name, str):
#         cls_name = str(cls_name)
#     parts = cls_name.split("_")
#     if len(parts) != 2:
#         return None, None
#     try:
#         return parts[0], int(parts[1])
#     except (ValueError, IndexError):
#         return None, None


# class Scoring:
#     def __init__(self):
#         """
#         Initialize scoring system with per-die state tracking.
#         Uses hysteresis (debouncing) to prevent score flickering.
#         """
#         self.die_states = {}
        
#         # Tuned thresholds for stability
#         self.STABLE_ON_THRESH = 5    # Frames to confirm die is ON a card (reduced for faster response)
#         self.STABLE_OFF_THRESH = 12  # Frames to confirm die is OFF a card
#         self.MAX_MISSING_FRAMES = 20 # Frames before die is considered truly gone (tightened)

#     def update_scores(self, tracked_dice, associations):
#         """
#         Calculate scores using snapshot method with hysteresis.
        
#         Args:
#             tracked_dice: dict {die_id: (bbox, cls_name)}
#             associations: list of {die_class, die_centroid, card_class}
            
#         Returns:
#             dict: {"red": score, "blue": score, "yellow": score}
#         """
#         # Build current frame associations map: die_class -> card_class
#         current_associations = {}
#         for assoc in associations:
#             current_associations[assoc["die_class"]] = assoc["card_class"]

#         # Track which die IDs are present in this frame
#         active_ids = set(tracked_dice.keys())

#         # Update state for all tracked dice
#         for die_id, (bbox, cls_name) in tracked_dice.items():
#             color, die_val = parse_die_value(cls_name)
#             if color is None:
#                 continue

#             # Get current card assignment for this die class
#             current_card = current_associations.get(cls_name, None)

#             # Initialize state for new dice
#             if die_id not in self.die_states:
#                 self.die_states[die_id] = {
#                     "color": color,
#                     "value": die_val,
#                     "confirmed_card": None,      # The card we're scoring
#                     "pending_card": None,        # The card we see now
#                     "stable_frames": 0,          # Counter for current state
#                     "missing_frames": 0          # Counter for absence
#                 }

#             state = self.die_states[die_id]
#             state["missing_frames"] = 0  # Die is present this frame

#             # HYSTERESIS LOGIC - Prevents flickering
            
#             # Case 1: Same state as before - increment stability counter
#             if current_card == state["pending_card"]:
#                 state["stable_frames"] += 1
#             else:
#                 # Case 2: State changed - reset counter and update pending
#                 state["pending_card"] = current_card
#                 state["stable_frames"] = 1

#             # Case 3: Confirm transition to ON state (die landed on card)
#             if (state["pending_card"] is not None and 
#                 state["stable_frames"] >= self.STABLE_ON_THRESH):
#                 state["confirmed_card"] = state["pending_card"]

#             # Case 4: Confirm transition to OFF state (die removed from card)
#             if (state["pending_card"] is None and 
#                 state["stable_frames"] >= self.STABLE_OFF_THRESH):
#                 state["confirmed_card"] = None

#         # Garbage collection - Remove dice that have been gone too long
#         for die_id in list(self.die_states.keys()):
#             if die_id not in active_ids:
#                 self.die_states[die_id]["missing_frames"] += 1
                
#                 # If missing for too long, remove completely
#                 if self.die_states[die_id]["missing_frames"] > self.MAX_MISSING_FRAMES:
#                     del self.die_states[die_id]

#         # SNAPSHOT SCORING - Calculate total from scratch each frame
#         totals = {"red": 0, "blue": 0, "yellow": 0}
        
#         for die_id, state in self.die_states.items():
#             card = state["confirmed_card"]
#             if card in CARD_VALUES:
#                 points = state["value"] * CARD_VALUES[card]
#                 totals[state["color"]] += points

#         return totals

# scoring.py - Improved Stable Scoring System

CARD_VALUES = {"ace": 1, "two": 2, "three": 3, "four": 4}

def parse_die_value(cls_name):
    """Parse die class name into color and value."""
    if not isinstance(cls_name, str):
        cls_name = str(cls_name)
    parts = cls_name.split("_")
    if len(parts) != 2:
        return None, None
    try:
        return parts[0], int(parts[1])
    except (ValueError, IndexError):
        return None, None


class Scoring:
    def __init__(self):
        """
        Initialize scoring system with per-die state tracking.
        Uses hysteresis (debouncing) to prevent score flickering.
        """
        self.die_states = {}
        
        # Tuned thresholds for stability
        self.STABLE_ON_THRESH = 5    # Frames to confirm die is ON a card (reduced for faster response)
        self.STABLE_OFF_THRESH = 12  # Frames to confirm die is OFF a card
        self.MAX_MISSING_FRAMES = 20 # Frames before die is considered truly gone (tightened)

    def update_scores(self, tracked_dice, associations):
        """
        Calculate scores using snapshot method with hysteresis.
        
        Args:
            tracked_dice: dict {die_id: (bbox, cls_name)}
            associations: list of {die_class, die_centroid, card_class}
            
        Returns:
            dict: {"red": score, "blue": score, "yellow": score}
        """
        # Build current frame associations map: die_class -> card_class
        current_associations = {}
        for assoc in associations:
            current_associations[assoc["die_class"]] = assoc["card_class"]

        # Track which die IDs are present in this frame
        active_ids = set(tracked_dice.keys())

        # Update state for all tracked dice
        for die_id, (bbox, cls_name) in tracked_dice.items():
            color, die_val = parse_die_value(cls_name)
            if color is None:
                continue

            # Get current card assignment for this die class
            current_card = current_associations.get(cls_name, None)

            # Initialize state for new dice
            if die_id not in self.die_states:
                self.die_states[die_id] = {
                    "color": color,
                    "value": die_val,
                    "confirmed_card": None,      # The card we're scoring
                    "pending_card": None,        # The card we see now
                    "stable_frames": 0,          # Counter for current state
                    "missing_frames": 0          # Counter for absence
                }

            state = self.die_states[die_id]
            state["missing_frames"] = 0  # Die is present this frame

            # HYSTERESIS LOGIC - Prevents flickering
            
            # Case 1: Same state as before - increment stability counter
            if current_card == state["pending_card"]:
                state["stable_frames"] += 1
            else:
                # Case 2: State changed - reset counter and update pending
                state["pending_card"] = current_card
                state["stable_frames"] = 1

            # Case 3: Confirm transition to ON state (die landed on card)
            if (state["pending_card"] is not None and 
                state["stable_frames"] >= self.STABLE_ON_THRESH):
                state["confirmed_card"] = state["pending_card"]

            # Case 4: Confirm transition to OFF state (die removed from card)
            if (state["pending_card"] is None and 
                state["stable_frames"] >= self.STABLE_OFF_THRESH):
                state["confirmed_card"] = None

        # Garbage collection - Remove dice that have been gone too long
        for die_id in list(self.die_states.keys()):
            if die_id not in active_ids:
                self.die_states[die_id]["missing_frames"] += 1
                
                # If missing for too long, remove completely
                if self.die_states[die_id]["missing_frames"] > self.MAX_MISSING_FRAMES:
                    del self.die_states[die_id]

        # SNAPSHOT SCORING - Calculate total from scratch each frame
        totals = {"red": 0, "blue": 0, "yellow": 0}
        
        for die_id, state in self.die_states.items():
            card = state["confirmed_card"]
            if card in CARD_VALUES:
                points = state["value"] * CARD_VALUES[card]
                totals[state["color"]] += points
        
        # Verification: Check for impossible scores
        for color in ["red", "blue", "yellow"]:
            if totals[color] > 100:  # Sanity check: no single color should exceed 100
                print(f"⚠️  WARNING: {color} score {totals[color]} seems too high!")
                print(f"  Contributing dice:")
                for die_id, state in self.die_states.items():
                    if state["color"] == color and state["confirmed_card"] in CARD_VALUES:
                        print(f"    ID {die_id}: {state['value']}×{CARD_VALUES[state['confirmed_card']]} = {state['value'] * CARD_VALUES[state['confirmed_card']]}")

        return totals