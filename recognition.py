import classify_dice

def setup_recog():
    """
    called once, at the start of a round/series of rounds
    - quick-trains player classification model to associate die colors with player indices
    - loads obb model?
    - loads value model?

    """
    ...

def infer_recog(obb_model, player_model, value_model, frame):
    """
    called every frame of the game (or at some other acceptable frequency)

    inputs:
        bounding box model
        player classification model
        value classification model
        video frame (ideally would like to gracefully handle whole videos, but this is a later concern)
    outputs:
        for each significant object in the frame, an oriented bounding box and a classification (ace, two, three, four, die) and for dice, a further specification of the player index and the value of that die
    """
    ...
