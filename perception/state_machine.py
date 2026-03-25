class GuidanceStateMachine:
    def __init__(self):
        self.state = "SEARCHING"

    def update_state(self, perception_data, distance):
        object_found = perception_data["object_found"]
        hand_found = perception_data["hand_found"]
        hand_open = perception_data["hand_open"]

        if not object_found:
            self.state = "SEARCHING"
            return self.state

        if not hand_found:
            self.state = "SEARCHING"
            return self.state

        if distance > 80:
            self.state = "GUIDING"
            return self.state

        if distance <= 80 and hand_open:
            self.state = "READY_TO_GRAB"
            return self.state

        if distance <= 80 and hand_open is False:
            self.state = "GRABBED"
            return self.state

        return self.state