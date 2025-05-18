from enum import Enum

class EventType(Enum):
    ARRIVAL = "arrival"
    DEPARTURE = "departure"

class Event:
    def __init__(self, event_id, event_type : EventType, event_time):
        if not isinstance(event_type, EventType):
            raise ValueError("event_type must be an instance of EventType Enum")
        self.event_id = event_id
        self.event_type = event_type
        self.event_time = event_time

    def __str__(self):
        return f"Event ID: {self.event_id}, Type: {self.event_type}, Time: {self.event_time}"
    
    