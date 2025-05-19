from enum import Enum

class EventType(Enum):
    ARRIVAL = "arrival"
    DEPARTURE = "departure"

class Event:
    def __init__(self, id, type : EventType, time):
        if not isinstance(type, EventType):
            raise ValueError("type must be an instance of EventType Enum")
        self.id = id
        self.type = type
        self.time = time

    def __str__(self):
        return f"Event ID: {self.id}, Type: {self.type}, Time: {self.time}"
    
    #overload the less than operator to compare events
    def __lt__(self, other: "Event") -> bool:
        if not isinstance(other, Event):
            return NotImplemented
        
        # Compare event times first
        if self.time != other.time:
            return self.time < other.time
        # Compare event types if times are equal
        # If event types are different, prioritize ARRIVAL over DEPARTURE
        if self.type != other.type:
            return True if self.type == EventType.ARRIVAL else False
        # Compare event IDs if both times and types are equal
        return self.id < other.id