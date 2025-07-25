import heapq


class Scheduler:
    _instance = None

    def __init__(self):
        if Scheduler._instance is not None:
            raise Exception(
                "Use the 'init()' method to get the instance of this class."
            )

    @classmethod
    def init(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            # Initialize instance variables here
            cls._instance.event_queue = []
        return cls._instance

    def schedule(self, event):
        """
        Schedule an event.
        """
        heapq.heappush(
            self.event_queue, event
        )  # heapq implements a min-heap, so the smallest
        # (accordingly to the overloaded "<" in the Event class")
        # event will be at the root
        return event.time

    def get_next_event(self):
        """
        Get the next event from the queue.
        """
        if self.event_queue:
            return heapq.heappop(self.event_queue)
        else:
            return None

    def is_empty(self):
        """
        Check if the event queue is empty.
        """
        return len(self.event_queue) == 0

    def get_queue_length(self):
        """
        Get the length of the event queue.
        """
        return len(self.event_queue)

    def flush(self):
        """
        Flush the event queue.
        """
        self.event_queue = []
