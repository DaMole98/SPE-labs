
import heapq

class Scheduler:
    _instance = None

    def __init__(self):
        if Scheduler._instance is not None:
            raise Exception("Use the 'init()' method to get the instance of this class.")

    @classmethod
    def init(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            # Initialize instance variables here
            cls._instance.event_queue = []
        return cls._instance
    

    def schedule(self,event):
        """
        Schedule an event.
        """
        heapq.heappush(self.event_queue, (event.event_time, event)) #heapq implements a min-heap, so the event with the smallest event_time will be at the top.
                                                                    #tuple is needed to sort the events by event_time
        
        return event.event_time
    
    def get_next_event(self):
        """
        Get the next event from the queue.
        """
        if self.event_queue:
            return heapq.heappop(self.event_queue)[1]
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