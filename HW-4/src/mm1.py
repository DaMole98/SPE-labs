
class MM1:
    def __init__(self, arrival_rate, service_rate, queue, busy):
        self._arrival_rate = arrival_rate
        self._service_rate = service_rate
        self._queue = queue
        self._busy = busy

    def arrival(self):
        if not self._busy:
            self._busy = True
        elif self._busy:
            self._queue += 1
        else:
            raise Exception("Arrival: inconsistent state of the server: busy flag: {} and queue length: {}".format(self._busy, self._queue))
        return self._queue # return to the scheduler the number of requests in the queue. INVARIANT: self.busy == True
    
    def departure(self):
        assert self._busy, "INVARIANT VIOLATED: Departure called when server is not busy"
        if self._queue == 0:
            self._busy = False
        elif self._queue > 0:
            self._queue -= 1
        else:
            raise Exception("Departure: inconsistent state of the server: busy flag: {} and queue length: {}".format(self._busy, self._queue))
        return self._queue # return to the scheduler the number of requests in the queue. if queue > 0, schedule next departure
        # INVARIANT: self.busy == True if queue > 0

    def get_queue_length(self):
        return self._queue
    
    def is_busy(self):
        return self._busy
    
    def enqueue(self):
        self.queue += 1
        return self.queue
    