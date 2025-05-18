from scheduler import Scheduler
from event import Event, EventType
from mm1 import MM1

import numpy as np



def simulator_init():
    """
    Initialize the simulator.
    """
    #initialize system state (empty queue, server free)
    mm1_server = MM1(arrival_rate=1.0, service_rate=2.0, queue=0, busy=False)

    # Initialize the scheduler
    scheduler = Scheduler.init()



    # Initialize the simulation time and counters
    simulation_time = 0
    num_arrivals = 0
    num_departures = 0


    return mm1_server, scheduler, simulation_time, num_arrivals, num_departures


def simulator_main():

    rng = np.random.default_rng(42)
    mm1_srv, scheduler, dtime, narr, ndep = simulator_init()
    
    # Schedule the first arrival
    arrival_time = rng.exponential(1.0 / mm1_srv._arrival_rate)
    arrival_event = Event(event_id=0, event_type=EventType.ARRIVAL, event_time=arrival_time)
    scheduler.schedule(arrival_event)

    # Schedule the first departure
    departure_time = arrival_time + rng.exponential(1.0 / mm1_srv._service_rate)
    departure_event = Event(event_id=0, event_type=EventType.DEPARTURE, event_time=departure_time)
    scheduler.schedule(departure_event)

    # Main simulation loop
    while not scheduler.is_empty():
        # Get the next event
        event = scheduler.get_next_event()
        dtime = event.event_time

        # Process the event
        if event.event_type == EventType.ARRIVAL:
            narr += 1
            mm1_srv.arrival()
            # Schedule the next arrival
            arrival_time = dtime + rng.exponential(1.0 / mm1_srv._arrival_rate)
            arrival_event = Event(event_id=narr, event_type=EventType.ARRIVAL, event_time=arrival_time)
            scheduler.schedule(arrival_event)

        elif event.event_type == EventType.DEPARTURE:
            ndep += 1
            mm1_srv.departure()
            # Schedule the next departure if the server is busy
            if mm1_srv.is_busy():
                departure_time = dtime + rng.exponential(1.0 / mm1_srv._service_rate)
                departure_event = Event(event_id=ndep, event_type=EventType.DEPARTURE, event_time=departure_time)
                scheduler.schedule(departure_event)

    return True




















if __name__ == "__main__":
    simulator_main()