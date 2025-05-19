from scheduler import Scheduler
from event import Event, EventType
from mm1 import MM1
import heapq

import numpy as np

class TerminationPolicy:
    def __init__(self):
        self.conditions = []

    def add(self, condition):
        """Add a single termination condition (callable srv, sched, narr, ndep, t → bool)."""
        self.conditions.append(condition)
        return self

    def any(self):
        """Return a predicate that terminates if any condition is True."""
        return lambda srv, sched, narr, ndep, t: any(
            cond(srv, sched, narr, ndep, t) for cond in self.conditions
        )

    def all(self):
        """Return a predicate that terminates only if all conditions are True."""
        return lambda srv, sched, narr, ndep, t: all(
            cond(srv, sched, narr, ndep, t) for cond in self.conditions
        )

class Simulator:
    def __init__(self, arrival_rate=1.0, service_rate=2.0, seed=42):
        """
        Configure simulation parameters (arrival_rate, service_rate, seed).
        reset() initializes the internal state.
        """
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.reset()

        self.old_dtime = 0.0

    def reset(self):
        """Reset the MM1 server, scheduler, simulation clock, and counters."""
        self.mm1_srv = MM1(arrival_rate=self.arrival_rate,
                           service_rate=self.service_rate,
                           queue=0, busy=False)
        self.scheduler = Scheduler.init()
        # In case the Scheduler singleton retains leftover events
        self.scheduler.event_queue = []

        self.step_num = 0
        self.abs_t = 0.0
        self.arr_t = 0.0
        self.dep_t = 0.0
        self.offset_t = 0.0
        self.narr = 0
        self.ndep = 0

    def schedule_initial_events(self):
        """Schedule the first arrival and the first departure events."""
        t_arr = self.rng.exponential(1.0 / self.arrival_rate)
        e_arr = Event(id=0, type=EventType.ARRIVAL, time=t_arr)
        self.scheduler.schedule(e_arr)


    def step(self):
        """
        Execute a single event:
        - extract the next event,
        - update simulation time and counters,
        - call arrival()/departure() on the MM1 server,
        - schedule the next arrival or departure as needed.
        """
        self.step_num += 1

        old_t = self.abs_t
        event = self.scheduler.get_next_event()
        self.abs_t = event.time

        print(f"Event time: {event.time:.3f}, event type: {event.type}, event id: {event.id}")
        assert old_t < self.abs_t, "Event time must be non-decreasing"

        if event.type == EventType.ARRIVAL:
            self.arr_t = event.time
            self.narr += 1

            srv_busy = self.mm1_srv.is_busy()
            self.mm1_srv.arrival()

            if not srv_busy:
                # If the server was free, schedule the departure of the incoming request
                t_next_dep = self.arr_t + self.rng.exponential(1.0 / self.service_rate)
                e_next_dep = Event(id=f"{self.ndep}D",
                               type=EventType.DEPARTURE,
                               time=t_next_dep)
                self.scheduler.schedule(e_next_dep)

            #schedule the next arrival
            t_next_arr = self.arr_t + self.rng.exponential(1.0 / self.arrival_rate)
            e_next_arr = Event(id=f"{self.narr}A",
                           type=EventType.ARRIVAL,
                           time=t_next_arr)
            self.scheduler.schedule(e_next_arr)

        elif event.type == EventType.DEPARTURE:
            self.dep_t = event.time
            self.ndep += 1
            self.mm1_srv.departure()
            if self.mm1_srv.is_busy():
                # If the server is still busy, schedule the next departure
                t_next_dep = self.dep_t + self.rng.exponential(1.0 / self.service_rate)
                e_next_dep = Event(id=f"{self.ndep}D",
                               type=EventType.DEPARTURE,
                               time=t_next_dep)
                self.scheduler.schedule(e_next_dep)


    def warmup(self, warmup_events=1000):
        """
        Warm-up phase: skip the first `warmup_events` events
        to eliminate the effect of initial conditions.
        """
        print(f"Running warmup phase with {warmup_events} events")
        self.schedule_initial_events()
        for _ in range(warmup_events):
            self.step()
        print(f"Warmup phase completed. Final time: {self.abs_t:.3f}")
        print(f"Final queue length: {self.mm1_srv.get_queue_length()}")
        print(f"Server busy: {self.mm1_srv.is_busy()}")
        print("Resetting simulation state after warmup.")
        self.offset_t = self.abs_t

        # offset the event times in the scheduler
        events = []
        for evt in self.scheduler.event_queue:
            new_t = evt.time - self.offset_t
            evt.time = new_t
            events.append(evt)

        # Flush the scheduler and reschedule the events
        self.scheduler.flush()
        for evt in events:
            self.scheduler.schedule(evt)
        self.step_num = 0
        self.abs_t = 0.0
        self.arr_t = 0.0
        self.dep_t = 0.0
        self.narr = 0
        self.ndep = 0



    def run(self,
            termination_condition,
            warmup=False,
            warmup_events=1000):
        """
        Run the simulation until `termination_condition(srv, sched, narr, ndep, t)` returns True.
        If warmup is True, perform the warm-up phase first.
        """
        # Full reset + reset RNG for reproducibility
        self.reset()
        self.rng = np.random.default_rng(self.seed)

        if warmup:
            self.warmup(warmup_events)
            print("-------------------------------")
            print("-------------------------------")

        else:
            print("Skipping warmup phase.")
            self.schedule_initial_events()


        print("Running simulation...")
        # Main simulation loop
        while not termination_condition(self.mm1_srv,self.scheduler,self.narr,self.ndep,self.abs_t):
            self.step()



    def report(self):
        """Print final simulation metrics."""
        qlen = self.mm1_srv.get_queue_length()
        busy = self.mm1_srv.is_busy()
        print(f"Arrivals: {self.narr}")
        print(f"Departures: {self.ndep}")
        print(f"Final queue length: {qlen}")
        print(f"Server busy: {busy}")
        total = self.narr + self.ndep
        if total > 0:
            print(f"Total Arrivals: {self.narr:.3f}")
            print(f"Total Services: {self.ndep:.3f}")



if __name__ == "__main__":
    # Define termination policy: stop after 10,000 processed events
    policy = (TerminationPolicy()
              .add(lambda srv, sched, narr, ndep, t: t >= 120.0)
              .all())

    arrival_rate = 1.0
    service_rate = 2.0
    warmup = True
    print(f"Running simulation with arrival rate: {arrival_rate}, service rate: {service_rate}, warmup: {warmup}")
    sim = Simulator(arrival_rate=1.0, service_rate=2.0, seed=42)
    sim.run(termination_condition=policy, warmup=warmup, warmup_events=1000)
    sim.report()
