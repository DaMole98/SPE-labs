from scheduler import Scheduler
from event import Event, EventType
from mm1 import MM1

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
        self.step_num = 0
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        """Reset the MM1 server, scheduler, simulation clock, and counters."""
        self.mm1_srv = MM1(arrival_rate=self.arrival_rate,
                           service_rate=self.service_rate,
                           queue=0, busy=False)
        self.scheduler = Scheduler.init()
        # In case the Scheduler singleton retains leftover events
        self.scheduler.event_queue = []

        self.dtime = 0.0
        self.narr = 0
        self.ndep = 0
        self.offset = 0.0

    def schedule_initial_events(self):
        """Schedule the first arrival and the first departure events."""
        t_arr = self.rng.exponential(1.0 / self.arrival_rate)
        e_arr = Event(event_id=0, event_type=EventType.ARRIVAL, event_time=t_arr)
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


        event = self.scheduler.get_next_event()
        self.dtime = event.event_time


        if event.event_type == EventType.ARRIVAL:
            t_next_arr = self.dtime + self.rng.exponential(1.0 / self.arrival_rate)
            e_next_arr = Event(event_id=self.narr,
                           event_type=EventType.ARRIVAL,
                           event_time=t_next_arr)
            self.scheduler.schedule(e_next_arr)
        
            self.narr += 1
            was_busy = self.mm1_srv.is_busy()
            self.mm1_srv.arrival()

            if was_busy:
                # If the server was busy, schedule the next departure
                t_next_dep = self.dtime + self.rng.exponential(1.0 / self.service_rate)
                e_next_dep = Event(event_id=self.ndep,
                               event_type=EventType.DEPARTURE,
                               event_time=t_next_dep)
                self.scheduler.schedule(e_next_dep)


        elif event.event_type == EventType.DEPARTURE:
            self.ndep += 1
            self.mm1_srv.departure()
            if self.mm1_srv.is_busy():
                t_next_dep = self.dtime + self.rng.exponential(1.0 / self.service_rate)
                e_next_dep = Event(event_id=self.ndep,
                               event_type=EventType.DEPARTURE,
                               event_time=t_next_dep)
                self.scheduler.schedule(e_next_dep)



    def warmup(self, warmup_events=1000):
        """
        Warm-up phase: skip the first `warmup_events` events
        to eliminate the effect of initial conditions.
        """
        self.schedule_initial_events()
        for _ in range(warmup_events):
            self.step()
        self.offset = self.dtime

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
        else:
            self.schedule_initial_events()

        # Main simulation loop
        while not termination_condition(self.mm1_srv,self.scheduler,self.narr,self.ndep,self.dtime):
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
            lam_obs = self.narr / total
            mu_obs = self.ndep / total
            print(f"Observed arrival rate: {lam_obs:.4f}")
            print(f"Observed service rate: {mu_obs:.4f}")



if __name__ == "__main__":
    # Define termination policy: stop after 10,000 processed events
    policy = (TerminationPolicy()
              .add(lambda srv, sched, narr, ndep, t: (narr + ndep) >= int(1e6))
              .all())

    sim = Simulator(arrival_rate=1.0, service_rate=2.0, seed=42)
    sim.run(termination_condition=policy, warmup=False, warmup_events=10000)
    sim.report()
