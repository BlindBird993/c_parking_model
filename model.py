from mesa import Agent, Model
from schedule import *
from agents import *
from mesa.datacollection import DataCollector#Data collector

class ConceptModel(Model):
    """A model with some number of agents."""

    number_of_car_agents = 10
    number_of_parking_slot_agents = 7

    verbose = True  # Print-monitoring

    def __init__(self, number_of_car_agents = 100,number_of_parking_slot_agents = 80):

        #set params
        self.number_of_car_agents = number_of_car_agents
        self.number_of_parking_slot_agents = number_of_parking_slot_agents


        self.schedule = CustomBaseSheduler(self)
        self.datacollector = DataCollector(
            {"Car agents": lambda m: m.schedule.get_breed_count(CarAgent),
             "Slot agents": lambda m: m.schedule.get_breed_count(ParkingSlotAgent)})

        #parking_slots
        for i in range(self.number_of_parking_slot_agents):
            slot_agent = ParkingSlotAgent("Parking slot "+str(i),self)
            self.schedule.add(slot_agent)

        #free parking
        parking_info = FreeParkingPlaces(0,self)
        self.schedule.add(parking_info)

        #car agents
        for i in range(self.number_of_car_agents):
            car_agent = CarAgent("Car "+str(i), self)
            self.schedule.add(car_agent)

        #control
        control_agent = ControlAgent(0,self)
        self.schedule.add(control_agent)


    def step(self):
        """Advance model by one step"""
        # self.datacollector.collect(self)
        self.schedule.step()
        """Collect data"""
        self.datacollector.collect(self)
        if self.verbose:
            print([self.schedule.time,
                   self.schedule.get_breed_count(CarAgent),
                   self.schedule.get_breed_count(ParkingSlotAgent)])

    def run_model(self,step_count = 7):

        for i in range(step_count):
            print("Step {}".format(i))
            for j in range(24):
                self.step()

