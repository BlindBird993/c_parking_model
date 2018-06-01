from mesa import Agent, Model
import random
import numpy as np
import collections as col
import operator
import copy

class ControlAgent(Agent):
    def __init__(self,unique_id, model):
        super().__init__(unique_id, model)

        self.hour = 0
        self.day = 0
        self.week = 0

        self.buyers = []
        self.sellers = []

        self.demands = []
        self.productions = []

        self.historyDemands = []
        self.historyProductions = []

        self.distributedDemands = []
        self.summedDemands = []

        self.buyerPriceList = []

        self.demandPrice = []
        self.supplyPrice = []

        self.priceLimit  = 0

        self.totalProduction = 0
        self.listOfConsumers = None
        self.dictOfConsumers = None

        self.numberOfBuyers = 0
        self.numberOfSellers = 0
        self.numberOfConsumers = 0

    def getPriceLimit(self):
        self.priceLimit = round(random.uniform(50,200),1)
        print("Parking price {}".format(self.priceLimit))

    def getConsumerDict(self):
        self.listOfConsumers = []
        check_list = self.buyers
        customer_count = self.numberOfBuyers
        print(check_list)
        for agent in self.model.schedule.agents:
            if (isinstance(agent,CarAgent) and agent.readyToBuy is True):
                self.listOfConsumers.append((agent.unique_id,agent.price))
        self.dictOfConsumers = dict(self.listOfConsumers)
        x = sorted(self.dictOfConsumers.items(), key=operator.itemgetter(1))
        x.reverse()
        self.dictOfConsumers = dict(x)
        print("Consumers {}".format(self.dictOfConsumers))

    def getSellers(self):
        self.numberOfSellers = 0
        self.sellers = []
        for agent in self.model.schedule.agents:
            if (isinstance(agent, ParkingSlotAgent) and agent.readyToSell is True):
                    self.numberOfSellers += 1
                    self.sellers.append(agent.unique_id)

        self.historyProductions.append(self.numberOfSellers)
        print("List of Sellers {}".format(self.sellers))
        print("Number of sellers {}".format(self.numberOfSellers))

    def getBuyres(self):
        self.numberOfBuyers = 0
        self.buyers = []
        for agent in self.model.schedule.agents:
            if (isinstance(agent, CarAgent) and agent.readyToBuy is True):
                    self.numberOfBuyers += 1
                    self.buyers.append(agent.unique_id)

        self.historyDemands.append(self.numberOfBuyers)
        print("List of Buyers {}".format(self.buyers))
        print("Number of buyers {}".format(self.numberOfBuyers))

    def calculateFitness(self,test_vector):
        fitness = 0
        summedDemand = 0.0
        summedValue = 0.0
        lst = []
        size_param = 0
        x = sorted(self.dictOfConsumers.items(), key=operator.itemgetter(1))
        x.reverse()
        for index,elem in enumerate(test_vector):
            if elem > 0:
                size_param += 1
                lst.append(list(x[index]))
        for elem in lst:
            summedValue += elem[1]
            summedValue = round(summedValue,3)
        #calculate fitness
        if size_param > self.numberOfSellers:
            fitness = 0.0
        else:
            fitness = summedValue
        return fitness

    def generatePopulation(self):

        if self.numberOfBuyers > 1:
            popSize = self.numberOfBuyers
        else:
            print("Not enough buyers")
            popSize = self.numberOfBuyers

        vector_list = []
        n, p = 1, 0.5
        d = col.defaultdict(list)
        for i in range(600):
            pop = np.random.binomial(n, p, popSize)
            pop = list(pop)
            self.calculateFitness(pop)
            d[self.calculateFitness(pop)].append(pop)
            vector_list.append((pop,self.calculateFitness(pop)))
        print("Population {}".format(vector_list))
        # print(d.items())
        ordered_dict = col.OrderedDict(sorted(d.items(), key=lambda t: t[0], reverse=True))
        print("Ordered dict {}".format(ordered_dict))
        sorted_x = sorted(ordered_dict.items(), key=operator.itemgetter(0))
        print("Sorted list {}".format(sorted_x))
        if self.numberOfSellers > 0:
            sorted_x.reverse()
        chosen_elem_list = sorted_x[0]
        print("Best element {}".format(list(chosen_elem_list)[1][0]))
        print("Best fitness {}".format(list(chosen_elem_list)[0]))

        if list(chosen_elem_list)[0] <= 0.0:
            print("Unable to satisfy demand!")
            self.summedDemands.append(0)
            return 0

        dna = list(chosen_elem_list)[1][0]

        #tournament
        tournament_pool = []
        for i in range(5):
            elem = random.choice(list(d.items()))
            tournament_pool.append(elem)
        print("Tournament pool {}".format(tournament_pool))
        tournament_dict = col.OrderedDict(sorted(tournament_pool, key=lambda t: t[1], reverse=True))
        print(tournament_dict)

        mating_partners = list(tournament_dict.items())

        partners_list = []
        for elem in mating_partners:
            partners_list.append(list(elem)[1][0])
        #get dna for mating
        number_of_partners = 2
        partner = partners_list[0]
        print("Partner {}".format(partner))

        for i in range(600):
            coef = np.random.uniform(0, 1, 1)
            if coef > 0.8:
                dna1 = copy.deepcopy(dna)
                mutated_dna = self.mutate(dna1,self.numberOfBuyers)

                fitness_mutated = self.calculateFitness(mutated_dna)
                fitness_old = self.calculateFitness(dna)

                if fitness_mutated > fitness_old:
                    dna = mutated_dna
            else:
                # print("Crossover")
                cross_dna1,cross_dna2 = self.crossover(dna,partner,self.numberOfBuyers)
                fitnes_corss1 = self.calculateFitness(cross_dna1)
                fitnes_corss2 = self.calculateFitness(cross_dna2)
                fitness_old = self.calculateFitness(dna)

                if fitnes_corss1 > fitness_old:
                    dna = cross_dna1

                elif fitnes_corss2 > fitness_old:
                    dna = cross_dna2

        print("Chosen DNA {}".format(dna))
        self.decodeList(dna)

    def crossover(self,dna1, dna2, dna_size):
        pos = int(random.random() * dna_size)
        return (dna1[:pos] + dna2[pos:], dna2[:pos] + dna1[pos:])

    def mutate(self,dna,size):
        mutation_chance = 100 #mutation chance
        for index, elem in enumerate(dna):
            if int(random.random() * mutation_chance) == 1:
                if dna[index] == 1:
                    dna[index] = 0
                else:
                    dna[index] = 1
        return dna

    def decodeList(self,dna_variant):
        buyerNumber = 0
        buyers_list = list(self.dictOfConsumers.items())
        print(list(buyers_list))
        for index,elem in enumerate(dna_variant):
            if elem > 0:
                buyerNumber +=1
                print("CAR Agent TO SET {}".format(list(buyers_list[index])))
                self.distributeCars(list(buyers_list[index]))
        self.summedDemands.append(buyerNumber)

    def setParkingSlots(self,car,slot_id):
        for agent in self.model.schedule.agents:
            if (isinstance(agent, ParkingSlotAgent) and agent.readyToSell is True):
                if agent.unique_id == slot_id:
                    agent.status = 'busy'
                    agent.readyToSell = False
                    agent.busyTime = car.parkingTime

                    car.status = 'busy'
                    car.readyToBuy = False
                    car.busyTime = car.parkingTime
                    print("Car {} busy time {}".format(car.unique_id,car.busyTime))
                    print("Slot {} busy time {}".format(agent.unique_id,agent.busyTime))

                    agent.queue.append(car.unique_id)
                    print("Queue {}".format(agent.queue))

                    self.numberOfBuyers -= 1
                    self.numberOfSellers -= 1

                    self.buyers.remove(car.unique_id)
                    self.sellers.remove(agent.unique_id)

                    self.dictOfConsumers = self.removeItem(self.dictOfConsumers, car.unique_id)

    def distributeCars(self,data_list):
        seller_id = np.random.choice(self.sellers)
        print("Seller {}".format(seller_id))
        for agent in self.model.schedule.agents:
             if (isinstance(agent, CarAgent)):
                if (agent.unique_id == data_list[0] and agent.readyToBuy is True):
                    #distribution
                    self.buyerPriceList.append(agent.price)
                    self.setParkingSlots(agent,seller_id)
                    print("Total amount of parking places {}".format(self.numberOfSellers))

        print("Consumers dictionary {}".format(self.dictOfConsumers))

    def checkIfConsumersLeft(self):
        if self.dictOfConsumers:
            ordered_dict = col.OrderedDict(sorted(self.dictOfConsumers.items(), key=lambda t: t[1],reverse=True))
            print("Consumers left {}".format(ordered_dict.items()))
            for k,v in ordered_dict.items():
                for agent in self.model.schedule.agents:
                    if (isinstance(agent, CarAgent)):
                        if (agent.unique_id == k and agent.readyToBuy is True):
                            print("Agent {}".format(agent.unique_id))
                            print("Total amount of parking places {}".format(self.numberOfSellers))

                            if self.numberOfSellers > 0:
                                print("Distribute left customers")
                                seller_id = np.random.choice(self.sellers)
                                self.setParkingSlots(agent,seller_id)

            print("Customers left {}".format(self.dictOfConsumers))
        else:
            print("All demands are satisfied")

    # delete elements from set of buyers
    def removeItem(self,d,key):
        del d[key]
        return d

    def test_func(self):
        print("Control Agent {}".format(self.unique_id))

    def step(self):
        self.test_func()
        self.getPriceLimit()
        self.getSellers()
        self.getBuyres()
        self.getConsumerDict()
        if self.numberOfSellers > 0 and self.numberOfBuyers > 0:
            self.generatePopulation()
            self.checkIfConsumersLeft()
        else:
            print("Not enough sellers or no buyers")
        self.hour +=1

        if self.hour > 23:
            self.day += 1
            self.hour = 0

        if self.day > 7:
            self.week += 1
            self.day = 0

class CarAgent(Agent):
    def __init__(self,unique_id, model):
        super().__init__(unique_id, model)

        self.needToWash = False
        self.wantToPark = True
        self.price = 0
        self.traided = None
        self.readyToBuy = True

        self.parkingTime = 0
        self.priceHistory = []
        self.priorityHistorySell = []
        self.priorityHistoryBuy = []

        self.busyTime = 0

        self.status = 'free'
        self.hour = 0
        self.day = 0
        self.week = 0

    def checkBusyTime(self):
        if self.busyTime > 0:
            self.status = 'busy'
            self.wantToPark = False
            self.busyTime -= 1
        else:
            self.busyTime = 0
            self.status = 'free'
        print("Busy time {}".format(self.busyTime))

    def getParkingTime(self):
        if not self.wantToPark:
            self.parkingTime = 0
        else:
            self.parkingTime = random.randint(1,5) #from 1 to max available time of parking
        print("Desirable parking time {}".format(self.parkingTime))

    def checkIfPark(self):
        if self.status == 'free':
            if self.hour >= 7 and self.hour <= 9:
                self.wantToPark = np.random.choice([True, False], p=[0.9, 0.1])
            elif self.hour >= 15 and self.hour <= 17:
                self.wantToPark = np.random.choice([True, False], p=[0.9, 0.1])
            else:
                self.wantToPark = np.random.choice([True, False])
        print("Want to park {}".format(self.wantToPark))

    def getTradeStatus(self):
        if self.wantToPark:
            self.readyToBuy = True
        else:
            self.readyToBuy = False

    def calculatePrice(self):
        self.price = round(random.uniform(50,200),1) #price for parking, NOK
        print("Price {}".format(self.price))

    def name_func(self):
        print("Agent {}".format(self.unique_id))

    def step(self):
        self.name_func()
        self.checkBusyTime()
        self.checkIfPark()
        self.getTradeStatus()
        self.getParkingTime()
        self.calculatePrice()
        self.hour += 1

        if self.hour > 23:
            self.day += 1
            self.hour = 0

        if self.day > 7:
            self.week += 1
            self.day = 0

class FreeParkingPlaces(Agent):
    def __init__(self,unique_id, model):
        super().__init__(unique_id, model)
        self.freeSpace = 0

    def calculateFreePlaces(self):
        self.freeSpace = 0
        for agent in self.model.schedule.agents:
            if (isinstance(agent, ParkingSlotAgent)):
                if agent.status == 'free':
                    self.freeSpace += 1
        print("Amount of free parking slots {}".format(self.freeSpace))

    def step(self):
        self.calculateFreePlaces()

class ParkingSlotAgent(Agent):
    def __init__(self,unique_id, model):
        super().__init__(unique_id, model)

        self.price = 0
        self.status = 'free' #can be free or busy
        self.queue = []
        self.readyToSell = True

        self.busyTime = 0
        self.hour = 0
        self.day = 0
        self.week = 0

    def updateQueue(self):
        if self.status == 'free':
            self.queue = []

    def checkBusyTime(self):
        if self.busyTime > 0:
            self.status = 'busy'
            self.wantToPark = False
            self.busyTime -= 1
        else:
            self.busyTime = 0
            self.status = 'free'
        print("Busy time {}".format(self.busyTime))

    def getSellStatus(self):
        if self.status == 'free':
            self.readyToSell = True
        else:
            self.readyToSell = False

    def getStatus(self):
        print(self.status)

    def calculatePrice(self):
        self.price = round(random.uniform(50,200),1) #price for parking, NOK
        print("Price {}".format(self.price))

    def name_func(self):
        print("Agent {}".format(self.unique_id))

    def step(self):
        self.name_func()
        self.checkBusyTime()
        self.updateQueue()
        self.getStatus()
        self.getSellStatus()

        self.hour += 1

        if self.hour > 23:
            self.day += 1
            self.hour = 0

        if self.day > 7:
            self.week += 1
            self.day = 0

