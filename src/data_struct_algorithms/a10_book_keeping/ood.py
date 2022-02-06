

# https://github.com/donnemartin/system-design-primer
# clarify requirements
# hash out primary use cases
# identify key objects/operations
# identify key interactions
# abstraction, encapsulation, inheritance, and polymorphism

# https://github.com/zxqiu/leetcode-lintcode/blob/master/system%20design/Load_Balancer.java
# https://zhengyang2015.gitbooks.io/lintcode/content/load_balancer_526.html

# Load Balancer
class LoadBalancer:
    def __init__(self, servers):
        self.routing_policy = RoundRobin(servers)
    def dispatch(self, url):  # ask this interface
        svr = self.routing_policy.next_server()
        svr.handle(url) # ask how to dispatch
class RoutingPolicy:
    def next_server(self):
        pass
class RoundRobin:
    def __init__(self, servers):
        self.servers = servers
        self.idx = 0
    def next_server(self):
        svr = self.servers[self.idx]
        self.idx = self.idx+1 if self.idx+1 < len(self.server) else 0
# or use heaps for loads for routing, Least traffice or utilization, or sticky

# Elevator Design
from enum import Enum
class ElevatorState(Enum):
    UP = 1
    DOWN = 2
    IDLE = 3
# For single elevator
# group floors or request floors
# https://en.wikipedia.org/wiki/Elevator_algorithm
# https://en.wikipedia.org/wiki/LOOK_algorithm
# https://www.quora.com/Is-there-any-public-elevator-scheduling-algorithm-standard
# https://www.programmersought.com/article/3196375498/
# https://www.popularmechanics.com/technology/infrastructure/a20986/the-hidden-science-of-elevators/
# https://ux.stackexchange.com/questions/94442/what-is-the-rationale-behind-the-way-modern-elevator-dispatch-systems-are-implem
# http://incompleteideas.net/book/first/ebook/node111.html
class ElevatorNextStopStrategy:
    def next_stop(self, curr_floor, curr_state, dest_floors, req_from_floors):
        pass
class Elevator:
    def __init__(self):
        self.operating = 0 # shutdown or operating
        self.current_floor = -1
        self.current_state = ElevatorState.IDLE
        self.dest_floors = [] # or sorted list by floor or req time
        self.req_from_floors = []
        self.next_stop_strategy = None
    def select_floor(self):
        pass
    def request_from_floor(self):
        pass

# Write an algorithm for, user req should be completed in logN time in a N-story
# building with M elevators.
# binary search m evevator positions to find floor/ceiling around user florr,
# then select the closer one.

# LC1603. Design Parking System
# Parking Lot Design
# Entities: ParkingTicket, ParkingLot(with list of spaces), SpaceRate
# PaymentService, TicketService ParkingService(handle rates)
class ParkingTicket: # id, enter_time, exit_time, paid(boolean)
    pass
class TicketService: # issue_ticket, receive_ticket(open gate or not)
    pass
class Payment: # ticket_id, lot_rate, pay_method, pay_amount, timestamp
    pass
class PaymentService:
    def charge(self, ticket): # credit card or cash
        pass
class ParkingAdmin:
    pass
    # rates admin, daily, monthly
    # show vacancies near all gates
# I've never seen a parking lot with these settings.
# if we have different size lots, we need to model ParkingSpace, large, mid, small
# if we care car size, we need to model Vehicle, large, mid, small
# https://www.educative.io/courses/grokking-the-object-oriented-design-interview/gxM3gRxmr8Z

# LC535. Encode and Decode TinyURL
from random import choices
chars = string.ascii_letters + string.digits  # upper, lower, digits
class Codec:
    def __init__(self):
        self.url2code = {}
        self.code2url = {}
    def encode(self, longUrl: str) -> str:
        while longUrl not in self.url2code: # or pre-generate for speed
            code = ''.join(choices(chars, k=7))
            if code not in self.code2url:
                self.code2url[code] = longUrl
                self.url2code[longUrl] = code
        return 'http://tinyurl.com/' + self.url2code[longUrl]
    def decode(self, shortUrl: str) -> str:
        return self.code2url[shortUrl[-7:]]
# Base 62, UUID
# https://www.tfzx.net/article/1230313.html

# ticket booking, movies, airlines, hotels, Car rental
class Ticket:
    pass # seat id, price, movie id
class Seat:
    pass # Cinema, theatre, seat id, seat location
class Show:
    pass # show id, movie name, theatre id, list of seats
class SeatReservationSerivce:
    pass # for open seat selection for a given show
class PriceService:
    pass
class PaymentService:
    pass

# Trading System

# Vending machines
class VendingMachine:
    slots = {} # map slot to product
    keypad = []
    payment = 0
    def select_product(self): pass
    def deliver_product(self): pass
    def receive_cash(self): pass  # insert coins
    def receive_credit_card(self): pass  # swipe_card
    def refund(self): pass
class Product: pass
    # name, price, category
class CreditCardService: pass
class CashService: pass
class RestockService: pass
class AppException: pass

# Amazon go

# Card games, blackjack

# Traffic Light
