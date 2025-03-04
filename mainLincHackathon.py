# This is a sample Python script.
import hackathon_linc as hl
import pandas_ta as pta
import pandas as pd
import numpy as np
from utils import moving_average, computeRSI
from collections import defaultdict
import csv
import math
import time
import os
import sys
status = "LIVE"
testData = {}
token = "25ddd34d-e22e-4dcd-97c3-e3efae1d404b"
hl.init(token)

def pullData():
    if status=="TEST":
        # gmtTime,askMedian,bidMedian,askVolume,bidVolume,spreadMedian,symbol
        filename = 'backtest.csv'
        d_by_time = defaultdict(list)
        d_by_stock = defaultdict(list)
        a = pd.read_csv(filename, usecols=['gmtTime','askMedian','bidMedian','askVolume','bidVolume','spreadMedian','symbol'])
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            next(reader)

            for row in reader:
                d_by_time[row[0]].append(row)
                d_by_stock[row[-1]].append(row)

        return d_by_time
# Example usage
class Strategy():
        def __init__(self):
            self.owns = {}
            self.status = status
            self.stocks = hl.get_all_tickers() #['STOCK5', 'STOCK2', 'STOCK3', 'STOCK4', 'STOCK7', 'STOCK6', 'STOCK8', 'STOCK9', 'STOCK10', 'STOCK1']
            self.stock_to_ints = {x: [] for x in self.stocks}
            self.portfolio = None
            self.balance = None
            self.historicalData = {}



        def get_portfolio(self):
            if self.status == "TEST":
                return {s : 10 for s in self.stocks}
            elif self.status == "LIVE":
                return hl.get_portfolio()

        def get_balance(self):
            if self.status == "TEST":
                return 10000
            elif self.status == "LIVE":
                return hl.get_balance()

        def get_historical(self, days=3):
            if status == "TEST":
                rows_to_keep = [i for i in range(81*days)]#wtf
                # gmtTime,askMedian,bidMedian,askVolume,bidVolume,spreadMedian,symbol
                filename = 'backtest.csv'
                data = pd.read_csv(filename, skiprows = lambda x: x not in rows_to_keep, usecols=['gmtTime', 'askMedian', 'bidMedian', 'askVolume', 'bidVolume', 'spreadMedian','symbol'])
                for c, s in enumerate(data['symbol']):
                    self.stock_to_ints[s].append(c)
                return data
            else:
                data = pd.DataFrame( hl.get_historical_data(days))
                for c,s in enumerate(data['symbol']):
                    self.stock_to_ints[s].append(c)
                return data


        def sell(self, ticker, amount, price, days_to_cancel):
            if self.status == "TEST":
                pass
            elif self.status == "LIVE":
               hl.sell(ticker=ticker, amount=amount, price=price, days_to_cancel=days_to_cancel)

        def buy(self, ticker, amount, price, days_to_cancel):
            if self.status == "TEST":
                pass
            elif self.status == "LIVE":
                hl.buy(ticker=ticker, amount=amount, price=price, days_to_cancel=days_to_cancel)

        def get_current_price(self, s):
            if status=="TEST":
                return  self.historicalData["askMedian"][self.stock_to_ints[s][-1]]
            else:
                return self.get_current_price(s)

        def calc_price(self, i):
            # Tested
            # Get necessary variables about the stock (MODIFY IF NEEDED)
            #if (i<len(self.historicalData['askMedian'])) and i (i<len(self.historicalData['bidMedian'])) and(i<len(self.historicalData['bidVolume'])) and(i<len(self.historicalData['askVolume'])):
            try:
                ask_median = self.historicalData['askMedian'][i]
                bid_median = self.historicalData['bidMedian'][i]
                bid_vol = self.historicalData['bidVolume'][i]
                ask_vol = self.historicalData['askVolume'][i]
                return (ask_median * bid_vol + bid_median * ask_vol) / (ask_vol + bid_vol)
            except KeyError:
                os.execl(sys.executable, sys.executable, *sys.argv)
            #else:
            #    print("Now we missed data")
            #    os.execl(sys.executable, sys.executable, *sys.argv)




        def getMovingaverageByCommand(self, stock, hours_back, command):
            if len(self.stock_to_ints[stock]) >= hours_back:
                L = [self.historicalData[command][i] for i in self.stock_to_ints[stock][-hours_back:]]
            else:
                L = [self.historicalData[command][i] for i in self.stock_to_ints[stock]]

            avg = sum(L) / len(L)
            return avg

        def price_moving_avg(self, stock, hours_back):
            # tested
            if len(self.stock_to_ints[stock]) >= hours_back:
                prices = [self.calc_price(i) for i in self.stock_to_ints[stock][-hours_back:]]
            else:
                prices = [self.calc_price(i) for i in self.stock_to_ints[stock]]
            avg = sum(prices) / len(prices)
            return avg

        def ask_vol_moving_avg(self, stock, hours_back):
            return self.getMovingaverageByCommand(stock, hours_back, "askVolume")

        def bid_vol_moving_avg(self, stock, hours_back):
            # tested
            return self.getMovingaverageByCommand(stock, hours_back, "bidVolume")

        def calc_stock_value(self, stock):
            price_weight = 50
            ask_vol_weight = 0
            bid_vol_weight = 0
            shortTermH = 3
            longTermH = 15
            first_term = price_weight * self.price_moving_avg(stock, shortTermH) / self.price_moving_avg(stock, longTermH)
            second_term = ask_vol_weight * self.ask_vol_moving_avg(stock, shortTermH) / self.ask_vol_moving_avg(stock, longTermH)
            third_term = bid_vol_weight * self.bid_vol_moving_avg(stock, shortTermH) / self.bid_vol_moving_avg(stock, longTermH)
            return first_term + second_term - third_term
            
        def getPortfolioVal(self):
            return sum([self.portfolio[s]*self.historicalData["askMedian"][self.stock_to_ints[s][-1]] for s in self.stocks if s in self.portfolio]) + self.balance

        def findBuyingVolume(self, findVal):
            portfolioVal = self.getPortfolioVal()
            total = sum([math.exp(findVal(s)) for s in self.stocks])
            buyingVolumes = {x : portfolioVal*math.exp(findVal(x))/total for x in self.stocks}
            return buyingVolumes

        def case1(self):
            """
            Assume we want all money on market and there are no courtage.
            :return:
            """

            N = 20
            T = 5
            C = 1


        def printUpdate(self, toBuy, toSell):
            d = {
                "PORTFOLIOVAL" : self.getPortfolioVal(),
                "balance" : self.balance,
                 "portfolio" : self.portfolio,
                 "date" : self.historicalData["gmtTime"][len(self.historicalData)-1],
                 "toSell" : toSell,
                 "toBuy" : toBuy
                 }
            for k, v in d.items():
                if k=="toSell":
                    print("---------")
                    print("To sell")
                    for k1 in v:
                        print(f"{k1}")
                    print("---------")
                elif k=="toBuy":
                    print("To buy")
                    for k1 in v:
                        print(f"{k1}")
                else:
                    print(f"{k} : {v}")
            print("____________________________________________________________________________________________________________")




        def main(self):
            while True:
                """
                Set up for each new request
                """
                t = time.time()
                self.historicalData = self.get_historical(days=3)
                self.portfolio = self.get_portfolio()
                self.balance = self.get_balance()


                buyVols = self.findBuyingVolume(findVal=self.calc_stock_value)
                stocksSortedByBuyVols = sorted(list((k, v) for k, v in buyVols.items()), key=lambda x: buyVols[x[0]])


                #print(self.getPortfolioVal()==sum([want for s, want in stocksSortedByBuyVols]))



                toSell = []
                toBuy = []
                for s, want in stocksSortedByBuyVols:
                    current_price = self.historicalData["askMedian"][self.stock_to_ints[s][-1]]
                    #current_price = self.get_current_price(s)
                    if s in self.portfolio:
                        diff = want - self.portfolio[s]*current_price

                        if diff<0:
                            numberToSell = math.floor(abs(diff / current_price))

                            toSell.append({"stock" : s,"amount" : int(numberToSell), "price" : int(current_price)})
                            self.sell(ticker=s, amount=int(numberToSell), price=int(abs(current_price)), days_to_cancel=2)

                for s, want in stocksSortedByBuyVols:
                    current_price = self.historicalData["bidMedian"][self.stock_to_ints[s][-1]]
                    #if s in toSellSet:
                    #    continue
                    #current_price = self.get_current_price(s)
                    if s in self.portfolio:
                        diff = want - self.portfolio[s] * current_price
                    else:
                        diff = want

                    if diff>0:
                        numberToBuy = math.floor(diff / current_price)
                        toBuy.append({"stock": s, "amount": int(numberToBuy), "price": int(abs(current_price))})
                        self.buy(ticker=s, amount=int(numberToBuy), price=int(current_price), days_to_cancel=2)

                calctime = time.time()-t
                PERIODTIME = 2
                self.printUpdate(toBuy, toSell)
                if calctime < PERIODTIME:
                    time.sleep(PERIODTIME-calctime)


if __name__ == "__main__":
    strat = Strategy()
    strat.main()





