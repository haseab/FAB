from trade import Trade


class TradeHistory:
    """Turning the Trading history into an object. This way, the trade information is more legible."""

    def __init__(self):
        self.allTrades = [["List of Trades"]]

    def first_trade(self):
        if len(self.allTrades) > 2:
            return Trade(self.allTrades[1])

    def last_trade(self):
        return Trade(self.allTrades[-1])

    def append(self, trade: Trade) -> bool:
        self.allTrades.append(trade)
        return True

    def convert_list(lst):
        new_list = TradeHistory()
        new_list.allTrades = lst
        return new_list

    def __len__(self):
        return len(self.allTrades)

    def __getitem__(self, item):
        return self.allTrades[item]

    def __repr__(self):
        print("[", end="")
        for i in self.allTrades[:-1]:
            print(f"{i},")
        print(f"{self.allTrades[-1]}]", end="")
        return ""
